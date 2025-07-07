import pathlib
import pymupdf4llm
import pdfplumber
import pandas as pd
import re
import fitz
import io
from PIL import Image
from typing import List, Dict, Optional, Tuple, NamedTuple
import logging
import os

# Updated imports for Qwen2.5-VL
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Reduce transformers verbosity
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


class ImageRegion(NamedTuple):
    """Represents an image region in a PDF page."""
    bbox: Tuple[float, float, float, float]
    image: Image.Image
    page_num: int
    image_id: str


class QwenOCRProcessor:
    """OCR processor using Qwen2.5-VL-3B for layout-preserving text extraction."""
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "auto"):
        """Initialize Qwen OCR processor."""
        self.logger = logging.getLogger(__name__)
        
        # Check GPU availability and set device
        if not torch.cuda.is_available():
            if device != "cpu":
                self.logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
            device_map = "cpu"
            torch_dtype = "auto"
        else:
            self.device = "cuda:0"
            device_map = {"": 0}
            torch_dtype = torch.float16
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        self.logger.info("Loading Qwen2.5-VL model...")
        
        # Load model with proper device mapping
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load processor with warning fixes
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True,
                use_fast=True
            )
            self.logger.info("Fast processor loaded successfully")
        except Exception as e:
            self.logger.warning(f"Fast processor failed, falling back to slow: {e}")
            self.processor = AutoProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True,
                use_fast=False
            )
            self.logger.info("Slow processor loaded as fallback")
        
        # Verify model placement
        model_device = next(self.model.parameters()).device
        self.logger.info(f"Model loaded on: {model_device}")
        
        # OCR prompts for different scenarios
        self.ocr_prompts = {
            "layout_preserve": """Analyze this image and extract all text while preserving the exact layout and formatting. 
Convert the content to markdown format, maintaining:
- Headers and subheaders (use #, ##, ###)
- Tables (use markdown table format)
- Lists (use - or 1. format)
- Text alignment and spacing
- Any special formatting like bold or italic text

Return only the markdown content, no explanations.""",
            
            "table_focus": """This image contains tabular data. Extract the table(s) and convert to markdown table format.
Preserve column alignment, headers, and data relationships. If multiple tables exist, separate them clearly.
Return only the markdown table(s), no explanations.""",
            
            "text_only": """Extract all text from this image in reading order. 
Convert to clean markdown format with appropriate headers and formatting.
Return only the text content, no explanations."""
        }
    
    def _move_inputs_to_device(self, inputs):
        """Move all inputs to the correct device without breaking object structure."""
        if self.device == "cpu":
            return inputs
        
        try:
            return inputs.to(self.device)
        except Exception as e:
            self.logger.debug(f"Whole object move failed: {e}, trying individual tensors...")
            try:
                for key in inputs:
                    if hasattr(inputs[key], 'to'):
                        inputs[key] = inputs[key].to(self.device)
                return inputs
            except Exception as e2:
                self.logger.warning(f"Failed to move tensors to device: {e2}")
                return inputs
    
    def extract_text_from_image(self, image: Image.Image, 
                               prompt_type: str = "layout_preserve") -> str:
        """Extract text from image using Qwen2.5-VL with layout preservation."""
        try:
            # Prepare the conversation format expected by Qwen
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.ocr_prompts[prompt_type]}
                    ]
                }
            ]
            
            # Process the input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Use process_vision_info from qwen_vl_utils
            vision_info = process_vision_info(messages)
            image_inputs, video_inputs = vision_info[0], vision_info[1]
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to correct device (preserves object structure)
            inputs = self._move_inputs_to_device(inputs)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Robust access to input_ids
            try:
                input_ids = inputs.input_ids
            except AttributeError:
                try:
                    input_ids = inputs['input_ids']
                except KeyError:
                    self.logger.error("Cannot find input_ids in inputs object")
                    return ""
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return ""
        
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def batch_extract_text(self, images: List[Image.Image], 
                          prompt_type: str = "layout_preserve") -> List[str]:
        """Extract text from multiple images efficiently."""
        results = []
        for i, image in enumerate(images):
            self.logger.info(f"Processing image {i+1}/{len(images)} with OCR...")
            text = self.extract_text_from_image(image, prompt_type)
            results.append(text)
            
            # Clear GPU cache more frequently to prevent OOM
            if torch.cuda.is_available() and (i + 1) % 3 == 0:
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                self.logger.info(f"GPU memory after {i+1} images: {memory_allocated:.2f}GB")
        
        return results


class HybridPDFConverter:
    """
    Hybrid converter that uses:
    - PyMuPDF4LLM for text extraction and document structure
    - pdfplumber for superior table extraction
    """
    
    def __init__(self, table_strategy: str = "replace"):
        """Initialize hybrid converter."""
        self.table_strategy = table_strategy
        self.logger = logging.getLogger(__name__)
    
    def extract_tables_with_pdfplumber(self, pdf_path: str) -> Dict[int, List[str]]:
        """Extract tables from PDF using pdfplumber."""
        tables_by_page = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = []
                tables = page.extract_tables()
                
                for table in tables:
                    if table and len(table) > 1:
                        typed_table: List[List[str]] = [
                            [str(cell) if cell is not None else "" for cell in row]
                            for row in table
                        ]
                        markdown_table = self._table_to_markdown(typed_table)
                        if markdown_table:
                            page_tables.append(markdown_table)
                
                if page_tables:
                    tables_by_page[page_num] = page_tables
        
        return tables_by_page
    
    def _table_to_markdown(self, table: List[List[str]]) -> Optional[str]:
        """Convert table data to markdown format."""
        if not table or len(table) < 2:
            return None
        
        cleaned_table = []
        for row in table:
            cleaned_row = []
            for cell in row:
                cleaned_cell = cell.strip().replace('\n', ' ').replace('|', '\\|')
                cleaned_row.append(cleaned_cell)
            cleaned_table.append(cleaned_row)
        
        try:
            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
            return df.to_markdown(index=False)
        except Exception as e:
            self.logger.warning(f"Failed to convert table to markdown: {e}")
            return None
    
    def _detect_table_regions(self, text: str) -> List[Tuple[int, int]]:
        """Detect potential table regions in text based on patterns."""
        table_regions = []
        lines = text.split('\n')
        in_table = False
        start_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if not in_table and (
                line.count('|') >= 2 or
                line.count('\t') >= 2 or
                re.match(r'^[^|]*\|[^|]*\|[^|]*', line) or
                re.match(r'^.{10,}\s{3,}.{10,}', line)
            ):
                in_table = True
                start_idx = i
            
            elif in_table and (
                not line or
                (line.count('|') < 2 and line.count('\t') < 2 and 
                 not re.match(r'^.{10,}\s{3,}.{10,}', line))
            ):
                in_table = False
                if i > start_idx + 1:
                    table_regions.append((start_idx, i))
        
        return table_regions
    
    def _replace_tables_in_text(self, text: str, page_num: int, 
                               pdfplumber_tables: Dict[int, List[str]]) -> str:
        """Replace detected table regions with pdfplumber-extracted tables."""
        if page_num not in pdfplumber_tables:
            return text
        
        lines = text.split('\n')
        table_regions = self._detect_table_regions(text)
        pdfplumber_page_tables = pdfplumber_tables[page_num]
        
        offset = 0
        for i, (start, end) in enumerate(table_regions):
            if i < len(pdfplumber_page_tables):
                new_lines = lines[:start-offset] + [pdfplumber_page_tables[i]] + lines[end-offset:]
                lines = new_lines
                offset += (end - start - 1)
        
        return '\n'.join(lines)
    
    def _enhance_text_with_tables(self, text: str, page_num: int, 
                                 pdfplumber_tables: Dict[int, List[str]]) -> str:
        """Add pdfplumber tables to text if they don't already exist."""
        if page_num not in pdfplumber_tables:
            return text
        
        existing_tables = len(self._detect_table_regions(text))
        pdfplumber_page_tables = pdfplumber_tables[page_num]
        
        if len(pdfplumber_page_tables) > existing_tables:
            missing_tables = pdfplumber_page_tables[existing_tables:]
            if missing_tables:
                text += '\n\n## Additional Tables\n\n' + '\n\n'.join(missing_tables)
        
        return text
    
    def convert_pdf_hybrid(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """Convert PDF using hybrid approach."""
        self.logger.info("Extracting tables with pdfplumber...")
        pdfplumber_tables = self.extract_tables_with_pdfplumber(pdf_path)
        
        self.logger.info("Extracting text and structure with PyMuPDF4LLM...")
        chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        
        processed_pages = []
        for i, chunk in enumerate(chunks):
            page_text = chunk['text']
            
            if self.table_strategy == "replace":
                page_text = self._replace_tables_in_text(page_text, i, pdfplumber_tables)
            elif self.table_strategy == "enhance":
                page_text = self._enhance_text_with_tables(page_text, i, pdfplumber_tables)
            
            processed_pages.append(page_text)
        
        final_markdown = '\n\n---\n\n'.join(processed_pages)
        
        if output_path:
            pathlib.Path(output_path).write_text(final_markdown, encoding='utf-8')
            self.logger.info(f"Hybrid markdown saved to: {output_path}")
        
        return final_markdown


class EnhancedHybridPDFConverter(HybridPDFConverter):
    """Enhanced hybrid converter with Qwen2.5-VL OCR integration."""
    
    def __init__(self, table_strategy: str = "replace", 
                 ocr_strategy: str = "smart",
                 min_image_size: Tuple[int, int] = (100, 100),
                 ocr_confidence_threshold: float = 0.7):
        """Initialize enhanced converter with OCR capabilities."""
        super().__init__(table_strategy)
        self.ocr_strategy = ocr_strategy
        self.min_image_size = min_image_size
        self.ocr_confidence_threshold = ocr_confidence_threshold
        
        self.ocr_processor = None
        if ocr_strategy != "never":
            self.logger.info("Initializing OCR processor...")
            self.ocr_processor = QwenOCRProcessor()
    
    def extract_images_from_pdf(self, pdf_path: str) -> Dict[int, List[ImageRegion]]:
        """Extract images from PDF pages."""
        images_by_page = {}
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            page_images = []
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("ppm")
                        pil_image = Image.open(io.BytesIO(img_data))
                    else:
                        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix_rgb.tobytes("ppm")
                        pil_image = Image.open(io.BytesIO(img_data))
                        pix_rgb = None
                    
                    if (pil_image.width >= self.min_image_size[0] and 
                        pil_image.height >= self.min_image_size[1]):
                        
                        try:
                            image_rects = page.get_image_rects(xref)
                            if image_rects:
                                bbox = tuple(image_rects[0])
                            else:
                                bbox = (0.0, 0.0, float(page.rect.width), float(page.rect.height))
                        except Exception as e:
                            self.logger.debug(f"Failed to get image rects for xref {xref} on page {page_num}: {e}")
                            bbox = (0.0, 0.0, float(page.rect.width), float(page.rect.height))
                        
                        image_region = ImageRegion(
                            bbox=bbox,
                            image=pil_image,
                            page_num=page_num,
                            image_id=f"page_{page_num}_img_{img_index}"
                        )
                        page_images.append(image_region)
                    
                    pix = None
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                    continue
            
            if page_images:
                images_by_page[page_num] = page_images
        
        doc.close()
        return images_by_page
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess the quality of extracted text."""
        if not text or len(text.strip()) < 10:
            return 0.0
        
        total_chars = len(text)
        problematic = text.count('�') + text.count('□') + text.count('?')
        alphanumeric = sum(1 for c in text if c.isalnum())
        
        if total_chars == 0:
            return 0.0
        
        problem_ratio = problematic / total_chars
        alphanumeric_ratio = alphanumeric / total_chars
        quality_score = max(0, alphanumeric_ratio - problem_ratio)
        return min(1.0, quality_score)
    
    def _should_use_ocr_for_page(self, page_text: str, page_images: List[ImageRegion]) -> bool:
        """Determine if OCR should be used for a page based on strategy and content."""
        if self.ocr_strategy == "never":
            return False
        elif self.ocr_strategy == "always":
            return bool(page_images)
        elif self.ocr_strategy == "smart":
            text_quality = self._assess_text_quality(page_text)
            has_significant_images = len(page_images) > 0
            return (text_quality < self.ocr_confidence_threshold or 
                   (has_significant_images and text_quality < 0.9))
        elif self.ocr_strategy == "fallback":
            text_quality = self._assess_text_quality(page_text)
            return text_quality < 0.3
        
        return False
    
    def _determine_ocr_prompt_type(self, image: Image.Image, context: str = "") -> str:
        """Determine the best OCR prompt type based on image characteristics."""
        width, height = image.size
        aspect_ratio = width / height
        
        if any(keyword in context.lower() for keyword in ['table', 'data', 'column', 'row']):
            return "table_focus"
        
        if aspect_ratio > 2.0:
            return "table_focus"
        
        return "layout_preserve"
    
    def _find_image_placeholders(self, text: str) -> List[Dict]:
        """Find image placeholders in the markdown text."""
        placeholders = []
        
        patterns = [
            (r'!\[[^\]]*\]\([^)]*\)', 'markdown_image'),
            (r'<img[^>]*/?>', 'html_image'),
            (r'\[(?:Figure|Fig\.?)\s*\d+\]', 'figure_ref'),
            (r'\((?:Figure|Fig\.?)\s*\d+\)', 'figure_ref_paren'),
            (r'\b\w+\.(png|jpg|jpeg|gif|bmp|tiff|svg)\b', 'file_ref'),
            (r'\$[^$]+\$', 'math_notation'),
        ]
        
        for pattern, placeholder_type in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                placeholders.append({
                    'match': match,
                    'start': match.start(),
                    'end': match.end(),
                    'type': placeholder_type,
                    'text': match.group()
                })
        
        placeholders.sort(key=lambda x: x['start'])
        return placeholders
    
    def _match_images_to_placeholders(self, placeholders: List[Dict], 
                                    ocr_results: List[Tuple[ImageRegion, str]]) -> List[Tuple[Dict, str]]:
        """Match OCR results to placeholders based on position and context."""
        matches = []
        used_ocr_indices = set()
        
        for placeholder in placeholders:
            if placeholder['type'] == 'empty_space':
                continue
                
            best_match_idx = None
            for i, (image_region, ocr_text) in enumerate(ocr_results):
                if i in used_ocr_indices:
                    continue
                    
                if ocr_text.strip() and len(ocr_text.strip()) > 10:
                    best_match_idx = i
                    break
            
            if best_match_idx is not None:
                _, ocr_text = ocr_results[best_match_idx]
                matches.append((placeholder, ocr_text))
                used_ocr_indices.add(best_match_idx)
        
        remaining_ocr = [
            (image_region, ocr_text) for i, (image_region, ocr_text) in enumerate(ocr_results)
            if i not in used_ocr_indices and ocr_text.strip()
        ]
        
        return matches, remaining_ocr
    
    def _integrate_ocr_results(self, original_text: str, 
                              ocr_results: List[Tuple[ImageRegion, str]]) -> str:
        """Integrate OCR results by replacing image placeholders in the original text."""
        if not ocr_results:
            return original_text
        
        self.logger.info(f"Integrating {len(ocr_results)} OCR results into text...")
        
        placeholders = self._find_image_placeholders(original_text)
        self.logger.info(f"Found {len(placeholders)} potential image placeholders")
        
        matches, remaining_ocr = self._match_images_to_placeholders(placeholders, ocr_results)
        self.logger.info(f"Matched {len(matches)} placeholders with OCR results")
        
        integrated_text = original_text
        
        for placeholder, ocr_text in reversed(matches):
            start = placeholder['start']
            end = placeholder['end']
            formatted_ocr = self._format_ocr_text(ocr_text, placeholder['type'])
            integrated_text = integrated_text[:start] + formatted_ocr + integrated_text[end:]
            self.logger.debug("Replaced placeholder with OCR text")
        
        if remaining_ocr:
            integrated_text += "\n\n## Additional Image Content (OCR)\n\n"
            for image_region, ocr_text in remaining_ocr:
                if ocr_text.strip():
                    integrated_text += f"\n### {image_region.image_id}\n\n"
                    integrated_text += ocr_text + "\n\n"
        
        return integrated_text
    
    def _format_ocr_text(self, ocr_text: str, placeholder_type: str) -> str:
        """Format OCR text appropriately based on the placeholder type."""
        if not ocr_text.strip():
            return ""
        
        cleaned_text = ocr_text.strip()
        
        if placeholder_type in ['markdown_image', 'html_image']:
            return cleaned_text
        elif placeholder_type == 'math_notation':
            if cleaned_text.startswith('$') and cleaned_text.endswith('$'):
                return cleaned_text
            else:
                return f"${cleaned_text}$"
        elif placeholder_type in ['figure_ref', 'figure_ref_paren']:
            return f"\n**Figure Content:**\n{cleaned_text}\n"
        else:
            return f"\n{cleaned_text}\n"
    
    def convert_pdf_with_ocr(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """Convert PDF with OCR integration that replaces image placeholders."""
        images_by_page = {}
        if self.ocr_strategy != "never":
            self.logger.info("Extracting images from PDF...")
            images_by_page = self.extract_images_from_pdf(pdf_path)
        
        self.logger.info("Extracting tables with pdfplumber...")
        pdfplumber_tables = self.extract_tables_with_pdfplumber(pdf_path)
        
        self.logger.info("Extracting text and structure with PyMuPDF4LLM...")
        chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        
        processed_pages = []
        for i, chunk in enumerate(chunks):
            page_text = chunk['text']
            
            if self.table_strategy == "replace":
                page_text = self._replace_tables_in_text(page_text, i, pdfplumber_tables)
            elif self.table_strategy == "enhance":
                page_text = self._enhance_text_with_tables(page_text, i, pdfplumber_tables)
            
            page_images = images_by_page.get(i, [])
            if self._should_use_ocr_for_page(page_text, page_images):
                self.logger.info(f"Applying OCR to page {i+1} with {len(page_images)} images...")
                
                ocr_results = []
                for image_region in page_images:
                    prompt_type = self._determine_ocr_prompt_type(
                        image_region.image, page_text
                    )
                    ocr_text = self.ocr_processor.extract_text_from_image(
                        image_region.image, prompt_type
                    )
                    if ocr_text.strip():
                        ocr_results.append((image_region, ocr_text))
                        self.logger.info(f"OCR extracted {len(ocr_text)} characters from {image_region.image_id}")
                
                page_text = self._integrate_ocr_results(page_text, ocr_results)
            
            processed_pages.append(page_text)
        
        final_markdown = '\n\n---\n\n'.join(processed_pages)
        
        if output_path:
            pathlib.Path(output_path).write_text(final_markdown, encoding='utf-8')
            self.logger.info(f"Enhanced markdown with integrated OCR saved to: {output_path}")
        
        return final_markdown


def convert_pdf_hybrid(pdf_path: str, output_path: Optional[str] = None, 
                      strategy: str = "replace") -> str:
    """Simple function interface for hybrid PDF conversion."""
    converter = HybridPDFConverter(table_strategy=strategy)
    return converter.convert_pdf_hybrid(pdf_path, output_path)


def convert_pdf_with_ocr(pdf_path: str, output_path: Optional[str] = None, 
                        table_strategy: str = "replace",
                        ocr_strategy: str = "smart") -> str:
    """Simple function interface for OCR-integrated PDF conversion."""
    converter = EnhancedHybridPDFConverter(
        table_strategy=table_strategy,
        ocr_strategy=ocr_strategy
    )
    
    return converter.convert_pdf_with_ocr(pdf_path, output_path)


def main():
    """Main function demonstrating OCR integration."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Enhanced PDF Converter with Smart OCR Integration")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("No GPU available - will run on CPU")
    
    pdf_file = "sample.pdf"

    try:
        print(f"\nProcessing {pdf_file} with OCR integration...")
        result = convert_pdf_with_ocr(
            pdf_path=pdf_file,
            output_path="output_enhanced_ocr.md",
            table_strategy="replace",
            ocr_strategy="smart"
        )
        print("OCR conversion completed successfully!")
        print("Output saved to: output_enhanced_ocr.md")
        print(f"Result length: {len(result)} characters")
        
    except FileNotFoundError:
        print(f"File {pdf_file} not found. Please update the path to your PDF file.")
    except Exception as e:
        print(f"Error in conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()