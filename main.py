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

# Reduce transformers verbosity to avoid clutter
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


class ImageRegion(NamedTuple):
    """Represents an image region in a PDF page."""
    bbox: Tuple[float, float, float, float]
    image: Image.Image
    page_num: int
    image_id: str


class QwenOCRProcessor:
    """
    OCR processor using Qwen2.5-VL-3B for layout-preserving text extraction.
    FIXED: No more GPU warnings!
    """
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "auto"):
        """
        Initialize Qwen OCR processor.
        
        Args:
            model_path: Path to Qwen2.5-VL model
            device: Device to run model on ("auto", "cuda", "cpu")
        """
        self.logger = logging.getLogger(__name__)
        
        # Check GPU availability and set device
        if not torch.cuda.is_available():
            if device != "cpu":
                self.logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
            device_map = "cpu"
            torch_dtype = "auto"
        else:
            self.device = "cuda:0"  # FIXED: Explicit device
            device_map = {"": 0}    # FIXED: Explicit GPU mapping
            torch_dtype = torch.float16  # FIXED: Explicit dtype for GPU
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        self.logger.info("Loading Qwen2.5-VL model...")
        
        # FIXED: Updated model loading with proper device mapping
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # FIXED: Verify model placement
        model_device = next(self.model.parameters()).device
        self.logger.info(f"✅ Model loaded on: {model_device}")
        
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
        """
        FIXED: Move all inputs to the correct device to prevent warnings.
        """
        if self.device == "cpu":
            return inputs
        
        moved_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, 'to'):
                moved_inputs[key] = value.to(self.device)
            else:
                moved_inputs[key] = value
        return moved_inputs
    
    def extract_text_from_image(self, image: Image.Image, 
                               prompt_type: str = "layout_preserve") -> str:
        """
        Extract text from image using Qwen2.5-VL with layout preservation.
        FIXED: Proper device handling and generation parameters.
        
        Args:
            image: PIL Image to process
            prompt_type: Type of OCR prompt ("layout_preserve", "table_focus", "text_only")
            
        Returns:
            Extracted text in markdown format
        """
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
            
            # Updated: Use process_vision_info from qwen_vl_utils
            vision_info = process_vision_info(messages)
            image_inputs, video_inputs = vision_info[0], vision_info[1]
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # FIXED: Move inputs to correct device
            inputs = self._move_inputs_to_device(inputs)
            
            # FIXED: Generate response with proper parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    # REMOVED: temperature=0.1,  # This parameter is not supported
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return ""
        
        finally:
            # FIXED: Clean up GPU memory after each inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def batch_extract_text(self, images: List[Image.Image], 
                          prompt_type: str = "layout_preserve") -> List[str]:
        """
        Extract text from multiple images efficiently.
        FIXED: Better memory management.
        
        Args:
            images: List of PIL Images to process
            prompt_type: Type of OCR prompt
            
        Returns:
            List of extracted texts in markdown format
        """
        results = []
        for i, image in enumerate(images):
            self.logger.info(f"Processing image {i+1}/{len(images)} with OCR...")
            text = self.extract_text_from_image(image, prompt_type)
            results.append(text)
            
            # FIXED: Clear GPU cache more frequently to prevent OOM
            if torch.cuda.is_available() and (i + 1) % 3 == 0:
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                self.logger.info(f"GPU memory after {i+1} images: {memory_allocated:.2f}GB")
        
        return results


class HybridPDFConverter:
    """
    Original hybrid converter that uses:
    - PyMuPDF4LLM for text extraction and document structure
    - pdfplumber for superior table extraction
    """
    
    def __init__(self, table_strategy: str = "replace"):
        """
        Initialize hybrid converter.
        
        Args:
            table_strategy: "replace" or "enhance"
                - "replace": Replace PyMuPDF tables with pdfplumber tables
                - "enhance": Add pdfplumber tables if PyMuPDF missed them
        """
        self.table_strategy = table_strategy
        self.logger = logging.getLogger(__name__)
    
    def extract_tables_with_pdfplumber(self, pdf_path: str) -> Dict[int, List[str]]:
        """
        Extract tables from PDF using pdfplumber.
        
        Returns:
            Dictionary mapping page numbers to list of table markdown strings
        """
        tables_by_page = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = []
                
                # Extract all tables from page
                tables = page.extract_tables()
                
                for table in tables:
                    if table and len(table) > 1:  # Ensure table has header + data
                        # Convert to markdown table - handle None values in cells
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
        
        # Clean and process table data
        cleaned_table = []
        for row in table:
            cleaned_row = []
            for cell in row:
                # Clean cell content - cell is already a string from preprocessing
                cleaned_cell = cell.strip().replace('\n', ' ').replace('|', '\\|')
                cleaned_row.append(cleaned_cell)
            cleaned_table.append(cleaned_row)
        
        # Convert to DataFrame for easier markdown generation
        try:
            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
            return df.to_markdown(index=False)
        except Exception as e:
            self.logger.warning(f"Failed to convert table to markdown: {e}")
            return None
    
    def _detect_table_regions(self, text: str) -> List[Tuple[int, int]]:
        """
        Detect potential table regions in text based on patterns.
        
        Returns:
            List of (start, end) positions of potential table regions
        """
        table_regions = []
        
        # Look for table-like patterns
        lines = text.split('\n')
        in_table = False
        start_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Detect table start (multiple separators or aligned columns)
            if not in_table and (
                line.count('|') >= 2 or  # Markdown table
                line.count('\t') >= 2 or  # Tab-separated
                re.match(r'^[^|]*\|[^|]*\|[^|]*', line) or  # Pipe-separated
                re.match(r'^.{10,}\s{3,}.{10,}', line)  # Column-aligned
            ):
                in_table = True
                start_idx = i
            
            # Detect table end (empty line or different pattern)
            elif in_table and (
                not line or
                (line.count('|') < 2 and line.count('\t') < 2 and 
                 not re.match(r'^.{10,}\s{3,}.{10,}', line))
            ):
                in_table = False
                if i > start_idx + 1:  # Ensure minimum table size
                    table_regions.append((start_idx, i))
        
        return table_regions
    
    def _replace_tables_in_text(self, text: str, page_num: int, 
                               pdfplumber_tables: Dict[int, List[str]]) -> str:
        """
        Replace detected table regions with pdfplumber-extracted tables.
        """
        if page_num not in pdfplumber_tables:
            return text
        
        lines = text.split('\n')
        table_regions = self._detect_table_regions(text)
        pdfplumber_page_tables = pdfplumber_tables[page_num]
        
        # Replace table regions with pdfplumber tables
        offset = 0
        for i, (start, end) in enumerate(table_regions):
            if i < len(pdfplumber_page_tables):
                # Replace the table region
                new_lines = lines[:start-offset] + [pdfplumber_page_tables[i]] + lines[end-offset:]
                lines = new_lines
                offset += (end - start - 1)  # Adjust offset for replaced content
        
        return '\n'.join(lines)
    
    def _enhance_text_with_tables(self, text: str, page_num: int, 
                                 pdfplumber_tables: Dict[int, List[str]]) -> str:
        """
        Add pdfplumber tables to text if they don't already exist.
        """
        if page_num not in pdfplumber_tables:
            return text
        
        # Check if text already has well-formed tables
        existing_tables = len(self._detect_table_regions(text))
        pdfplumber_page_tables = pdfplumber_tables[page_num]
        
        # If pdfplumber found more tables, add them
        if len(pdfplumber_page_tables) > existing_tables:
            missing_tables = pdfplumber_page_tables[existing_tables:]
            
            # Add missing tables at the end of the page
            if missing_tables:
                text += '\n\n## Additional Tables\n\n' + '\n\n'.join(missing_tables)
        
        return text
    
    def convert_pdf_hybrid(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert PDF using hybrid approach.
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path to output markdown file (optional)
            
        Returns:
            Markdown text
        """
        # Step 1: Extract tables using pdfplumber
        self.logger.info("Extracting tables with pdfplumber...")
        pdfplumber_tables = self.extract_tables_with_pdfplumber(pdf_path)
        
        # Step 2: Extract text and basic structure using PyMuPDF4LLM
        self.logger.info("Extracting text and structure with PyMuPDF4LLM...")
        chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        
        # Step 3: Process each page
        processed_pages = []
        for i, chunk in enumerate(chunks):
            page_text = chunk['text']
            
            # Apply table strategy
            if self.table_strategy == "replace":
                page_text = self._replace_tables_in_text(page_text, i, pdfplumber_tables)
            elif self.table_strategy == "enhance":
                page_text = self._enhance_text_with_tables(page_text, i, pdfplumber_tables)
            
            processed_pages.append(page_text)
        
        # Step 4: Combine all pages
        final_markdown = '\n\n---\n\n'.join(processed_pages)
        
        # Step 5: Save if output path provided
        if output_path:
            pathlib.Path(output_path).write_text(final_markdown, encoding='utf-8')
            self.logger.info(f"Hybrid markdown saved to: {output_path}")
        
        return final_markdown


class EnhancedHybridPDFConverter(HybridPDFConverter):
    """
    Enhanced hybrid converter with Qwen2.5-VL OCR integration.
    ENHANCED: Now properly replaces image placeholders with OCR text!
    """
    
    def __init__(self, table_strategy: str = "replace", 
                 ocr_strategy: str = "smart",
                 min_image_size: Tuple[int, int] = (100, 100),
                 ocr_confidence_threshold: float = 0.7):
        """
        Initialize enhanced converter with OCR capabilities.
        
        Args:
            table_strategy: "replace" or "enhance" 
            ocr_strategy: "always", "smart", "fallback", or "never"
            min_image_size: Minimum image size (width, height) to process
            ocr_confidence_threshold: Threshold for text extraction quality
        """
        super().__init__(table_strategy)
        self.ocr_strategy = ocr_strategy
        self.min_image_size = min_image_size
        self.ocr_confidence_threshold = ocr_confidence_threshold
        
        # Initialize OCR processor if needed
        self.ocr_processor = None
        if ocr_strategy != "never":
            self.logger.info("Initializing ENHANCED OCR processor...")
            self.ocr_processor = QwenOCRProcessor()
    
    def extract_images_from_pdf(self, pdf_path: str) -> Dict[int, List[ImageRegion]]:
        """
        Extract images from PDF pages.
        
        Returns:
            Dictionary mapping page numbers to list of ImageRegion objects
        """
        images_by_page = {}
        
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            page_images = []
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Convert to PIL Image
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("ppm")
                        pil_image = Image.open(io.BytesIO(img_data))
                    else:  # CMYK: convert to RGB first
                        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix_rgb.tobytes("ppm")
                        pil_image = Image.open(io.BytesIO(img_data))
                        pix_rgb = None
                    
                    # Check minimum size
                    if (pil_image.width >= self.min_image_size[0] and 
                        pil_image.height >= self.min_image_size[1]):
                        
                        # Get image bbox on page using xref
                        try:
                            image_rects = page.get_image_rects(xref)
                            if image_rects:
                                bbox = tuple(image_rects[0])  # Convert to tuple
                            else:
                                bbox = (0.0, 0.0, float(page.rect.width), float(page.rect.height))
                        except Exception as e:
                            # Fallback: use full page dimensions if get_image_rects fails
                            self.logger.debug(f"Warning: Failed to get image rects for xref {xref} on page {page_num}: {e}")
                            bbox = (0.0, 0.0, float(page.rect.width), float(page.rect.height))
                        
                        # Create ImageRegion
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
        """
        Assess the quality of extracted text.
        
        Returns:
            Quality score between 0 and 1
        """
        if not text or len(text.strip()) < 10:
            return 0.0
        
        # Simple quality metrics
        total_chars = len(text)
        
        # Count problematic characters
        problematic = text.count('�') + text.count('□') + text.count('?')
        
        # Count alphanumeric characters
        alphanumeric = sum(1 for c in text if c.isalnum())
        
        # Calculate quality score
        if total_chars == 0:
            return 0.0
        
        problem_ratio = problematic / total_chars
        alphanumeric_ratio = alphanumeric / total_chars
        
        quality_score = max(0, alphanumeric_ratio - problem_ratio)
        return min(1.0, quality_score)
    
    def _should_use_ocr_for_page(self, page_text: str, page_images: List[ImageRegion]) -> bool:
        """
        Determine if OCR should be used for a page based on strategy and content.
        """
        if self.ocr_strategy == "never":
            return False
        elif self.ocr_strategy == "always":
            return bool(page_images)
        elif self.ocr_strategy == "smart":
            # Use OCR if text quality is poor or significant images exist
            text_quality = self._assess_text_quality(page_text)
            has_significant_images = len(page_images) > 0
            
            return (text_quality < self.ocr_confidence_threshold or 
                   (has_significant_images and text_quality < 0.9))
        elif self.ocr_strategy == "fallback":
            # Only use OCR if native text extraction is very poor
            text_quality = self._assess_text_quality(page_text)
            return text_quality < 0.3
        
        return False
    
    def _determine_ocr_prompt_type(self, image: Image.Image, context: str = "") -> str:
        """
        Determine the best OCR prompt type based on image characteristics.
        """
        # Simple heuristic - can be enhanced with image analysis
        width, height = image.size
        aspect_ratio = width / height
        
        # If context suggests tables, use table-focused OCR
        if any(keyword in context.lower() for keyword in ['table', 'data', 'column', 'row']):
            return "table_focus"
        
        # If image is very wide, likely contains tabular data
        if aspect_ratio > 2.0:
            return "table_focus"
        
        # Default to layout preserving
        return "layout_preserve"
    
    def _find_image_placeholders(self, text: str) -> List[Dict]:
        """
        Find image placeholders in the markdown text.
        
        Returns:
            List of dictionaries with placeholder info: {'match': match_obj, 'start': int, 'end': int, 'type': str}
        """
        placeholders = []
        
        # Common image placeholder patterns
        patterns = [
            # Markdown image syntax: ![alt](src), ![alt text](image.png), ![](image.jpg)
            (r'!\[[^\]]*\]\([^)]*\)', 'markdown_image'),
            
            # HTML img tags: <img src="..." />, <img ...>
            (r'<img[^>]*/?>', 'html_image'),
            
            # Figure references: [Figure 1], (Fig. 1), etc.
            (r'\[(?:Figure|Fig\.?)\s*\d+\]', 'figure_ref'),
            (r'\((?:Figure|Fig\.?)\s*\d+\)', 'figure_ref_paren'),
            
            # Image file references: image.png, picture.jpg, etc.
            (r'\b\w+\.(png|jpg|jpeg|gif|bmp|tiff|svg)\b', 'file_ref'),
            
            # Empty lines that might represent image placeholders (common in PDF extraction)
            (r'\n\s*\n\s*\n', 'empty_space'),
            
            # Single character or symbols that might be image placeholders
            (r'^\s*[▪◾▴◾◼◾▫◾▮◾█◾]\s*$', 'symbol_placeholder'),
            
            # Mathematical notation that might be an image (LaTeX-style)
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
        
        # Sort by position in text
        placeholders.sort(key=lambda x: x['start'])
        return placeholders
    
    def _match_images_to_placeholders(self, placeholders: List[Dict], 
                                    ocr_results: List[Tuple[ImageRegion, str]]) -> List[Tuple[Dict, str]]:
        """
        Match OCR results to placeholders based on position and context.
        
        Returns:
            List of (placeholder_dict, ocr_text) tuples
        """
        matches = []
        
        # Simple strategy: match in order of appearance
        # More sophisticated matching could use bbox coordinates
        
        used_ocr_indices = set()
        
        for placeholder in placeholders:
            # Skip empty space placeholders unless we have unused OCR results
            if placeholder['type'] == 'empty_space':
                continue
                
            # Find the best matching OCR result
            best_match_idx = None
            
            for i, (image_region, ocr_text) in enumerate(ocr_results):
                if i in used_ocr_indices:
                    continue
                    
                # If OCR text is substantial, it's a good match
                if ocr_text.strip() and len(ocr_text.strip()) > 10:
                    best_match_idx = i
                    break
            
            if best_match_idx is not None:
                _, ocr_text = ocr_results[best_match_idx]
                matches.append((placeholder, ocr_text))
                used_ocr_indices.add(best_match_idx)
        
        # Handle remaining OCR results that didn't match placeholders
        remaining_ocr = [
            (image_region, ocr_text) for i, (image_region, ocr_text) in enumerate(ocr_results)
            if i not in used_ocr_indices and ocr_text.strip()
        ]
        
        return matches, remaining_ocr
    
    def _integrate_ocr_results(self, original_text: str, 
                              ocr_results: List[Tuple[ImageRegion, str]]) -> str:
        """
        ENHANCED: Integrate OCR results by replacing image placeholders in the original text.
        """
        if not ocr_results:
            return original_text
        
        self.logger.info(f"Integrating {len(ocr_results)} OCR results into text...")
        
        # Step 1: Find image placeholders in the text
        placeholders = self._find_image_placeholders(original_text)
        self.logger.info(f"Found {len(placeholders)} potential image placeholders")
        
        # Step 2: Match OCR results to placeholders
        matches, remaining_ocr = self._match_images_to_placeholders(placeholders, ocr_results)
        self.logger.info(f"Matched {len(matches)} placeholders with OCR results")
        
        # Step 3: Replace placeholders with OCR text (in reverse order to maintain positions)
        integrated_text = original_text
        
        for placeholder, ocr_text in reversed(matches):
            start = placeholder['start']
            end = placeholder['end']
            
            # Format OCR text nicely
            formatted_ocr = self._format_ocr_text(ocr_text, placeholder['type'])
            
            # Replace the placeholder
            integrated_text = integrated_text[:start] + formatted_ocr + integrated_text[end:]
            self.logger.debug(f"Replaced placeholder '{placeholder['text'][:50]}...' with OCR text")
        
        # Step 4: Add any remaining OCR results that couldn't be matched to placeholders
        if remaining_ocr:
            integrated_text += "\n\n## Additional Image Content (OCR)\n\n"
            for image_region, ocr_text in remaining_ocr:
                if ocr_text.strip():
                    integrated_text += f"\n### {image_region.image_id}\n\n"
                    integrated_text += ocr_text + "\n\n"
        
        return integrated_text
    
    def _format_ocr_text(self, ocr_text: str, placeholder_type: str) -> str:
        """
        Format OCR text appropriately based on the placeholder type.
        """
        if not ocr_text.strip():
            return ""
        
        # Clean up the OCR text
        cleaned_text = ocr_text.strip()
        
        # Add appropriate formatting based on placeholder type
        if placeholder_type in ['markdown_image', 'html_image']:
            # For image syntax, just replace with the text
            return cleaned_text
        elif placeholder_type == 'math_notation':
            # Keep math notation format if it looks like LaTeX
            if cleaned_text.startswith('$') and cleaned_text.endswith('$'):
                return cleaned_text
            else:
                return f"${cleaned_text}$"
        elif placeholder_type in ['figure_ref', 'figure_ref_paren']:
            # For figure references, add a small header
            return f"\n**Figure Content:**\n{cleaned_text}\n"
        else:
            # Default formatting
            return f"\n{cleaned_text}\n"
    
    def convert_pdf_with_ocr(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert PDF with ENHANCED OCR integration that replaces image placeholders.
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path to output markdown file (optional)
            
        Returns:
            Markdown text with OCR content properly integrated
        """
        # Step 1: Extract images if OCR is enabled
        images_by_page = {}
        if self.ocr_strategy != "never":
            self.logger.info("Extracting images from PDF...")
            images_by_page = self.extract_images_from_pdf(pdf_path)
        
        # Step 2: Extract tables using pdfplumber
        self.logger.info("Extracting tables with pdfplumber...")
        pdfplumber_tables = self.extract_tables_with_pdfplumber(pdf_path)
        
        # Step 3: Extract text and structure using PyMuPDF4LLM
        self.logger.info("Extracting text and structure with PyMuPDF4LLM...")
        chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        
        # Step 4: Process each page with ENHANCED OCR integration
        processed_pages = []
        for i, chunk in enumerate(chunks):
            page_text = chunk['text']
            
            # Apply table strategy
            if self.table_strategy == "replace":
                page_text = self._replace_tables_in_text(page_text, i, pdfplumber_tables)
            elif self.table_strategy == "enhance":
                page_text = self._enhance_text_with_tables(page_text, i, pdfplumber_tables)
            
            # ENHANCED OCR integration
            page_images = images_by_page.get(i, [])
            if self._should_use_ocr_for_page(page_text, page_images):
                self.logger.info(f"Applying ENHANCED OCR to page {i+1} with {len(page_images)} images...")
                
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
                
                # ENHANCED: Replace placeholders with OCR results
                page_text = self._integrate_ocr_results(page_text, ocr_results)
            
            processed_pages.append(page_text)
        
        # Step 5: Combine all pages
        final_markdown = '\n\n---\n\n'.join(processed_pages)
        
        # Step 6: Save if output path provided
        if output_path:
            pathlib.Path(output_path).write_text(final_markdown, encoding='utf-8')
            self.logger.info(f"ENHANCED markdown with integrated OCR saved to: {output_path}")
        
        return final_markdown


def convert_pdf_hybrid(pdf_path: str, output_path: Optional[str] = None, 
                      strategy: str = "replace") -> str:
    """
    Simple function interface for hybrid PDF conversion (original).
    
    Args:
        pdf_path: Path to input PDF
        output_path: Path to output markdown file (optional)
        strategy: "replace", "enhance", or "table_focused"
    
    Returns:
        Markdown text
    """
    converter = HybridPDFConverter(table_strategy=strategy)
    return converter.convert_pdf_hybrid(pdf_path, output_path)


def convert_pdf_with_ocr(pdf_path: str, output_path: Optional[str] = None, 
                        table_strategy: str = "replace",
                        ocr_strategy: str = "smart") -> str:
    """
    Simple function interface for ENHANCED OCR-integrated PDF conversion.
    
    Args:
        pdf_path: Path to input PDF
        output_path: Path to output markdown file (optional)
        table_strategy: "replace" or "enhance"
        ocr_strategy: "always", "smart", "fallback", or "never"
    
    Returns:
        Markdown text with OCR content properly integrated (not appended)
    """
    converter = EnhancedHybridPDFConverter(
        table_strategy=table_strategy,
        ocr_strategy=ocr_strategy
    )
    
    return converter.convert_pdf_with_ocr(pdf_path, output_path)


def main():
    """
    Main function demonstrating ENHANCED OCR integration.
    """
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 ENHANCED PDF Converter with Smart OCR Integration")
    print("✨ Now replaces image placeholders with OCR text!")
    print("=" * 60)
    
    # Check GPU status
    if torch.cuda.is_available():
        print(f"🔥 GPU available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️  No GPU available - will run on CPU")
    
    # Test file (replace with your PDF)
    pdf_file = "sample.pdf"

    try:
        print(f"\n🔄 Processing {pdf_file} with ENHANCED OCR integration...")
        result = convert_pdf_with_ocr(
            pdf_path=pdf_file,
            output_path="output_enhanced_ocr.md",
            table_strategy="replace",
            ocr_strategy="smart"
        )
        print("✅ ENHANCED OCR conversion completed successfully!")
        print("📄 Output saved to: output_enhanced_ocr.md")
        print(f"📊 Result length: {len(result)} characters")
        print("\n🎯 Key improvements:")
        print("  • OCR text now replaces image placeholders instead of being appended")
        print("  • Smarter matching of images to placeholders")
        print("  • Better formatting of integrated OCR content")
        print("  • Fallback handling for unmatched content")
        
    except FileNotFoundError:
        print(f"❌ File {pdf_file} not found. Please update the path to your PDF file.")
    except Exception as e:
        print(f"❌ Error in conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()