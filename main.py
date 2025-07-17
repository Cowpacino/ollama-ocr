import pathlib
import pymupdf4llm
import pdfplumber
import pandas as pd
import re
import base64
import os
import time
from typing import List, Dict, Optional, Tuple, Any
import logging

# LangChain imports
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


class LangChainOCRProcessor(Runnable[Dict[str, Any], str]):
    """
    LangChain-compatible OCR processor using Ollama VLM models.
    Fixed for qwen2.5vl repetition and prompt contamination issues.
    """
    
    def __init__(self, 
                 model: str = "qwen2.5vl:7b",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.3,  # Increased from 0.1 to reduce repetition and reduced from 0.8 to eliminate randomness
                 skip_failed_images: bool = True):
        """
        Initialize LangChain OCR processor with fixes for qwen2.5vl issues.
        
        Args:
            model: Ollama VLM model name (default: qwen2.5vl:7b)
            base_url: Ollama server URL
            temperature: Model temperature (0.3 recommended for qwen2.5vl)
            skip_failed_images: Whether to skip failed images
        """
        self.logger = logging.getLogger(__name__)
        self.skip_failed_images = skip_failed_images
        
        # Optimized settings for qwen2.5vl to prevent repetition
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            top_p=0.9,  # Nucleus sampling to reduce repetition
            repeat_penalty=1.15,  # Higher penalty for repetition
            repeat_last_n=128,  # Look back further to prevent repetition
            num_ctx=32768,  # Increased context window (from default 2048)
            num_predict=1024,  # Limit response length
            stop=["<|im_end|>", "<|endoftext|>", "```", "\n\n---", "**Extracted OCR text:**"],  # Stop tokens
            # Additional parameters for stability
            top_k=40,
            seed=42,  # For reproducibility
            num_thread=None,  # Auto-detect threads
        )
        
        # Simplified prompt to prevent contamination
        self.ocr_prompt = self._create_ocr_prompt()
        
        # Create the processing chain
        self.chain = self.ocr_prompt | self.llm | StrOutputParser()
    
    def _create_ocr_prompt(self) -> ChatPromptTemplate:
        """Create optimized OCR prompt template to prevent contamination."""
        
        # Simplified system prompt to prevent leakage
        system_prompt = """You are an OCR assistant. Extract all visible text from the image accurately. 
Output only the extracted text with no explanations or formatting."""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", [
                {"type": "text", "text": "Extract all text from this image:"},
                {"type": "image", "source_type": "base64", "data": "{image_data}", "mime_type": "image/png"}
            ])
        ])
    
    def invoke(self, 
               input: Dict[str, Any], 
               config: Optional[RunnableConfig] = None,
               **kwargs) -> Any:
        """
        Process a single image with fast error handling (no retries).
        
        Args:
            input: Dictionary containing 'image_path' or 'image_data' key
            config: Optional runnable configuration
            
        Returns:
            Extracted text or failure message
        """
        if 'image_path' in input:
            image_path = input['image_path']
            if not os.path.exists(image_path):
                self.logger.warning(f"Image file '{image_path}' not found.")
                return None
            
            try:
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")
            except Exception as e:
                self.logger.error(f"Failed to read image file {image_path}: {e}")
                return None
        else:
            image_data = input.get('image_data')
            image_path = input.get('image_name', 'unknown')
        
        if not image_data:
            self.logger.error("No image data provided")
            return None
        
        # CRITICAL FIX: Fast processing - no retries for parsing efficiency
        try:
            self.logger.info(f"Processing OCR for {image_path}")
            
            # Direct invocation to avoid LangChain template issues
            result = self.chain.invoke(
                {"image_data": image_data}, 
                config=config, 
                **kwargs
            )
            
            # Enhanced text cleaning to prevent contamination
            extracted_text = self._clean_raw_text_response(result)
            
            if extracted_text and extracted_text.strip():
                # Check for repetition patterns
                if self._detect_repetition(extracted_text):
                    self.logger.warning(f"Detected repetition in {image_path}")
                    if self.skip_failed_images:
                        return "[OCR_FAILED: Repetition detected in output]"
                    else:
                        return None
                
                self.logger.info(f"Successfully extracted text from {image_path} ({len(extracted_text)} chars)")
                return extracted_text
            else:
                self.logger.warning(f"No text extracted from {image_path}")
                
        except Exception as e:
            self.logger.error(f"OCR error for {image_path}: {e}")
        
        # Handle failure
        if self.skip_failed_images:
            self.logger.warning(f"Skipping failed image {image_path}")
            return "[OCR_FAILED: Image processing failed]"
        else:
            return None
    
    def _detect_repetition(self, text: str) -> bool:
        """Detect repetitive patterns in the text."""
        if not text or len(text) < 50:
            return False
        
        # Check for repeated phrases (3+ words)
        words = text.split()
        if len(words) < 10:
            return False
        
        # Look for patterns that repeat more than 3 times
        for i in range(len(words) - 9):
            phrase = ' '.join(words[i:i+3])
            if text.count(phrase) >= 3:
                return True
        
        # Check for character repetition
        if re.search(r'(.)\1{10,}', text):
            return True
        
        return False
    
    def _clean_raw_text_response(self, response_text: str) -> Optional[str]:
        """Enhanced text cleaning to prevent contamination."""
        if not response_text:
            return None
        
        # Remove common prompt artifacts
        cleaned = response_text.strip()
        
        # Remove system prompt contamination
        contamination_patterns = [
            r'^.*?You are an OCR assistant.*?\n',
            r'^.*?Extract all visible text.*?\n',
            r'^.*?Output only the extracted text.*?\n',
            r'^.*?Extracted text:\s*',
            r'^.*?Extract all text from this image:\s*',
            r'<\|im_start\|>.*?<\|im_end\|>',
            r'<\|.*?\|>',
            r'Human:.*?AI:',
            r'system:.*?user:',
            r'assistant:.*?user:',
        ]
        
        for pattern in contamination_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove markdown artifacts
        cleaned = re.sub(r'^```.*?\n', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n```$', '', cleaned, flags=re.MULTILINE)
        
        # CRITICAL FIX: Enhanced repetition detection and removal
        # Remove repeated lines
        lines = cleaned.split('\n')
        unique_lines = []
        seen_lines = set()
        
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean not in seen_lines:
                unique_lines.append(line)
                seen_lines.add(line_clean)
        
        cleaned = '\n'.join(unique_lines)
        
        # Clean up whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)
        cleaned = re.sub(r'\n[ \t]+', '\n', cleaned)
        
        # Strip final whitespace
        cleaned = cleaned.strip()
        
        # Return None if empty
        if not cleaned or re.match(r'^[\s\n]*$', cleaned):
            return None
            
        return cleaned


class LangChainHybridPDFConverter(Runnable[Dict[str, Any], str]):
    """
    LangChain-compatible Hybrid PDF converter with comprehensive fixes.
    """
    
    def __init__(self, 
                 table_strategy: str = "replace", 
                 extract_images: bool = True, 
                 image_format: str = "png", 
                 dpi: int = 150, 
                 ocr_model: str = "qwen2.5vl:7b",
                 ocr_base_url: str = "http://localhost:11434",
                 skip_failed_images: bool = True):
        """
        Initialize LangChain hybrid converter with comprehensive fixes.
        
        Args:
            table_strategy: "replace" or "enhance"
            extract_images: Whether to extract images from PDF
            image_format: Image format for extracted images
            dpi: Image resolution for extracted images
            ocr_model: Ollama VLM model for OCR
            ocr_base_url: Ollama server URL
            skip_failed_images: Whether to skip images that fail OCR
        """
        logging.basicConfig(level=logging.INFO)
        self.table_strategy = table_strategy
        self.extract_images = extract_images
        self.image_format = image_format
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        
        # CRITICAL FIX: Initialize OCR processor with optimized settings
        self.ocr_processor = LangChainOCRProcessor(
            model=ocr_model,
            base_url=ocr_base_url,
            temperature=0.3,  # Optimal for qwen2.5vl
            skip_failed_images=skip_failed_images
        )
    
    def invoke(self, 
               input: Dict[str, Any], 
               config: Optional[RunnableConfig] = None,
               **kwargs) -> str:
        """
        Convert PDF using hybrid approach with comprehensive fixes.
        
        Args:
            input: Dictionary containing 'pdf_path' and optional 'output_path'
            config: Optional runnable configuration
            
        Returns:
            Markdown text with OCR applied to images
        """
        pdf_path = input['pdf_path']
        output_path = input.get('output_path')
        
        # Step 1: Run standard hybrid conversion
        markdown_content = self.convert_pdf_hybrid(pdf_path, output_path)
        
        # Step 2: Apply enhanced OCR to extracted images
        if output_path and self.extract_images:
            self.logger.info("Applying enhanced LangChain OCR to extracted images...")
            markdown_content = self.apply_langchain_ocr_to_images(markdown_content, output_path, config)
            
            # Save updated markdown with OCR content
            pathlib.Path(output_path).write_text(markdown_content, encoding='utf-8')
            self.logger.info(f"Updated markdown with enhanced OCR saved to: {output_path}")
        
        return markdown_content

    def apply_langchain_ocr_to_images(self, 
                                     markdown_content: str, 
                                     output_path: str,
                                     config: Optional[RunnableConfig] = None) -> str:
        """
        Apply enhanced OCR to images with fast error handling (no retries).
        """
        image_pattern = r'!\[(.*?)\]\((.*?)\)'
        matches = list(re.finditer(image_pattern, markdown_content))
        
        if not matches:
            return markdown_content
        
        self.logger.info(f"Found {len(matches)} images to process with enhanced OCR")
        
        updated_content = markdown_content
        output_dir = os.path.dirname(output_path)
        
        for idx, match in enumerate(reversed(matches), 1):
            image_path = match.group(2)
            full_image_path = os.path.join(output_dir, image_path)
            
            # Progress indicator
            total_images = len(matches)
            current_image = total_images - idx + 1
            self.logger.info(f"Processing image {current_image}/{total_images}: {image_path}")
            
            # CRITICAL FIX: Enhanced OCR processing with fast error handling
            try:
                ocr_text = self.ocr_processor.invoke(
                    {"image_path": full_image_path}, 
                    config=config
                )
                
                if ocr_text and ocr_text.strip():
                    # Check for failure markers
                    if ocr_text.startswith("[OCR_FAILED"):
                        addition = f"\n\n**OCR Status:**\n{ocr_text.strip()}\n"
                        self.logger.warning(f"OCR failed for image {current_image}/{total_images}: {image_path}")
                    else:
                        # CRITICAL FIX: Enhanced OCR content formatting
                        addition = f"\n\n**Extracted Text:**\n{ocr_text.strip()}\n"
                        self.logger.info(f"Successfully processed image {current_image}/{total_images}: {image_path} ({len(ocr_text)} chars)")
                    
                    # Insert after the original image placeholder
                    end_pos = match.end()
                    updated_content = updated_content[:end_pos] + addition + updated_content[end_pos:]
                    
                else:
                    self.logger.warning(f"No text extracted from image {current_image}/{total_images}: {image_path}")
                    
            except Exception as e:
                self.logger.error(f"Error processing image {current_image}/{total_images}: {image_path} - {e}")
                addition = f"\n\n**OCR Status:**\n[OCR_ERROR: {str(e)}]\n"
                end_pos = match.end()
                updated_content = updated_content[:end_pos] + addition + updated_content[end_pos:]

        self.logger.info(f"Completed enhanced OCR processing for all {len(matches)} images")
        return updated_content

    def extract_tables_with_pdfplumber(self, pdf_path: str) -> Dict[int, List[str]]:
        """Extract tables from PDF using pdfplumber."""
        tables_by_page = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = []
                
                # Extract all tables from page
                tables = page.extract_tables()
                
                for table in tables:
                    if table and len(table) > 1:  # Ensure table has header + data
                        # Convert to markdown table
                        markdown_table = self._table_to_markdown(table)
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
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Clean cell content
                    cleaned_cell = str(cell).strip().replace('\n', ' ').replace('|', '\\|')
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
        """Detect potential table regions in text."""
        table_regions = []
        
        # Look for table-like patterns
        lines = text.split('\n')
        in_table = False
        start_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Detect table start
            if not in_table and (
                line.count('|') >= 2 or  # Markdown table
                line.count('\t') >= 2 or  # Tab-separated
                re.match(r'^[^|]*\|[^|]*\|[^|]*', line) or  # Pipe-separated
                re.match(r'^.{10,}\s{3,}.{10,}', line)  # Column-aligned
            ):
                in_table = True
                start_idx = i
            
            # Detect table end
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
        """Replace detected table regions with pdfplumber-extracted tables."""
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
                offset += (end - start - 1)
        
        return '\n'.join(lines)
    
    def _enhance_text_with_tables(self, text: str, page_num: int, 
                                 pdfplumber_tables: Dict[int, List[str]]) -> str:
        """Add pdfplumber tables to text if they don't already exist."""
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
        """Convert PDF using hybrid approach with enhanced settings."""
        # Step 1: Extract tables using pdfplumber
        self.logger.info("Extracting tables with pdfplumber...")
        pdfplumber_tables = self.extract_tables_with_pdfplumber(pdf_path)
        
        # Step 2: Extract text, structure, and images using PyMuPDF4LLM
        self.logger.info("Extracting text, structure, and images with PyMuPDF4LLM...")
        
        # Configure image extraction path
        image_path = ""
        if self.extract_images and output_path:
            output_dir = pathlib.Path(output_path).parent
            image_path = str(output_dir / "images")
        
        # Use PyMuPDF4LLM's built-in image extraction
        chunks = pymupdf4llm.to_markdown(
            pdf_path, 
            page_chunks=True,
            write_images=self.extract_images,
            image_path=image_path,
            image_format=self.image_format,
            dpi=self.dpi,
            image_size_limit=0.05,  # Ignore images smaller than 5% of page
            ignore_images=not self.extract_images
        )
        
        # Step 3: Process each page for table enhancement
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
            if self.extract_images:
                self.logger.info(f"Images saved to: {image_path}")
        
        return final_markdown


# convenience functions
def create_langchain_pdf_converter(model: str = "qwen2.5vl:7b", **kwargs) -> LangChainHybridPDFConverter:
    """Create a LangChain-based PDF converter with qwen2.5vl optimizations."""
    return LangChainHybridPDFConverter(ocr_model=model, **kwargs)


def convert_pdf_with_langchain(pdf_path: str, 
                              output_path: Optional[str] = None,
                              model: str = "qwen2.5vl:7b",
                              **kwargs) -> str:
    """
    Enhanced PDF conversion with comprehensive fixes for qwen2.5vl.
    
    Args:
        pdf_path: Path to input PDF
        output_path: Path to output markdown file (optional)
        model: Ollama VLM model to use (default: qwen2.5vl:7b)
        **kwargs: Additional arguments for the converter
    
    Returns:
        Markdown text with enhanced OCR applied to images
    """
    converter = create_langchain_pdf_converter(model=model, **kwargs)
    return converter.invoke({
        "pdf_path": pdf_path,
        "output_path": output_path
    })


def create_pdf_processing_chain(model: str = "qwen2.5vl:7b", **kwargs):
    """
    Create an enhanced LangChain chain for PDF processing.
    
    Returns:
        A Runnable chain optimized for qwen2.5vl
    """
    converter = create_langchain_pdf_converter(model=model, **kwargs)
    
    def preprocess(input_data):
        # preprocessing
        return input_data
    
    def postprocess(markdown_content):
        # postprocessing 
        return {
            "markdown": markdown_content, 
            "status": "completed",
            "model": model,
            "timestamp": time.time()
        }
    
    # Create the enhanced chain
    chain = (
        RunnableLambda(preprocess) |
        converter |
        RunnableLambda(postprocess)
    )
    
    return chain


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pdf_chain = create_pdf_processing_chain(
        model="qwen2.5vl:7b",
        table_strategy="replace",
        extract_images=True,
        dpi = 400,
        skip_failed_images=True
    )
    
    result = pdf_chain.invoke({
        "pdf_path": "reporting.pdf",
        "output_path": "advanced_enhanced_output.md"
    })
    
    print(f"Advanced processing completed with status: {result['status']}")
    print(f"Model used: {result['model']}")
    print(f"Timestamp: {result['timestamp']}")