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
    Maintains the original prompt and functionality while using LangChain's interface.
    """
    
    def __init__(self, 
                 model: str = "qwen2.5vl:7b",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.1,
                 max_retries: int = 2,
                 timeout: int = 120,
                 skip_failed_images: bool = True):
        """
        Initialize LangChain OCR processor.
        
        Args:
            model: Ollama VLM model name
            base_url: Ollama server URL
            temperature: Model temperature
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds (stored for potential future use)
            skip_failed_images: Whether to skip failed images
        """
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        self.timeout = timeout  # Store timeout for potential future use
        self.skip_failed_images = skip_failed_images
        
        # Initialize ChatOllama with optimal settings for OCR
        # Note: ChatOllama doesn't have a timeout parameter
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            top_p=0.001,
            repeat_penalty=1.05,
            num_predict=2048
        )
        
        # Create the OCR prompt template (preserving original prompt)
        self.ocr_prompt = self._create_ocr_prompt()
        
        # Create the processing chain
        self.chain = self.ocr_prompt | self.llm | StrOutputParser()
    
    def _create_ocr_prompt(self) -> ChatPromptTemplate:
        """Create the OCR prompt template maintaining original functionality."""
        system_prompt = """Extract all visible text from this image accurately and completely.

Instructions:
- Extract ALL text including headers, body text, annotations, numbers, and special characters
- Preserve the logical reading order and line breaks
- Include handwritten text if present
- Output only the extracted text with no explanations, formatting, or extra commentary
- If no text is found, respond with nothing

Extracted text:"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", [
                {"type": "text", "text": "Please extract all text from this image:"},
                {"type": "image", "source_type": "base64", "data": "{image_data}", "mime_type": "image/png"}
            ])
        ])
    
    def invoke(self, 
               input: Dict[str, Any], 
               config: Optional[RunnableConfig] = None,
               **kwargs) -> Any:
        """
        Process a single image with retry logic.
        
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
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"Retrying OCR for {image_path} (attempt {attempt + 1}/{self.max_retries + 1})")
                
                self.logger.info(f"Starting OCR for {image_path}")
                
                result = self.chain.invoke(
                    {"image_data": image_data}, 
                    config=config, 
                    **kwargs
                )
                
                # Clean up the response
                extracted_text = self._clean_raw_text_response(result)
                
                if extracted_text and extracted_text.strip():
                    self.logger.info(f"Successfully extracted text from {image_path} ({len(extracted_text)} chars)")
                    return extracted_text
                else:
                    self.logger.warning(f"No text extracted from {image_path} on attempt {attempt + 1}")
                    
            except Exception as e:
                self.logger.error(f"OCR error for {image_path} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
        
        # All retries failed
        self.logger.error(f"Failed to extract text from {image_path} after {self.max_retries + 1} attempts")
        
        if self.skip_failed_images:
            self.logger.warning(f"Skipping failed image {image_path} and continuing processing")
            return "[OCR_FAILED: Image processing timed out or failed]"
        else:
            return None
    
    def _clean_raw_text_response(self, response_text: str) -> Optional[str]:
        """Clean up raw text response (preserving original logic)."""
        if not response_text:
            return None
        
        # Remove common prompt artifacts that might leak through
        cleaned = response_text.strip()
        
        # Remove any leftover prompt text that might appear at the start
        cleaned = re.sub(r'^.*?Extracted text:\s*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove any markdown code blocks that might wrap the text
        cleaned = re.sub(r'^```.*?\n', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n```$', '', cleaned, flags=re.MULTILINE)
        
        # Clean up excessive whitespace but preserve meaningful line breaks
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)  # Remove trailing spaces
        cleaned = re.sub(r'\n[ \t]+', '\n', cleaned)  # Remove leading spaces on lines
        
        # Strip final whitespace
        cleaned = cleaned.strip()
        
        # Return None if we ended up with only whitespace
        if not cleaned or re.match(r'^[\s\n]*$', cleaned):
            return None
            
        return cleaned


class LangChainHybridPDFConverter(Runnable[Dict[str, Any], str]):
    """
    LangChain-compatible Hybrid PDF converter maintaining all original functionality.
    """
    
    def __init__(self, 
                 table_strategy: str = "replace", 
                 extract_images: bool = True, 
                 image_format: str = "png", 
                 dpi: int = 150, 
                 ocr_model: str = "qwen2.5vl:7b",
                 ocr_base_url: str = "http://localhost:11434",
                 ocr_timeout: int = 60, 
                 max_retries: int = 2, 
                 skip_failed_images: bool = True):
        """
        Initialize LangChain hybrid converter.
        
        Args:
            table_strategy: "replace" or "enhance"
            extract_images: Whether to extract images from PDF
            image_format: Image format for extracted images
            dpi: Image resolution for extracted images
            ocr_model: Ollama VLM model for OCR
            ocr_base_url: Ollama server URL
            ocr_timeout: Timeout in seconds for OCR requests
            max_retries: Maximum number of retry attempts for failed OCR
            skip_failed_images: Whether to skip images that fail OCR
        """
        logging.basicConfig(level=logging.INFO)
        self.table_strategy = table_strategy
        self.extract_images = extract_images
        self.image_format = image_format
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        
        # Initialize LangChain OCR processor
        self.ocr_processor = LangChainOCRProcessor(
            model=ocr_model,
            base_url=ocr_base_url,
            temperature=0.1,
            max_retries=max_retries,
            timeout=ocr_timeout,
            skip_failed_images=skip_failed_images
        )
    
    def invoke(self, 
               input: Dict[str, Any], 
               config: Optional[RunnableConfig] = None,
               **kwargs) -> str:
        """
        Convert PDF using hybrid approach with LangChain OCR.
        
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
        
        # Step 2: Apply LangChain OCR to extracted images if output path provided
        if output_path and self.extract_images:
            self.logger.info("Applying LangChain OCR to extracted images...")
            markdown_content = self.apply_langchain_ocr_to_images(markdown_content, output_path, config)
            
            # Save updated markdown with OCR content
            pathlib.Path(output_path).write_text(markdown_content, encoding='utf-8')
            self.logger.info(f"Updated markdown with LangChain OCR saved to: {output_path}")
        
        return markdown_content

    def apply_langchain_ocr_to_images(self, 
                                     markdown_content: str, 
                                     output_path: str,
                                     config: Optional[RunnableConfig] = None) -> str:
        """
        Find image placeholders in markdown and add OCR text using LangChain.
        """
        image_pattern = r'!\[(.*?)\]\((.*?)\)'
        matches = list(re.finditer(image_pattern, markdown_content))
        
        if not matches:
            return markdown_content
        
        self.logger.info(f"Found {len(matches)} images to process with LangChain OCR")
        
        updated_content = markdown_content
        output_dir = os.path.dirname(output_path)
        
        for idx, match in enumerate(reversed(matches), 1):
            image_path = match.group(2)
            full_image_path = os.path.join(output_dir, image_path)
            
            # Progress indicator
            total_images = len(matches)
            current_image = total_images - idx + 1
            self.logger.info(f"Processing image {current_image}/{total_images}: {image_path}")
            
            # Extract text from image using LangChain OCR
            ocr_text = self.ocr_processor.invoke(
                {"image_path": full_image_path}, 
                config=config
            )
            
            if ocr_text and ocr_text.strip():
                # Check if this is a failure marker
                if ocr_text.startswith("[OCR_FAILED"):
                    addition = f"\n\n**Extracted OCR text:**\n{ocr_text.strip()}\n"
                    self.logger.warning(f"OCR failed for image {current_image}/{total_images}: {image_path}")
                else:
                    # Create addition with OCR content
                    addition = f"\n\n**Extracted OCR text:**\n{ocr_text.strip()}\n"
                    self.logger.info(f"Successfully processed image {current_image}/{total_images}: {image_path} ({len(ocr_text)} chars)")
                
                # Insert after the original image placeholder
                end_pos = match.end()
                updated_content = updated_content[:end_pos] + addition + updated_content[end_pos:]
                
            else:
                self.logger.warning(f"No text extracted from image {current_image}/{total_images}: {image_path}")

        self.logger.info(f"Completed LangChain OCR processing for all {len(matches)} images")
        return updated_content

    def extract_tables_with_pdfplumber(self, pdf_path: str) -> Dict[int, List[str]]:
        """Extract tables from PDF using pdfplumber (unchanged)."""
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
        """Convert table data to markdown format (unchanged)."""
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
        """Detect potential table regions in text (unchanged)."""
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
        """Replace detected table regions with pdfplumber-extracted tables (unchanged)."""
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
        """Add pdfplumber tables to text if they don't already exist (unchanged)."""
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
        """Convert PDF using hybrid approach (mostly unchanged)."""
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


# Convenience functions for easy usage
def create_langchain_pdf_converter(model: str = "qwen2.5vl:7b", **kwargs) -> LangChainHybridPDFConverter:
    """Create a LangChain-based PDF converter with specified model."""
    return LangChainHybridPDFConverter(ocr_model=model, **kwargs)


def convert_pdf_with_langchain(pdf_path: str, 
                              output_path: Optional[str] = None,
                              model: str = "qwen2.5vl:7b",
                              **kwargs) -> str:
    """
    Simple function interface for LangChain-based hybrid PDF conversion.
    
    Args:
        pdf_path: Path to input PDF
        output_path: Path to output markdown file (optional)
        model: Ollama VLM model to use
        **kwargs: Additional arguments for the converter
    
    Returns:
        Markdown text with LangChain OCR applied to images
    """
    converter = create_langchain_pdf_converter(model=model, **kwargs)
    return converter.invoke({
        "pdf_path": pdf_path,
        "output_path": output_path
    })


# Advanced usage with LangChain chains
def create_pdf_processing_chain(model: str = "qwen2.5vl:7b", **kwargs):
    """
    Create a LangChain chain for PDF processing.
    
    Returns:
        A Runnable chain that can be used with LangChain LCEL
    """
    converter = create_langchain_pdf_converter(model=model, **kwargs)
    
    # Define preprocessing and postprocessing steps if needed
    def preprocess(input_data):
        # Add any preprocessing logic here
        return input_data
    
    def postprocess(markdown_content):
        # Add any postprocessing logic here
        return {"markdown": markdown_content, "status": "completed"}
    
    # Create the full chain using LCEL
    chain = (
        RunnableLambda(preprocess) |
        converter |
        RunnableLambda(postprocess)
    )
    
    return chain


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Simple usage
    markdown = convert_pdf_with_langchain(
        "Notes_250713_233056.pdf", 
        "langchain_output.md", 
        model="qwen2.5vl:7b",
        table_strategy="enhance", 
        extract_images=True,
        image_format="png", 
        dpi=500,
        ocr_timeout=120,
        max_retries=1,
        skip_failed_images=True
    )
    
    # Advanced usage with custom chain
    pdf_chain = create_pdf_processing_chain(
        model="qwen2.5vl:7b",
        table_strategy="replace",
        extract_images=True
    )
    
    result = pdf_chain.invoke({
        "pdf_path": "Notes_250713_233056.pdf",
        "output_path": "advanced_output.md"
    })