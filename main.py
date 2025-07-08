import pathlib
import pymupdf4llm
import pdfplumber
import pandas as pd
import re
import requests
import base64
import os
from typing import List, Dict, Optional, Tuple
import logging


class HybridPDFConverter:
    """
    Hybrid converter that uses:
    - PyMuPDF4LLM for text extraction, document structure, AND image extraction
    - pdfplumber for superior table extraction
    - Ollama OCR for image text extraction
    """
    
    def __init__(self, table_strategy: str = "replace", extract_images: bool = True, 
                 image_format: str = "png", dpi: int = 150):
        """
        Initialize hybrid converter.
        
        Args:
            table_strategy: "replace" or "enhance"
                - "replace": Replace PyMuPDF tables with pdfplumber tables
                - "enhance": Add pdfplumber tables if PyMuPDF missed them
            extract_images: Whether to extract images from PDF
            image_format: Image format for extracted images (png, jpg, etc.)
            dpi: Image resolution for extracted images
        """
        self.table_strategy = table_strategy
        self.extract_images = extract_images
        self.image_format = image_format
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_image_ollama(self, image_path: str) -> Optional[str]:
        """Extract text from image using Ollama OCR"""
        if not os.path.exists(image_path):
            self.logger.warning(f"Image file '{image_path}' not found.")
            return None

        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        url = "http://localhost:11434/api/generate"
        prompt = """
                    You are a highly accurate OCR assistant. Your goal is to extract **all** visible text from the supplied image—printed, handwritten, or stylized—without adding any commentary, formatting, or markup.  
                    • Output only the raw text, preserving line breaks and paragraph breaks as they appear in the image.  
                    • Do not include any bounding box data or JSON—just plaintext.  
                    • Do not label or annotate anything.  
                    • If you encounter illegible regions, skip them silently.  
                    Begin now:
    """

        payload = {
            "model": "qwen2.5vl:3b",
            "prompt": prompt,
            "images": [image_data],
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.1},
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error communicating with Ollama: {e}")
            return None

    def apply_ocr_to_images(self, markdown_content: str, output_path: str) -> str:

        """
        Find image placeholders in markdown and add OCR text below them
        """
        # Find all image references in markdown
        image_pattern = r'!\[(.*?)\]\((.*?)\)'
        matches = list(re.finditer(image_pattern, markdown_content))
        
        if not matches:
            return markdown_content
        
        # Process images from end to start to maintain string positions
        updated_content = markdown_content
        output_dir = os.path.dirname(output_path)
        
        for match in reversed(matches):
            image_path = match.group(2)
            full_image_path = os.path.join(output_dir, image_path)
            
            # Extract text from image using OCR
            ocr_text = self.extract_text_from_image_ollama(full_image_path)
            
            if ocr_text and ocr_text.strip():
                # Create addition with OCR content below the image
                addition = f"\n\n**Extracted OCR text:**\n```\n{ocr_text.strip()}\n```"
                
                # Insert after the original image placeholder
                end_pos = match.end()
                updated_content = updated_content[:end_pos] + addition + updated_content[end_pos:]
                
                self.logger.info(f"Applied OCR to: {image_path}")
            else:
                self.logger.warning(f"No text extracted from: {image_path}")

        return updated_content

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
    
    def convert_pdf_hybrid_with_ocr(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert PDF using hybrid approach with OCR integration.
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path to output markdown file (optional)
            
        Returns:
            Markdown text with OCR applied to images
        """
        # Step 1: Run standard hybrid conversion
        markdown_content = self.convert_pdf_hybrid(pdf_path, output_path)
        
        # Step 2: Apply OCR to extracted images if output path provided
        if output_path and self.extract_images:
            self.logger.info("Applying OCR to extracted images...")
            markdown_content = self.apply_ocr_to_images(markdown_content, output_path)
            
            # Save updated markdown with OCR content
            pathlib.Path(output_path).write_text(markdown_content, encoding='utf-8')
            self.logger.info(f"Updated markdown with OCR saved to: {output_path}")
        
        return markdown_content

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
            ignore_images=not self.extract_images  # Only ignore if not extracting
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
    
    def convert_pdf_table_focused(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert PDF with focus on table extraction quality.
        Uses pdfplumber for everything on pages with tables, PyMuPDF4LLM for text-only pages.
        """
        # Extract tables first to identify table-heavy pages
        pdfplumber_tables = self.extract_tables_with_pdfplumber(pdf_path)
        table_pages = set(pdfplumber_tables.keys())
        
        # Configure image extraction
        image_path = ""
        if self.extract_images and output_path:
            output_dir = pathlib.Path(output_path).parent
            image_path = str(output_dir / "images")
        
        # Get PyMuPDF4LLM chunks with built-in image extraction
        chunks = pymupdf4llm.to_markdown(
            pdf_path, 
            page_chunks=True,
            write_images=self.extract_images,
            image_path=image_path,
            image_format=self.image_format,
            dpi=self.dpi,
            ignore_images=not self.extract_images
        )
        
        processed_pages = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, chunk in enumerate(chunks):
                if i in table_pages:
                    # Use pdfplumber for table-heavy pages
                    page = pdf.pages[i]
                    
                    # Extract text
                    text = page.extract_text() or ""
                    
                    # Add tables
                    if i in pdfplumber_tables:
                        text += '\n\n' + '\n\n'.join(pdfplumber_tables[i])
                    
                    processed_pages.append(text)
                else:
                    # Use PyMuPDF4LLM for text-only pages (images already handled)
                    processed_pages.append(chunk['text'])
        
        final_markdown = '\n\n---\n\n'.join(processed_pages)
        
        if output_path:
            pathlib.Path(output_path).write_text(final_markdown, encoding='utf-8')
        
        return final_markdown


def convert_pdf_hybrid_with_ocr(pdf_path: str, output_path: Optional[str] = None, 
                                strategy: str = "replace", extract_images: bool = True,
                                image_format: str = "png", dpi: int = 150) -> str:
    """
    Simple function interface for hybrid PDF conversion with OCR.
    
    Args:
        pdf_path: Path to input PDF
        output_path: Path to output markdown file (optional)
        strategy: "replace", "enhance", or "table_focused"
        extract_images: Whether to extract and reference images
        image_format: Image format (png, jpg, etc.)
        dpi: Image resolution
    
    Returns:
        Markdown text with OCR applied to images
    """
    converter = HybridPDFConverter(
        table_strategy=strategy, 
        extract_images=extract_images,
        image_format=image_format,
        dpi=dpi
    )
    
    return converter.convert_pdf_hybrid_with_ocr(pdf_path, output_path)

if __name__ == "__main__":
    markdown = convert_pdf_hybrid_with_ocr("sample.pdf", "output.md", 
                                          strategy="replace", extract_images=True,
                                          image_format="png", dpi=500)