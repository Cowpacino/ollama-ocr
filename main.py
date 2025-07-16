import pathlib
import pymupdf4llm
import pdfplumber
import pandas as pd
import re
import base64
import os
import time
import json
from typing import List, Dict, Optional, Tuple, Any
import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

# LangChain imports
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


class DocumentType(Enum):
    """Document type classification for specialized processing"""
    STRUCTURED_FORM = "structured_form"
    SEMI_STRUCTURED_FORM = "semi_structured_form"
    UNSTRUCTURED_TEXT = "unstructured_text"
    TABLE_HEAVY = "table_heavy"
    HANDWRITTEN = "handwritten"
    TECHNICAL_DIAGRAM = "technical_diagram"
    INVOICE_RECEIPT = "invoice_receipt"
    LEGAL_DOCUMENT = "legal_document"
    SCIENTIFIC_PAPER = "scientific_paper"
    MIXED_CONTENT = "mixed_content"


@dataclass
class OCRResult:
    """Structured OCR result with metadata"""
    text: str
    confidence: float
    document_type: DocumentType
    language: Optional[str]
    processing_time: float
    word_count: int
    character_count: int
    has_tables: bool
    has_handwriting: bool
    error_message: Optional[str] = None


class ProductionOCRProcessor(Runnable[Dict[str, Any], OCRResult]):
    """
    Production-grade OCR processor with enhanced error handling and repetition detection.
    """
    
    def __init__(self, 
                 model: str = "qwen2.5vl:7b",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.1,
                 enable_confidence_scoring: bool = True,
                 enable_language_detection: bool = True,
                 quality_threshold: float = 0.8,
                 max_tokens: int = 2048):
        """
        Initialize with improved parameters and validation
        """
        self.logger = logging.getLogger(__name__)
        self.enable_confidence_scoring = enable_confidence_scoring
        self.enable_language_detection = enable_language_detection
        self.quality_threshold = quality_threshold
        
        # Initialize ChatOllama with production settings
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            top_p=0.3,
            repeat_penalty=1.5,  # Higher penalty for repetition
            num_predict=max_tokens,
            # Add stop sequences to prevent runaway generation
            stop=[
                "---END---",
                "[EXTRACTION_COMPLETE]",
                "## END OF DOCUMENT",
                "\n\n\n\n\n"
            ]
        )
        
        # Create simplified prompt templates
        self.prompts = self._create_simplified_prompts()
        
        # Document type detector
        self.type_detector = self._create_type_detector()
    
    def _create_simplified_prompts(self) -> Dict[DocumentType, ChatPromptTemplate]:
        """Create simplified prompts that reduce hallucination risk"""
        
        # Simple, direct base instruction
        base_instruction = """You are an OCR system. Extract ALL visible text from this image exactly as it appears.

CRITICAL RULES:
- Extract only what you can clearly see
- Do not repeat the same text multiple times
- Do not generate or imagine content
- If text is unclear, mark it as [unclear]
- Preserve formatting and structure
- Stop when you reach the end of visible content
- Write '---END OF EXTRACTION---' when finished"""

        prompts = {}
        
        # Simplified invoice/receipt prompt
        prompts[DocumentType.INVOICE_RECEIPT] = ChatPromptTemplate.from_messages([
            ("system", f"""{base_instruction}

For invoices/receipts, focus on:
- Company name and details
- Invoice number and date
- Line items with quantities and prices
- Tax information
- Total amounts

Extract exactly what you see, no more, no less. Do not repeat any content."""),
            ("human", [
                {"type": "text", "text": "Extract all visible text from this invoice/receipt:"},
                {"type": "image", "source_type": "base64", "data": "{image_data}", "mime_type": "image/png"}
            ])
        ])
        
        # Simplified structured form prompt
        prompts[DocumentType.STRUCTURED_FORM] = ChatPromptTemplate.from_messages([
            ("system", f"""{base_instruction}

For structured forms:
- Extract field labels and their values
- Maintain label-value relationships
- Preserve table structure if present
- Include form numbers and dates

Extract exactly what you see, no repetition."""),
            ("human", [
                {"type": "text", "text": "Extract all text from this structured form:"},
                {"type": "image", "source_type": "base64", "data": "{image_data}", "mime_type": "image/png"}
            ])
        ])
        
        # Simplified handwritten prompt
        prompts[DocumentType.HANDWRITTEN] = ChatPromptTemplate.from_messages([
            ("system", f"""{base_instruction}

For handwritten documents:
- Take extra care with unclear characters
- Mark uncertain words as [uncertain: word]
- Note illegible sections as [illegible]
- Context helps with unclear characters

Extract exactly what you see, no repetition."""),
            ("human", [
                {"type": "text", "text": "Carefully extract all handwritten and printed text:"},
                {"type": "image", "source_type": "base64", "data": "{image_data}", "mime_type": "image/png"}
            ])
        ])
        
        # Simplified table-heavy prompt
        prompts[DocumentType.TABLE_HEAVY] = ChatPromptTemplate.from_messages([
            ("system", f"""{base_instruction}

For table-heavy documents:
- Extract tables with proper structure
- Use markdown table format
- Include headers and data rows
- Preserve column alignment

Extract exactly what you see, no repetition."""),
            ("human", [
                {"type": "text", "text": "Extract all tables and data from this document:"},
                {"type": "image", "source_type": "base64", "data": "{image_data}", "mime_type": "image/png"}
            ])
        ])
        
        # Default/Mixed Content
        prompts[DocumentType.MIXED_CONTENT] = ChatPromptTemplate.from_messages([
            ("system", f"""{base_instruction}

Extract all visible text in logical reading order. Preserve formatting and structure."""),
            ("human", [
                {"type": "text", "text": "Extract all visible text from this document:"},
                {"type": "image", "source_type": "base64", "data": "{image_data}", "mime_type": "image/png"}
            ])
        ])
        
        # Apply same pattern to other document types
        for doc_type in DocumentType:
            if doc_type not in prompts:
                prompts[doc_type] = prompts[DocumentType.MIXED_CONTENT]
        
        return prompts
    
    def _create_type_detector(self) -> ChatPromptTemplate:
        """Simplified document type detection"""
        return ChatPromptTemplate.from_messages([
            ("system", """Analyze this document image and classify it.

Respond in exactly this format:
TYPE: [document_type]
CONFIDENCE: [0.0-1.0]
LANGUAGE: [language]
HAS_TABLES: [yes/no]
HAS_HANDWRITING: [yes/no]

Document types: invoice_receipt, structured_form, handwritten, table_heavy, mixed_content

Keep the response brief and focused."""),
            ("human", [
                {"type": "text", "text": "Classify this document:"},
                {"type": "image", "source_type": "base64", "data": "{image_data}", "mime_type": "image/png"}
            ])
        ])
    
    def _detect_document_type(self, image_data: str) -> Tuple[DocumentType, float, Optional[str], bool, bool]:
        """Detect document type"""
        try:
            chain = self.type_detector | self.llm | StrOutputParser()
            response = chain.invoke({"image_data": image_data})
            
            # Parse response
            lines = response.strip().split('\n')
            doc_type = DocumentType.MIXED_CONTENT
            confidence = 0.5
            language = None
            has_tables = False
            has_handwriting = False
            
            for line in lines:
                if line.startswith('TYPE:'):
                    type_str = line.split(':', 1)[1].strip()
                    try:
                        doc_type = DocumentType(type_str)
                    except ValueError:
                        self.logger.warning(f"Unknown document type: {type_str}")
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith('LANGUAGE:'):
                    language = line.split(':', 1)[1].strip()
                elif line.startswith('HAS_TABLES:'):
                    has_tables = line.split(':', 1)[1].strip().lower() == 'yes'
                elif line.startswith('HAS_HANDWRITING:'):
                    has_handwriting = line.split(':', 1)[1].strip().lower() == 'yes'
            
            return doc_type, confidence, language, has_tables, has_handwriting
            
        except Exception as e:
            self.logger.error(f"Document type detection failed: {e}")
            return DocumentType.MIXED_CONTENT, 0.5, None, False, False
    
    def _estimate_confidence(self, text: str, processing_time: float) -> float:
        """Improved confidence estimation that detects repetitive/corrupted text"""
        if not text or not text.strip():
            return 0.0
        
        confidence = 1.0
        
        # Check for repetitive patterns (major red flag)
        lines = text.split('\n')
        if len(lines) > 10:
            # Check for repeated lines
            line_counts = {}
            for line in lines:
                line = line.strip()
                if len(line) > 10:  # Only count substantial lines
                    line_counts[line] = line_counts.get(line, 0) + 1
            
            # If any line repeats more than 3 times, severely reduce confidence
            max_repeats = max(line_counts.values()) if line_counts else 0
            if max_repeats > 3:
                confidence *= 0.1  # Severe penalty for repetition
                self.logger.warning(f"Detected {max_repeats} repetitions of same line")
            elif max_repeats > 2:
                confidence *= 0.5
        
        # Check for extremely long sequences of same character/word
        repetitive_patterns = re.findall(r'(.{3,20})\1{5,}', text)
        if repetitive_patterns:
            confidence *= 0.1
            self.logger.warning(f"Detected repetitive patterns: {len(repetitive_patterns)} instances")
        
        # Check for unrealistic content length vs image complexity
        char_count = len(text)
        if char_count > 15000:  # Suspiciously long for most invoices
            confidence *= 0.2
            self.logger.warning(f"Suspiciously long text: {char_count} characters")
        elif char_count > 10000:
            confidence *= 0.5
        
        # Check for uncertainty markers
        uncertainty_markers = ['[?]', '[uncertain:', '[illegible]', '[unclear']
        for marker in uncertainty_markers:
            confidence -= text.count(marker) * 0.1
        
        # Check for reasonable text diversity
        unique_chars = len(set(text.lower().replace(' ', '')))
        if unique_chars < 10:
            confidence *= 0.5
        
        # Processing time indicators
        if processing_time < 10:  # Too fast might indicate hallucination
            confidence *= 0.8
        elif processing_time > 300:  # Too slow might indicate problems
            confidence *= 0.9
        
        return max(0.0, min(1.0, confidence))
    
    def _remove_repetitive_patterns(self, text: str) -> str:
        """Remove repetitive patterns from text"""
        if not text:
            return text
        
        # Remove lines that repeat more than 3 times
        lines = text.split('\n')
        line_counts = {}
        unique_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                unique_lines.append('')
                continue
                
            line_counts[line] = line_counts.get(line, 0) + 1
            
            # Only keep first 2 occurrences of any line
            if line_counts[line] <= 2:
                unique_lines.append(line)
            elif line_counts[line] == 3:
                # Add a note about repetition instead of continuing
                unique_lines.append("[Content repeats - truncated after 2 occurrences]")
        
        text = '\n'.join(unique_lines)
        
        # Remove character/word repetitions like "GST18%GST18%GST18%..."
        # Pattern: same 3-20 character sequence repeated 5+ times
        text = re.sub(r'(.{3,20})\1{4,}', r'\1 [repetitive content removed]', text)
        
        # Remove extremely long sequences of same character
        text = re.sub(r'(.)\1{50,}', r'\1... [long sequence truncated]', text)
        
        return text
    
    def _clean_raw_text_response(self, response_text: str) -> Optional[str]:
        """Enhanced text cleaning that detects and removes repetitive content"""
        if not response_text:
            return None
        
        cleaned = response_text.strip()
        
        # Remove common prompt artifacts
        artifacts = [
            r'^.*?extracted text[:\s]*',
            r'^.*?output[:\s]*',
            r'^.*?result[:\s]*',
            r'^```.*?\n',
            r'\n```$',
            r'^text[:\s]*',
            r'^ocr[:\s]*',
            r'---END OF EXTRACTION---.*$'
        ]
        
        for pattern in artifacts:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        # CRITICAL: Detect and remove repetitive patterns
        cleaned = self._remove_repetitive_patterns(cleaned)
        
        # Clean up excessive whitespace
        cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)
        cleaned = re.sub(r'\n[ \t]+', '\n', cleaned)
        cleaned = re.sub(r'[ \t]{3,}', '  ', cleaned)
        
        # Final cleanup
        cleaned = cleaned.strip()
        
        # Return None if we ended up with only whitespace or suspiciously long text
        if not cleaned or re.match(r'^[\s\n]*$', cleaned) or len(cleaned) > 20000:
            self.logger.warning(f"Rejected text: empty={not cleaned}, too_long={len(cleaned) > 20000}")
            return None
                
        return cleaned
    
    def _validate_extraction_quality(self, text: str) -> bool:
        """Validate if extraction quality is acceptable"""
        if not text or len(text.strip()) < 10:
            return False
        
        # Check for excessive repetition
        lines = text.split('\n')
        if len(lines) > 10:
            line_counts = {}
            for line in lines:
                line = line.strip()
                if len(line) > 10:
                    line_counts[line] = line_counts.get(line, 0) + 1
            
            max_repeats = max(line_counts.values()) if line_counts else 0
            if max_repeats > 3:
                self.logger.warning(f"Quality check failed: {max_repeats} line repetitions")
                return False
        
        # Check for repetitive patterns
        repetitive_patterns = re.findall(r'(.{3,20})\1{5,}', text)
        if repetitive_patterns:
            self.logger.warning(f"Quality check failed: {len(repetitive_patterns)} repetitive patterns")
            return False
        
        # Check for reasonable length
        if len(text) > 15000:  # Suspiciously long
            self.logger.warning(f"Quality check failed: text too long ({len(text)} chars)")
            return False
        
        return True
    
    def invoke(self, 
               input: Dict[str, Any], 
               config: Optional[RunnableConfig] = None,
               **kwargs) -> OCRResult:
        """
        Enhanced invoke method with retry logic based on quality validation
        """
        start_time = time.time()
        
        # Handle input data
        if 'image_path' in input:
            image_path = input['image_path']
            if not os.path.exists(image_path):
                return OCRResult(
                    text="", confidence=0.0, document_type=DocumentType.MIXED_CONTENT,
                    language=None, processing_time=0, word_count=0, character_count=0,
                    has_tables=False, has_handwriting=False,
                    error_message=f"Image file '{image_path}' not found"
                )
            
            try:
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")
            except Exception as e:
                return OCRResult(
                    text="", confidence=0.0, document_type=DocumentType.MIXED_CONTENT,
                    language=None, processing_time=0, word_count=0, character_count=0,
                    has_tables=False, has_handwriting=False,
                    error_message=f"Failed to read image file: {e}"
                )
        else:
            image_data = input.get('image_data')
            image_path = input.get('image_name', 'unknown')
        
        if not image_data:
            return OCRResult(
                text="", confidence=0.0, document_type=DocumentType.MIXED_CONTENT,
                language=None, processing_time=0, word_count=0, character_count=0,
                has_tables=False, has_handwriting=False,
                error_message="No image data provided"
            )
        
        # Step 1: Detect document type
        self.logger.info(f"Detecting document type for {image_path}")
        doc_type, type_confidence, language, has_tables, has_handwriting = self._detect_document_type(image_data)
        self.logger.info(f"Detected type: {doc_type.value} (confidence: {type_confidence:.2f})")
        
        # Step 2: Select appropriate prompt
        prompt_template = self.prompts.get(doc_type, self.prompts[DocumentType.MIXED_CONTENT])
        chain = prompt_template | self.llm | StrOutputParser()
        
        # Step 3: Extract text with retry logic
        extracted_text = None
        error_message = None
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Starting OCR attempt {attempt + 1} for {image_path}")
                
                result = chain.invoke(
                    {"image_data": image_data}, 
                    config=config, 
                    **kwargs
                )
                
                # Clean up the response
                extracted_text = self._clean_raw_text_response(result)
                
                # Validate result quality
                if extracted_text and self._validate_extraction_quality(extracted_text):
                    self.logger.info(f"✓ Quality extraction from {image_path} on attempt {attempt + 1}")
                    break
                else:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"⚠️ Poor quality extraction, retrying... (attempt {attempt + 1})")
                        continue
                    else:
                        self.logger.error(f"❌ Failed to get quality extraction after {max_retries} attempts")
                        extracted_text = "[OCR_FAILED: Poor quality extraction after retries]"
                        error_message = "Poor quality extraction after retries"
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"⚠️ OCR attempt {attempt + 1} failed: {e}, retrying...")
                    continue
                else:
                    error_message = str(e)
                    extracted_text = "[OCR_FAILED: Processing error]"
                    break
        
        # Calculate metrics
        processing_time = time.time() - start_time
        word_count = len(extracted_text.split()) if extracted_text else 0
        character_count = len(extracted_text) if extracted_text else 0
        
        # Estimate confidence
        if self.enable_confidence_scoring and extracted_text:
            confidence = self._estimate_confidence(extracted_text, processing_time)
        else:
            confidence = type_confidence if extracted_text else 0.0
        
        # Final fallback text
        if not extracted_text:
            extracted_text = "[OCR_FAILED: Unable to extract text]"
            confidence = 0.0
        
        # Log final result
        self.logger.info(f"OCR completed for {image_path}: {character_count} chars, {confidence:.1%} confidence")
        
        return OCRResult(
            text=extracted_text,
            confidence=confidence,
            document_type=doc_type,
            language=language,
            processing_time=processing_time,
            word_count=word_count,
            character_count=character_count,
            has_tables=has_tables,
            has_handwriting=has_handwriting,
            error_message=error_message
        )


class ProductionHybridPDFConverter(Runnable[Dict[str, Any], Dict[str, Any]]):
    """
    Production-grade hybrid PDF converter with enhanced OCR capabilities.
    """
    
    def __init__(self, 
                 table_strategy: str = "enhance", 
                 extract_images: bool = True, 
                 image_format: str = "png", 
                 dpi: int = 300,
                 ocr_model: str = "qwen2.5vl:7b",
                 ocr_base_url: str = "http://localhost:11434",
                 quality_threshold: float = 0.8,
                 save_intermediate_results: bool = True,
                 max_pages: Optional[int] = None):
        """
        Initialize with enhanced settings
        """
        logging.basicConfig(level=logging.INFO)
        self.table_strategy = table_strategy
        self.extract_images = extract_images
        self.image_format = image_format
        self.dpi = dpi
        self.quality_threshold = quality_threshold
        self.save_intermediate_results = save_intermediate_results
        self.max_pages = max_pages
        self.logger = logging.getLogger(__name__)
        
        # Initialize enhanced OCR processor
        self.ocr_processor = ProductionOCRProcessor(
            model=ocr_model,
            base_url=ocr_base_url,
            temperature=0.1,
            enable_confidence_scoring=True,
            enable_language_detection=True,
            quality_threshold=quality_threshold,
            max_tokens=2048
        )
    
    def apply_production_ocr_to_images(self, 
                                     markdown_content: str, 
                                     output_path: str,
                                     config: Optional[RunnableConfig] = None) -> Tuple[str, List[OCRResult]]:
        """
        Apply production-grade OCR with enhanced validation and reporting
        """
        image_pattern = r'!\[(.*?)\]\((.*?)\)'
        matches = list(re.finditer(image_pattern, markdown_content))
        
        if not matches:
            return markdown_content, []
        
        self.logger.info(f"Found {len(matches)} images for OCR processing")
        
        updated_content = markdown_content
        output_dir = os.path.dirname(output_path)
        ocr_results = []
        successful_extractions = 0
        
        for idx, match in enumerate(reversed(matches), 1):
            image_path = match.group(2)
            full_image_path = os.path.join(output_dir, image_path)
            
            # Progress indicator
            total_images = len(matches)
            current_image = total_images - idx + 1
            self.logger.info(f"Processing image {current_image}/{total_images}: {image_path}")
            
            # Extract text using enhanced OCR
            ocr_result = self.ocr_processor.invoke(
                {"image_path": full_image_path}, 
                config=config
            )
            ocr_results.append(ocr_result)
            
            # Create detailed OCR section based on quality
            if ocr_result.error_message:
                # Error case
                addition = f"""

**❌ OCR Extraction Failed**
*Error: {ocr_result.error_message}*
*Document Type: {ocr_result.document_type.value.replace('_', ' ').title()}*
*Processing Time: {ocr_result.processing_time:.1f}s*

---
"""
                self.logger.error(f"❌ Failed to extract text from {image_path}: {ocr_result.error_message}")
                
            elif ocr_result.text and ocr_result.confidence >= self.quality_threshold:
                # High-quality extraction
                addition = f"""

**✅ OCR Extraction (High Quality: {ocr_result.confidence:.1%})**
*Document Type: {ocr_result.document_type.value.replace('_', ' ').title()}*
*Language: {ocr_result.language or 'Auto-detected'}*
*Processing Time: {ocr_result.processing_time:.1f}s*
*Words: {ocr_result.word_count} | Characters: {ocr_result.character_count}*

{ocr_result.text.strip()}

---
"""
                successful_extractions += 1
                self.logger.info(f"✅ High-quality extraction from {image_path} "
                               f"({ocr_result.confidence:.1%} confidence, {len(ocr_result.text)} chars)")
                
            elif ocr_result.text and ocr_result.confidence < self.quality_threshold:
                # Low-quality extraction
                addition = f"""

**⚠️ OCR Extraction (Low Quality: {ocr_result.confidence:.1%})**
*Document Type: {ocr_result.document_type.value.replace('_', ' ').title()}*
*⚠️ Note: Low confidence extraction - manual review recommended*
*Processing Time: {ocr_result.processing_time:.1f}s*

{ocr_result.text.strip()}

---
"""
                self.logger.warning(f"⚠️ Low-confidence extraction from {image_path} "
                                  f"({ocr_result.confidence:.1%} confidence)")
                
            else:
                # No text extracted
                addition = f"""

**❌ OCR Extraction Failed**
*No text could be extracted from this image*
*Document Type: {ocr_result.document_type.value.replace('_', ' ').title()}*
*Processing Time: {ocr_result.processing_time:.1f}s*

---
"""
                self.logger.error(f"❌ No text extracted from {image_path}")
            
            # Insert after the original image placeholder
            end_pos = match.end()
            updated_content = updated_content[:end_pos] + addition + updated_content[end_pos:]

        # Final summary
        success_rate = successful_extractions / len(matches) * 100 if matches else 0
        self.logger.info(f"OCR processing completed: {successful_extractions}/{len(matches)} successful ({success_rate:.1f}%)")
        
        return updated_content, ocr_results
    
    def invoke(self, 
               input: Dict[str, Any], 
               config: Optional[RunnableConfig] = None,
               **kwargs) -> Dict[str, Any]:
        """
        Enhanced PDF conversion with better error handling
        """
        pdf_path = input['pdf_path']
        output_path = input.get('output_path')
        
        processing_start = time.time()
        
        try:
            # Step 1: Run hybrid conversion
            self.logger.info(f"Starting enhanced PDF conversion: {pdf_path}")
            markdown_content = self.convert_pdf_hybrid(pdf_path, output_path)
            
            # Step 2: Apply enhanced OCR if images extracted
            ocr_results = []
            if output_path and self.extract_images:
                self.logger.info("Applying enhanced OCR to extracted images...")
                markdown_content, ocr_results = self.apply_production_ocr_to_images(
                    markdown_content, output_path, config
                )
                
                # Save updated markdown
                pathlib.Path(output_path).write_text(markdown_content, encoding='utf-8')
                self.logger.info(f"Enhanced markdown saved to: {output_path}")
            
            # Step 3: Generate processing report
            total_time = time.time() - processing_start
            report = self._generate_processing_report(ocr_results, total_time)
            
            # Save processing report
            if output_path and self.save_intermediate_results:
                report_path = output_path.replace('.md', '_processing_report.json')
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Processing report saved to: {report_path}")
            
            return {
                "markdown": markdown_content,
                "processing_report": report,
                "ocr_results": ocr_results,
                "total_processing_time": total_time
            }
            
        except Exception as e:
            self.logger.error(f"PDF conversion failed: {e}")
            return {
                "markdown": f"[PDF_CONVERSION_FAILED: {str(e)}]",
                "processing_report": {"error": str(e), "total_processing_time": time.time() - processing_start},
                "ocr_results": [],
                "total_processing_time": time.time() - processing_start
            }
    
    def convert_pdf_hybrid(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """Enhanced PDF conversion with better error handling"""
        try:
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
            
            # Use PyMuPDF4LLM with production settings
            chunks = pymupdf4llm.to_markdown(
                pdf_path, 
                page_chunks=True,
                write_images=self.extract_images,
                image_path=image_path,
                image_format=self.image_format,
                dpi=self.dpi,
                image_size_limit=0.03,
                ignore_images=not self.extract_images
            )
            
            # Limit pages if specified (for testing)
            if self.max_pages:
                chunks = chunks[:self.max_pages]
                self.logger.info(f"Limited processing to first {self.max_pages} pages")
            
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
                if self.extract_images:
                    self.logger.info(f"Images saved to: {image_path}")
            
            return final_markdown
            
        except Exception as e:
            self.logger.error(f"PDF hybrid conversion failed: {e}")
            return f"[PDF_CONVERSION_FAILED: {str(e)}]"
    
    def extract_tables_with_pdfplumber(self, pdf_path: str) -> Dict[int, List[str]]:
        """Extract tables from PDF using pdfplumber with error handling"""
        tables_by_page = {}
        
        try:
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
                        
        except Exception as e:
            self.logger.error(f"Table extraction failed: {e}")
        
        return tables_by_page
    
    def _table_to_markdown(self, table: List[List[str]]) -> Optional[str]:
        """Convert table data to markdown format with error handling"""
        if not table or len(table) < 2:
            return None
        
        try:
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
            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
            return df.to_markdown(index=False)
            
        except Exception as e:
            self.logger.warning(f"Failed to convert table to markdown: {e}")
            return None
    
    def _replace_tables_in_text(self, text: str, page_num: int, 
                               pdfplumber_tables: Dict[int, List[str]]) -> str:
        """Replace detected table regions with pdfplumber-extracted tables"""
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
        """Add pdfplumber tables to text if they don't already exist"""
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
    
    def _detect_table_regions(self, text: str) -> List[Tuple[int, int]]:
        """Detect potential table regions in text"""
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
    
    def _generate_processing_report(self, ocr_results: List[OCRResult], total_time: float) -> Dict[str, Any]:
        """Generate comprehensive processing report with enhanced metrics"""
        if not ocr_results:
            return {"total_processing_time": total_time, "images_processed": 0}
        
        # Calculate statistics
        successful_extractions = [r for r in ocr_results if r.confidence >= self.quality_threshold and not r.error_message]
        failed_extractions = [r for r in ocr_results if r.error_message or r.confidence == 0.0]
        low_confidence = [r for r in ocr_results if 0.0 < r.confidence < self.quality_threshold and not r.error_message]
        
        # Document type distribution
        doc_types = {}
        for result in ocr_results:
            doc_type = result.document_type.value
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Language distribution
        languages = {}
        for result in ocr_results:
            if result.language:
                languages[result.language] = languages.get(result.language, 0) + 1
        
        # Generate enhanced report
        report = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_processing_time": total_time,
            "images_processed": len(ocr_results),
            "statistics": {
                "successful_extractions": len(successful_extractions),
                "failed_extractions": len(failed_extractions), 
                "low_confidence_extractions": len(low_confidence),
                "success_rate": len(successful_extractions) / len(ocr_results) if ocr_results else 0,
                "average_confidence": sum(r.confidence for r in ocr_results) / len(ocr_results),
                "total_words_extracted": sum(r.word_count for r in ocr_results),
                "total_characters_extracted": sum(r.character_count for r in ocr_results),
                "average_processing_time_per_image": sum(r.processing_time for r in ocr_results) / len(ocr_results)
            },
            "document_type_distribution": doc_types,
            "language_distribution": languages,
            "quality_metrics": {
                "high_confidence_count": len([r for r in ocr_results if r.confidence >= 0.9]),
                "medium_confidence_count": len([r for r in ocr_results if 0.7 <= r.confidence < 0.9]),
                "low_confidence_count": len([r for r in ocr_results if 0.3 <= r.confidence < 0.7]),
                "very_low_confidence_count": len([r for r in ocr_results if 0.0 < r.confidence < 0.3])
            },
            "content_analysis": {
                "documents_with_tables": len([r for r in ocr_results if r.has_tables]),
                "documents_with_handwriting": len([r for r in ocr_results if r.has_handwriting])
            },
            "error_analysis": {
                "quality_failures": len([r for r in ocr_results if r.error_message and "quality" in r.error_message.lower()]),
                "processing_errors": len([r for r in ocr_results if r.error_message and "error" in r.error_message.lower()])
            }
        }
        
        return report


# Enhanced convenience functions
def create_production_pdf_converter(model: str = "qwen2.5vl:7b", 
                                   max_pages: Optional[int] = None,
                                   **kwargs) -> ProductionHybridPDFConverter:
    """Create an enhanced PDF converter with specified model"""
    return ProductionHybridPDFConverter(
        ocr_model=model,
        max_pages=max_pages,
        **kwargs
    )


def convert_pdf_production(pdf_path: str, 
                          output_path: Optional[str] = None,
                          model: str = "qwen2.5vl:7b",
                          max_pages: Optional[int] = None,
                          **kwargs) -> Dict[str, Any]:
    """
    Enhanced PDF conversion with comprehensive error handling and reporting
    """
    converter = create_production_pdf_converter(
        model=model,
        max_pages=max_pages,
        **kwargs
    )
    return converter.invoke({
        "pdf_path": pdf_path,
        "output_path": output_path
    })


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with limited pages first
    result = convert_pdf_production(
        "camera.pdf", 
        "camera_fixed.md", 
        model="qwen2.5vl:7b",
        table_strategy="replace",
        extract_images=True,
        image_format="png",
        dpi=400,
        quality_threshold=0.8,
        save_intermediate_results=True,
    )