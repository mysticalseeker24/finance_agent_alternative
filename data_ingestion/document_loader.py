"""Document loader for processing PDFs and extracting text from financial documents."""

import os
from typing import Dict, List, Any, Optional, Union, BinaryIO
import io

import pdfplumber
import PyPDF2
from loguru import logger


class DocumentLoader:
    """Loader for processing various document formats."""

    def __init__(self, output_dir: str = "processed_documents"):
        """Initialize the document loader.

        Args:
            output_dir: Directory to store processed document data.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Document loader initialized with output directory: {output_dir}")

    def extract_text_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract text from a PDF file using PDFPlumber.

        This method handles complex PDFs with tables and formatted text.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A dictionary containing extracted text and metadata.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": f"File not found: {file_path}"}

        try:
            # Extract text using PDFPlumber
            pages_text = []
            tables = []

            with pdfplumber.open(file_path) as pdf:
                # Get document metadata
                metadata = {
                    "total_pages": len(pdf.pages),
                    "document_name": os.path.basename(file_path),
                }

                # Process each page
                for i, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text() or ""
                    pages_text.append({"page_number": i + 1, "text": text})

                    # Extract tables
                    page_tables = page.extract_tables()
                    if page_tables:
                        for j, table in enumerate(page_tables):
                            # Convert table to a list of dictionaries
                            if (
                                table and len(table) > 1
                            ):  # Ensure table has headers and data
                                headers = table[0]
                                rows = table[1:]
                                structured_table = []

                                for row in rows:
                                    row_dict = {}
                                    for k, header in enumerate(headers):
                                        if (
                                            k < len(row) and header
                                        ):  # Ensure header exists
                                            row_dict[header] = row[k]
                                    structured_table.append(row_dict)

                                tables.append(
                                    {
                                        "page_number": i + 1,
                                        "table_number": j + 1,
                                        "data": structured_table,
                                    }
                                )

            # Combine results
            result = {
                "metadata": metadata,
                "pages": pages_text,
                "tables": tables,
                "source": file_path,
                "extraction_method": "pdfplumber",
            }

            logger.info(
                f"Successfully extracted text from {file_path} using PDFPlumber"
            )
            return result

        except Exception as e:
            logger.error(
                f"Error extracting text from {file_path} using PDFPlumber: {str(e)}"
            )
            # Fallback to PyPDF2
            logger.info(f"Falling back to PyPDF2 for {file_path}")
            return self.extract_text_from_pdf_simple(file_path)

    def extract_text_from_pdf_simple(self, file_path: str) -> Dict[str, Any]:
        """Extract text from a PDF file using PyPDF2.

        This is a simpler extraction method that works as a fallback.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A dictionary containing extracted text and metadata.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": f"File not found: {file_path}"}

        try:
            # Extract text using PyPDF2
            pages_text = []

            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)

                # Get document metadata
                metadata = {
                    "total_pages": len(reader.pages),
                    "document_name": os.path.basename(file_path),
                }

                # Process each page
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    pages_text.append({"page_number": i + 1, "text": text})

            # Combine results
            result = {
                "metadata": metadata,
                "pages": pages_text,
                "tables": [],  # PyPDF2 doesn't extract tables
                "source": file_path,
                "extraction_method": "pypdf2",
            }

            logger.info(f"Successfully extracted text from {file_path} using PyPDF2")
            return result

        except Exception as e:
            logger.error(
                f"Error extracting text from {file_path} using PyPDF2: {str(e)}"
            )
            return {"error": str(e), "source": file_path}

    def extract_text_from_bytes(
        self, file_bytes: Union[bytes, BinaryIO], filename: str
    ) -> Dict[str, Any]:
        """Extract text from PDF bytes.

        This is useful for processing in-memory PDFs from API responses.

        Args:
            file_bytes: The PDF file as bytes or file-like object.
            filename: A name for the file (used for logging and metadata).

        Returns:
            A dictionary containing extracted text and metadata.
        """
        try:
            # Try PDFPlumber first
            if isinstance(file_bytes, bytes):
                file_bytes = io.BytesIO(file_bytes)

            pages_text = []
            tables = []

            with pdfplumber.open(file_bytes) as pdf:
                # Get document metadata
                metadata = {
                    "total_pages": len(pdf.pages),
                    "document_name": filename,
                }

                # Process each page
                for i, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text() or ""
                    pages_text.append({"page_number": i + 1, "text": text})

                    # Extract tables
                    page_tables = page.extract_tables()
                    if page_tables:
                        for j, table in enumerate(page_tables):
                            # Convert table to a list of dictionaries
                            if (
                                table and len(table) > 1
                            ):  # Ensure table has headers and data
                                headers = table[0]
                                rows = table[1:]
                                structured_table = []

                                for row in rows:
                                    row_dict = {}
                                    for k, header in enumerate(headers):
                                        if (
                                            k < len(row) and header
                                        ):  # Ensure header exists
                                            row_dict[header] = row[k]
                                    structured_table.append(row_dict)

                                tables.append(
                                    {
                                        "page_number": i + 1,
                                        "table_number": j + 1,
                                        "data": structured_table,
                                    }
                                )

            # Combine results
            result = {
                "metadata": metadata,
                "pages": pages_text,
                "tables": tables,
                "source": filename,
                "extraction_method": "pdfplumber",
            }

            logger.info(f"Successfully extracted text from {filename} using PDFPlumber")
            return result

        except Exception as e:
            logger.error(
                f"Error extracting text from {filename} using PDFPlumber: {str(e)}"
            )
            # Fallback to PyPDF2
            logger.info(f"Falling back to PyPDF2 for {filename}")

            try:
                # Reset file pointer if it's a file-like object
                if hasattr(file_bytes, "seek"):
                    file_bytes.seek(0)

                # Extract text using PyPDF2
                if isinstance(file_bytes, bytes):
                    file_bytes = io.BytesIO(file_bytes)

                pages_text = []
                reader = PyPDF2.PdfReader(file_bytes)

                # Get document metadata
                metadata = {
                    "total_pages": len(reader.pages),
                    "document_name": filename,
                }

                # Process each page
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    pages_text.append({"page_number": i + 1, "text": text})

                # Combine results
                result = {
                    "metadata": metadata,
                    "pages": pages_text,
                    "tables": [],  # PyPDF2 doesn't extract tables
                    "source": filename,
                    "extraction_method": "pypdf2",
                }

                logger.info(f"Successfully extracted text from {filename} using PyPDF2")
                return result

            except Exception as e2:
                logger.error(
                    f"Error extracting text from {filename} using PyPDF2: {str(e2)}"
                )
                return {"error": str(e2), "source": filename}

    def extract_structured_data(self, file_path: str) -> Dict[str, Any]:
        """Extract structured data from financial documents.

        This method attempts to identify and extract common financial data points
        such as balance sheets, income statements, and cash flow statements.

        Args:
            file_path: Path to the document file.

        Returns:
            A dictionary containing structured financial data.
        """
        # First extract all text and tables
        extraction_result = self.extract_text_from_pdf(file_path)

        if "error" in extraction_result:
            return extraction_result

        # Initialize structured data
        structured_data = {
            "metadata": extraction_result["metadata"],
            "financial_data": {},
            "source": file_path,
        }

        # Look for common financial terms and sections
        financial_terms = {
            "balance_sheet": ["balance sheet", "assets", "liabilities", "equity"],
            "income_statement": [
                "income statement",
                "revenue",
                "earnings",
                "profit",
                "loss",
            ],
            "cash_flow": [
                "cash flow",
                "operating activities",
                "investing activities",
                "financing activities",
            ],
        }

        # Extract structured data from tables
        if extraction_result["tables"]:
            for table in extraction_result["tables"]:
                # Analyze table headers to determine the type
                table_type = None
                table_data = table["data"]

                if not table_data:  # Skip empty tables
                    continue

                # Check the first row for financial terms
                first_row = table_data[0]
                row_text = " ".join(str(value).lower() for value in first_row.values())

                for financial_type, terms in financial_terms.items():
                    if any(term in row_text for term in terms):
                        table_type = financial_type
                        break

                # If a financial table is identified, add it to structured data
                if table_type:
                    if table_type not in structured_data["financial_data"]:
                        structured_data["financial_data"][table_type] = []

                    structured_data["financial_data"][table_type].append(
                        {
                            "page_number": table["page_number"],
                            "data": table_data,
                        }
                    )

        # If no structured data was found in tables, try text-based extraction
        if not structured_data["financial_data"]:
            # Extract text from all pages
            all_text = " ".join(
                page["text"].lower() for page in extraction_result["pages"]
            )

            # Look for sections in the text
            for financial_type, terms in financial_terms.items():
                for term in terms:
                    if term in all_text:
                        structured_data["financial_data"][financial_type] = {
                            "identified": True,
                            "extraction_method": "text_based",
                            "confidence": "low",  # Text-based extraction is less reliable
                        }

        logger.info(f"Extracted structured data from {file_path}")
        return structured_data


if __name__ == "__main__":
    # Example usage
    loader = DocumentLoader()
    # Assuming a PDF file exists
    # result = loader.extract_text_from_pdf("example.pdf")
    # print(f"Extracted {len(result['pages'])} pages")
