import os
from typing import List
import pandas as pd

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    TextLoader,
)


class DocLoader:
    def __init__(self, chunk_size=1000, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def load_excel_to_docs(path: str) -> List[Document]:
        """Load Excel files (.xlsx or .xls) and convert to Documents."""
        try:
            # Determine the appropriate engine based on file extension
            file_ext = os.path.splitext(path)[1].lower()
            engine = None

            if file_ext == '.xlsx':
                engine = 'openpyxl'
            elif file_ext == '.xls':
                engine = 'xlrd'

            # Read all sheets from the Excel file
            try:
                df = pd.read_excel(path, sheet_name=None, engine=engine)
            except ImportError as e:
                if 'xlrd' in str(e):
                    raise ValueError(
                        f"Cannot read .xls files. Please install xlrd: pip install xlrd>=2.0.1"
                    )
                elif 'openpyxl' in str(e):
                    raise ValueError(
                        f"Cannot read .xlsx files. Please install openpyxl: pip install openpyxl>=3.1.0"
                    )
                else:
                    raise ValueError(f"Missing Excel dependency: {str(e)}")

            docs = []
            for sheet_name, sheet_df in df.items():
                # Convert DataFrame to string representation
                content = sheet_df.to_string(index=False)

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": os.path.basename(path),
                        "sheet": sheet_name,
                        "rows": len(sheet_df),
                        "columns": len(sheet_df.columns)
                    }
                )
                docs.append(doc)

            return docs
        except Exception as e:
            if "Missing Excel dependency" in str(e) or "Cannot read" in str(e):
                raise e  # Re-raise our custom error messages
            raise ValueError(f"Error loading Excel file: {str(e)}")

    @staticmethod
    def load_file_to_docs(path: str) -> List[Document]:
        """Pick the right loader based on file extension and return Documents."""
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(path)
            docs = loader.load()
        elif ext in (".docx",):
            loader = Docx2txtLoader(path)
            docs = loader.load()
        elif ext in (".md",):
            loader = UnstructuredMarkdownLoader(path)
            docs = loader.load()
        elif ext in (".csv",):
            loader = CSVLoader(path, autodetect_encoding=True)
            docs = loader.load()
        elif ext in (".xlsx", ".xls"):
            # Use our custom Excel loader instead of CSV loader
            docs = DocLoader.load_excel_to_docs(path)
        elif ext in (".txt",):
            loader = TextLoader(path, autodetect_encoding=True)
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        if not docs:
            raise ValueError(f"No content found in file '{path}'")

        # Normalize metadata
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata.setdefault("source", os.path.basename(path))

        print(f"Loaded {len(docs)} documents from {path}")
        return docs

    def split_docs(self, docs: List[Document]) -> List[Document]:
        """Chunk documents into overlapping windows for dense retrieval."""
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n"
        )
        if not docs:
            raise ValueError("No documents to split. Please load documents first.")
        splitted_docs = splitter.split_documents(docs)
        print(f"Split into {len(splitted_docs)} chunks.")
        return splitted_docs