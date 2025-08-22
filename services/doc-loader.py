import os
from typing import List

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
            loader = CSVLoader(path)
            docs = loader.load()
        elif ext in (".txt",):
            loader = TextLoader(path, autodetect_encoding=True)
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        if not docs:
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: pdf, docx, md, csv, txt")

        # Normalize metadata
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata.setdefault("source", os.path.basename(path))
        return docs

    def split_docs(self, docs: List[Document]) -> List[Document]:
        """Chunk documents into overlapping windows for dense retrieval."""
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n"
        )
        return splitter.split_documents(docs)