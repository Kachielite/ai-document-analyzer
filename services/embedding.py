from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil
import os


class Embedding:
    def __init__(self, db_name: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.db_name = db_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    def clear_vector_store(self):
        """Clear the existing vector database directory and all its contents."""
        try:
            if os.path.exists(self.db_name):
                shutil.rmtree(self.db_name)
                print(f"✅ Cleared vector database: {self.db_name}")
            else:
                print(f"ℹ️ No existing vector database found at: {self.db_name}")
        except Exception as e:
            print(f"❌ Error clearing vector database: {str(e)}")

    def create_vector_store(self, documents):
        try:
            if not documents:
                raise ValueError("No documents provided for embedding.")

            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.db_name
            )
            print(f"Vectorstore created with {vectorstore._collection.count()} documents")
            return vectorstore

        except Exception as e:
            print(f"Error while creating vectorstore: {str(e)}")
            return None