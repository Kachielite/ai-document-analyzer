from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class Embedding:
    def __init__(self, db_name: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.db_name = db_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

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