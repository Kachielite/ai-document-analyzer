class Embedding:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_model = self.load_model()

    def load_model(self):
        # Placeholder for loading the actual embedding model
        return f"Model {self.model_name} loaded"

    def embed_text(self, text: str) -> list:
        # Placeholder for embedding logic
        return [0.1, 0.2, 0.3]  # Example embedding vector

    def get_embedding(self, text: str) -> list:
        return self.embed_text(text)