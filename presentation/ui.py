import gradio as gr

from services.chat import Chat
from services.doc_loader import DocLoader
from services.embedding import Embedding
from services.visualize import EmbeddingVisualizer


class DocChatApp:
    def __init__(self, db_name="chroma_db", model="llama2"):
        self.loader = DocLoader()
        self.embedding = Embedding(db_name=db_name)
        self.chat_engine = Chat(model=model)
        self.vectorstore = None
        self.chain = None
        self.visualizer = None
        self.embeddings_matrix = None
        self.labels = None

    def process_file(self, file) -> str:
        """Load, split, and embed uploaded file."""
        docs = self.loader.load_file_to_docs(file.name)
        chunks = self.loader.split_docs(docs)
        self.vectorstore = self.embedding.create_vector_store(chunks)

        # Save embeddings for visualization
        self.embeddings_matrix = self.embedding.embeddings.embed_documents([d.page_content for d in chunks])
        self.labels = [f"Chunk {i}" for i in range(len(chunks))]
        self.visualizer = EmbeddingVisualizer(self.embeddings_matrix)

        # Initialize conversation chain
        self.chain = self.chat_engine.get_conversation_chain(self.vectorstore)

        return f"✅ File processed: {len(chunks)} chunks embedded."

    def visualize_embeddings(self, method="pca", dim="2d"):
        if not self.embeddings_matrix:
            return "No embeddings found. Please upload a document first."

        if dim == "2d":
            self.visualizer.plot_2d(labels=self.labels, method=method)
        else:
            self.visualizer.plot_3d(labels=self.labels, method=method)
        return f"✅ Showing {dim.upper()} visualization with {method.upper()}."

    def chat(self, question: str) -> str:
        if not self.chain:
            return "Please upload a document first."
        return self.chat_engine.chat(self.chain, question)

    def launch(self):
        """Start the Gradio UI."""
        with gr.Blocks() as demo:
            gr.Markdown("# 📄 DocChat with Embeddings + Visualization")

            with gr.Row():
                file_input = gr.File(label="Upload a document", type="filepath")

            with gr.Row():
                method_dropdown = gr.Dropdown(choices=["pca", "tsne"], value="pca", label="Reduction Method")
                dim_dropdown = gr.Dropdown(choices=["2d", "3d"], value="2d", label="Visualization Type")
                visualize_btn = gr.Button("Visualize Embeddings")

            with gr.Row():
                chatbot = gr.Chatbot(label="Chat with your document", type='messages')
                msg = gr.Textbox(label="Ask a question")
                clear = gr.Button("Clear")

            # Wire functions
            file_input.change(self.process_file, inputs=file_input, outputs=chatbot)
            visualize_btn.click(self.visualize_embeddings, inputs=[method_dropdown, dim_dropdown], outputs=chatbot)
            msg.submit(self.chat, inputs=msg, outputs=chatbot)
            clear.click(lambda: None, None, chatbot, queue=False)

        demo.launch()