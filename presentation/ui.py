import gradio as gr

from services.chat import Chat
from services.doc_loader import DocLoader
from services.embedding import Embedding
from services.visualize import EmbeddingVisualizer


class DocChatApp:
    def __init__(self, db_name="chroma_db", model="llama2"):
        self.loader = DocLoader()
        self.embedding = Embedding(db_name=db_name)
        # Clear any existing vector database on startup
        self.embedding.clear_vector_store()
        self.chat_engine = Chat(model=model)
        self.vectorstore = None
        self.chain = None
        self.visualizer = None
        self.embeddings_matrix = None
        self.labels = None

    def process_file(self, file) -> list:
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

        return [{"role": "assistant", "content": f"✅ File processed: {len(chunks)} chunks embedded. You can now ask questions about your document!"}]

    def visualize_embeddings(self, history, method="pca", dim="2d"):
        if not self.embeddings_matrix:
            message = "No embeddings found. Please upload a document first."
            plot = None
        else:
            if dim == "2d":
                self.visualizer.plot_2d(labels=self.labels, method=method)
            else:
                self.visualizer.plot_3d(labels=self.labels, method=method)
            message = f"✅ Showing {dim.upper()} visualization with {method.upper()}."

        # Add the message to existing history
        if history is None:
            history = []
        history.append({"role": "assistant", "content": message})
        return history

    def visualize_embeddings_with_plot(self, history, method="pca", dim="2d"):
        """Create visualization and return both chat message and plot"""
        if not self.embeddings_matrix:
            message = "No embeddings found. Please upload a document first."
            plot = None
        else:
            if dim == "2d":
                plot = self.visualizer.plot_2d(labels=self.labels, method=method)
            else:
                plot = self.visualizer.plot_3d(labels=self.labels, method=method)
            message = f"✅ Showing {dim.upper()} visualization with {method.upper()}."

        # Add the message to existing history
        if history is None:
            history = []
        history.append({"role": "assistant", "content": message})
        return history, plot

    def chat_with_document(self, message, history):
        """Handle chat interaction with proper message format"""
        # Delegate all chat logic to the Chat service
        response = self.chat_engine.chat_with_document(self.chain, message, history)

        # Add user message and assistant response to history
        if history is None:
            history = []

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        return history, ""  # Return updated history and clear the input

    def launch(self):
        """Start the Gradio UI."""
        with gr.Blocks() as demo:
            gr.Markdown("# 📄 AI Document Analyzer - RAG Learning Tool")

            with gr.Row():
                # Left Column - File Upload and Visualization
                with gr.Column(scale=1):
                    gr.Markdown("### Document Processing & Visualization")

                    # File upload
                    file_input = gr.File(label="Upload a document", type="filepath")

                    # Visualization controls
                    with gr.Row():
                        method_dropdown = gr.Dropdown(
                            choices=["pca", "tsne"],
                            value="pca",
                            label="Reduction Method"
                        )
                        dim_dropdown = gr.Dropdown(
                            choices=["2d", "3d"],
                            value="2d",
                            label="Visualization Type"
                        )

                    with gr.Row():
                        visualize_btn = gr.Button("Visualize Embeddings", variant="primary")
                        clear_viz_btn = gr.Button("Clear Visualization")

                    # Graph/Plot area
                    plot_output = gr.Plot(label="Embedding Visualization")

                # Right Column - Chat Interface
                with gr.Column(scale=1):
                    gr.Markdown("### Chat with Document")

                    chatbot = gr.Chatbot(
                        label="Ask questions about your document",
                        type='messages',
                        height=500
                    )

                    msg = gr.Textbox(
                        label="Ask a question",
                        placeholder="Type your question here...",
                        scale=4
                    )

                    with gr.Row():
                        send_btn = gr.Button("Send", scale=1, variant="primary")
                        clear_chat_btn = gr.Button("Clear Chat")



            # Wire functions
            file_input.change(self.process_file, inputs=file_input, outputs=chatbot)

            visualize_btn.click(
                self.visualize_embeddings_with_plot,
                inputs=[chatbot, method_dropdown, dim_dropdown],
                outputs=[chatbot, plot_output]
            )

            msg.submit(
                self.chat_with_document,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )

            send_btn.click(
                self.chat_with_document,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )

            clear_chat_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
            clear_viz_btn.click(lambda: None, outputs=plot_output)

        demo.launch()