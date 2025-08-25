# AI Document Analyzer - RAG Learning Project

## 🎯 Project Overview

This project is designed for **learning and mastering Retrieval-Augmented Generation (RAG)** - a powerful AI technique that combines information retrieval with language generation. Through building an AI document analyzer, you'll explore the core concepts of RAG including document processing, vector embeddings, similarity search, and context-aware response generation.

## 📚 What You'll Learn About RAG

- **Document Processing**: How to load, parse, and chunk documents for optimal retrieval
- **Vector Embeddings**: Converting text into numerical representations for semantic search
- **Vector Databases**: Storing and querying embeddings efficiently with ChromaDB
- **Retrieval Mechanisms**: Finding relevant context based on user queries
- **Context Injection**: Providing retrieved information to language models for better responses
- **Conversational Memory**: Maintaining context across chat interactions
- **Embedding Visualization**: Understanding how documents are represented in vector space

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   Processing    │───▶│   Vector Store  │
│  (PDF, DOCX,    │    │  (Chunking &    │    │   (ChromaDB)    │
│   TXT, etc.)    │    │   Embedding)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Response      │◀───│   LLM (Ollama)  │◀───│   Retrieval     │
│   Generation    │    │                 │    │   (Context)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## ✨ Features

### Core RAG Functionality
- **Multi-format Document Support**: PDF, DOCX, TXT, CSV, Excel, Markdown
- **Intelligent Text Chunking**: Configurable chunk size and overlap for optimal retrieval
- **Semantic Embeddings**: Using HuggingFace sentence-transformers for vector representations
- **Vector Database**: ChromaDB for persistent and efficient similarity search
- **Conversational Retrieval**: Memory-enabled chat with context from uploaded documents
- **Local LLM Integration**: Ollama support for privacy-focused AI interactions

### Learning & Analysis Tools
- **Embedding Visualization**: 2D/3D plots to understand document clustering
- **Interactive UI**: Gradio-based interface for hands-on experimentation
- **Retrieval Analytics**: See exactly which chunks are retrieved for each query
- **Real-time Processing**: Upload documents and see RAG in action immediately

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed locally ([Installation Guide](https://ollama.ai/))
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CustomerServiceBot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull an Ollama model**
   ```bash
   ollama pull llama2  # or your preferred model
   ```

### Running the Application

```bash
python app.py
```

The Gradio interface will open in your browser at `http://localhost:7860`

## 🎓 Learning Path

### 1. **Basic RAG Concepts**
   - Start by uploading a simple text document
   - Ask questions and observe how relevant chunks are retrieved
   - Experiment with different chunk sizes in `services/doc_loader.py`

### 2. **Embeddings Understanding**
   - Use the visualization features to see how documents cluster
   - Try different types of documents and observe embedding patterns
   - Experiment with different embedding models

### 3. **Retrieval Optimization**
   - Modify the retrieval parameters (k value, similarity thresholds)
   - Test different text splitting strategies
   - Compare retrieval quality with various document types

### 4. **Advanced Features**
   - Implement custom document loaders
   - Add metadata filtering
   - Experiment with different LLM models
   - Enhance the conversation memory system

## 📁 Project Structure

```
CustomerServiceBot/
├── app.py                 # Main application entry point
├── requirements.txt       # Project dependencies
├── chroma_db/            # Vector database storage
├── presentation/
│   └── ui.py            # Gradio interface
├── services/
│   ├── chat.py          # Conversational RAG chain
│   ├── doc_loader.py    # Document processing & chunking
│   ├── embedding.py     # Vector embeddings & ChromaDB
│   └── visualize.py     # Embedding visualization tools
```

## 🔧 Configuration

### Document Processing
```python
# In services/doc_loader.py
DocLoader(
    chunk_size=1000,      # Size of text chunks
    chunk_overlap=150     # Overlap between chunks
)
```

### Embedding Settings
```python
# In services/embedding.py
model_name = "all-MiniLM-L6-v2"  # HuggingFace embedding model
```

### Chat Configuration
```python
# In services/chat.py
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 25}  # Number of chunks to retrieve
)
```

## 📊 Understanding RAG Components

### 1. **Document Loading** (`services/doc_loader.py`)
Handles various file formats and converts them into processable text chunks. Key learning points:
- How different file formats are processed
- The importance of text chunking strategy
- Balancing chunk size vs. context preservation

### 2. **Embeddings** (`services/embedding.py`)
Creates vector representations of text for semantic search. Learn about:
- How text becomes numbers (vector embeddings)
- Similarity search in high-dimensional spaces
- Vector database operations

### 3. **Chat Engine** (`services/chat.py`)
Combines retrieval with generation. Explore:
- How retrieved context is injected into prompts
- Conversational memory management
- Chain-of-thought reasoning with retrieved information

### 4. **Visualization** (`services/visualize.py`)
Helps understand the embedding space. Discover:
- How documents cluster in vector space
- The impact of different embedding models
- Dimensionality reduction techniques (PCA, t-SNE)

## 🛠️ Customization Examples

### Add New Document Types
```python
# In services/doc_loader.py
def load_custom_format(self, path: str) -> List[Document]:
    # Your custom loader implementation
    pass
```

### Change Embedding Models
```python
# In services/embedding.py
model_name = "sentence-transformers/all-mpnet-base-v2"  # Better quality
# or
model_name = "sentence-transformers/all-distilroberta-v1"  # Faster
```

### Modify Retrieval Strategy
```python
# In services/chat.py
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={"k": 10, "fetch_k": 20}
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check if your model is available: `ollama list`

2. **Memory Issues with Large Documents**
   - Reduce chunk_size in DocLoader
   - Process documents in smaller batches

3. **Poor Retrieval Quality**
   - Experiment with different chunk sizes
   - Try different embedding models
   - Adjust the number of retrieved chunks (k parameter)

## 🤝 Contributing

This is a learning project! Feel free to:
- Experiment with different RAG techniques
- Add new document types support
- Improve visualization features
- Enhance the UI/UX
- Add evaluation metrics for retrieval quality

## 📖 Further Reading

- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Retrieval-Augmented Generation Paper](https://arxiv.org/abs/2005.11401)


---

**Happy Learning!** 🎉 Dive deep into RAG concepts and build your understanding through hands-on experimentation with this comprehensive learning project.
