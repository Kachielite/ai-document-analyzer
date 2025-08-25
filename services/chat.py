try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain



class Chat:
    def __init__(self, model: str):
        self.model = model

    def get_conversation_chain(self, vectorstore):
        llm = OllamaLLM(model=self.model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

        # Create the base chain without memory first
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
        )

        return chain

    def _convert_gradio_history_to_langchain(self, gradio_history):
        """Convert Gradio chat history format to LangChain format"""
        chat_history = []
        if gradio_history:
            for i in range(0, len(gradio_history), 2):
                if i + 1 < len(gradio_history):
                    user_msg = gradio_history[i].get('content', '') if isinstance(gradio_history[i], dict) else str(gradio_history[i])
                    ai_msg = gradio_history[i + 1].get('content', '') if isinstance(gradio_history[i + 1], dict) else str(gradio_history[i + 1])
                    chat_history.append((user_msg, ai_msg))
        return chat_history

    def chat_with_document(self, chain, message, gradio_history=None):
        """Handle complete chat interaction with document"""
        if not chain:
            return "Please upload a document first."

        if not message or not message.strip():
            return "Please enter a question."

        try:
            # Convert Gradio history to LangChain format
            chat_history = self._convert_gradio_history_to_langchain(gradio_history)

            # Get response from chain
            response = chain.invoke({
                "question": message,
                "chat_history": chat_history
            })

            return response.get('answer', 'No answer generated.')

        except Exception as e:
            print(f"Error during chat interaction: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

    def chat(self, chain: ConversationalRetrievalChain, question: str, chat_history=None):
        """Legacy method for backwards compatibility"""
        if not chain:
            raise ValueError("Conversation chain is not initialized.")
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Provide empty chat history if none provided
        if chat_history is None:
            chat_history = []

        try:
            response = chain.invoke({
                "question": question,
                "chat_history": chat_history
            })
        except Exception as e:
            print(f"Error during chat interaction: {str(e)}")
            return "Sorry, I encountered an error while processing your request."

        answer = response.get('answer', 'No answer generated.')
        return answer
