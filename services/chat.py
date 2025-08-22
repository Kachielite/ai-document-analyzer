from langchain_community.llms import Ollama
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


class Chat:
    def __init__(self, model: str):
        self.model = model

    def get_conversation_chain(self, vectorstore):
        if not vectorstore:
            raise ValueError("Vectorstore is required to initialize the conversation chain.")

        llm = Ollama(model=self.model)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
        )

    def chat(self, chain: ConversationalRetrievalChain, question: str):
        if not chain:
            raise ValueError("Conversation chain is not initialized.")
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        try:
            response = chain.invoke({"question": question})
            answer = response.get("answer", "No answer generated.")
            return answer
        except Exception as e:
            print(f"Error during chat interaction: {str(e)}")
            return "Sorry, I encountered an error while processing your request."


