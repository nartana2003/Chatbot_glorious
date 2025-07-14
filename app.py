import os
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime

import pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.tools import DuckDuckGoSearchRun
from langchain.callbacks.base import BaseCallbackHandler

# ========== Streaming Callback ==========
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# ========== ENV ==========
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# ========== Pinecone Setup ==========
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("langchain-glorious-chatbot-index")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=openai_api_key,
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.6},
)

search_tool = DuckDuckGoSearchRun()

# ========== Streamlit UI ==========
st.set_page_config(page_title="Glorious Chatbot")
col1, col2 = st.columns([1, 4])
with col1:
    st.image("C:\\Users\\narta\\OneDrive\\Pictures\\Camera Roll\\Glorious Picture.png",width=100)
with col2:
    st.title("Glorious AI Assistant")

st.markdown("Ask a question about mental health assessments, aged care workflows, or general topics.")

# ========== Chat State ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# ========== Sidebar ==========
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🔄 Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.timestamps = []
        st.session_state.feedback = []
        st.rerun()

    if st.session_state.get("messages"):
        chat_text = ""
        for msg, time in zip(st.session_state.messages, st.session_state.timestamps):
            role = (
                "User" if isinstance(msg, HumanMessage)
                else "Bot" if isinstance(msg, AIMessage)
                else "System"
            )
            chat_text += f"{time} - {role}:\n{msg.content}\n\n"

        st.download_button("📥 Download Chat", data=chat_text, file_name="chat_history.txt")

# ========== Display Chat History ==========
for msg in st.session_state.chat_history:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# ========== Handle User Input ==========
user_input = st.chat_input("Type your question here...")

if user_input:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.chat_message("user", avatar="👤").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.timestamps.append(timestamp)

    with st.spinner("Searching for relevant information..."):
        results = retriever.get_relevant_documents(user_input)

    if results:
        context_chunks = "\n\n".join([doc.page_content for doc in results])
        system_prompt = (
            "You are an expert AI assistant trained to support care workers, clinicians and aged care staff. "
            "Your goal is to provide detailed, accurate and well-structured answers to the user's question. "
            "Use the following context from care documentation and policies to answer the user's question.\n\n"
            "Always explain key terms clearly. Where appropriate, include:\n"
            "- Step-by-step instructions\n"
            "- Bullet-point lists\n"
            "- Definitions or examples\n\n"
            f"Context:\n{context_chunks}"
        )
    else:
        with st.spinner("Searching the web..."):
            web_results = search_tool.run(user_input)

        system_prompt = (
            "You are an expert AI assistant trained to support care workers, clinicians and aged care staff. "
            "The following information was found online."
            "Your goal is to provide detailed, accurate and well-structured answers to the user's question."
            "Always explain key terms clearly. Where appropriate, include:\n"
            "- Step-by-step instructions\n"
            "- Bullet-point lists\n"
            "- Definitions or examples\n\n"


            f"Web Results:\n{web_results}"
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    # ========== Stream GPT Response ==========
    with st.chat_message("assistant", avatar="🤖"):
        stream_container = st.empty()
        stream_handler = StreamlitCallbackHandler(stream_container)

        chat_model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5,
            api_key=openai_api_key,
            streaming=True,
            callbacks=[stream_handler]
        )

        chat_model.invoke(messages)

    assistant_reply = stream_handler.text
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
    st.session_state.messages.append(AIMessage(content=assistant_reply))
    st.session_state.timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
