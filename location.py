import os
import tempfile
import requests
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.tools import DuckDuckGoSearchRun
from langchain.callbacks.base import BaseCallbackHandler

# Add import for rerun exception workaround
try:
    from streamlit.runtime.scriptrunner import RerunException
except ImportError:
    RerunException = None  # fallback if API changes in future


# ========== Location Detection ==========
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_user_location():
    """Get user's location using IP-based geolocation"""
    try:
        # Using ipapi.co for free IP geolocation
        response = requests.get('https://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'country': data.get('country_name', 'Unknown'),
                'country_code': data.get('country_code', 'XX'),
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'timezone': data.get('timezone', 'UTC')
            }
    except Exception as e:
        st.error(f"Could not detect location: {e}")
    
    # Fallback location
    return {
        'country': 'Unknown',
        'country_code': 'XX',
        'city': 'Unknown',
        'region': 'Unknown',
        'timezone': 'UTC'
    }


# ========== Location-based Web Search ==========
@st.cache_data(ttl=3600)  # Cache for 1 hour to avoid repeated searches
def get_location_info_from_web(country, region):
    """Get location-specific healthcare information from web search"""
    try:
        # Search for emergency and healthcare info
        search_query = f"{country} emergency number healthcare system aged care regulator mental health crisis hotline"
        search_results = search_tool.run(search_query)
        return search_results
    except Exception as e:
        st.error(f"Could not fetch location information: {e}")
        return None


# ========== Whisper Model Load (cached) ==========
@st.cache_resource
def load_whisper_model():
    import whisper
    return whisper.load_model("base")

whisper_model = load_whisper_model()


# ========== Whisper Voice Recording ==========
def record_audio(duration, sample_rate=16000):
    import sounddevice as sd
    from scipy.io.wavfile import write

    st.info(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    st.success("✅ Recording complete.")

    # Save to temporary file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_wav.name, sample_rate, audio)
    return temp_wav.name


def transcribe_audio(audio_path):
    try:
        file_size = os.path.getsize(audio_path)
        if file_size < 1000:
            st.error("Audio file too small or empty. Please try recording again.")
            return ""

        st.info("Transcribing audio...")
        result = whisper_model.transcribe(audio_path)
        st.success("Transcription complete.")
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""


# ========== Callback Handler for Streaming ==========
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")


# ========== Enhanced System Prompt with Location Context ==========
def create_localized_system_prompt(context_chunks, user_location, location_web_info=None):
    """Create a system prompt with location context for web-based localization"""
    
    base_prompt = ("You are an expert AI assistant trained to support care workers, clinicians and aged care staff"
            "Your goal is to always provide detailed, accurate and well-structured answers to the user's question."
            "Use the following context from care documentation and policies to answer the user's question.\n\n"
            "Always provide detailed care plans with  when asked.\n\n"
            "Always focus on finer details of nutrition recommedations when appropriate"
            "Always explain key terms clearly. Where appropriate, include:\n"
            "- Step-by-step instructions\n"
            "- Bullet-point lists\n"
            "- Definitions or examples\n\n"
    )
    
    # Add location-specific information
    location_prompt = (
        f"IMPORTANT: The user is located in {user_location['city']}, {user_location['region']}, "
        f"{user_location['country']} ({user_location['country_code']}). "
        f"Please provide responses that are relevant to {user_location['country']} healthcare and aged care systems.\n\n"
    )
    
    # Add web-searched location info if available
    if location_web_info:
        location_prompt += f"Location-specific information from web search:\n{location_web_info}\n\n"
    
    instruction_prompt = (
        "When providing information:\n"
        "1. Always reference local regulations, standards, and contact information when relevant\n"
        "2. Include local emergency numbers and crisis contact information\n"
        "3. Reference local healthcare systems and aged care regulations\n"
        "4. Use appropriate currency and formats for the user's location\n"
        "5. Always provide local services and community services information and contact details\n\n"
        "6.Always provide detailed, accurate and well-structured answers to the user's question.\n\n"
        "9.Always provide detailed care plans when asked.\n\n"
        "Always focus on finer details of nutrition recommedations when appropriate\n\n"
        "7. Always explain key terms clearly and include step-by-step instructions when appropriate\n"
        "8. Always use bullet-point lists and provide definitions or examples where appropriate\n\n"

        
        f"Context from care documentation:\n{context_chunks}\n\n"
        
        "Always prioritize safety and encourage users to contact local emergency services "
        "in case of immediate danger or medical emergencies. If you don't know the specific "
        "emergency number for their location, advise them to search for local emergency services."
    )
    
    return base_prompt + location_prompt + instruction_prompt


# ========== Process User Input Function ==========
def process_user_input(user_input):
    """Process user input and generate localized response"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.chat_message("user", avatar="👤").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.timestamps.append(timestamp)

    # Get user location
    user_location = st.session_state.get('user_location', get_user_location())
    
    # Get location-specific info from web if needed
    location_web_info = None
    if any(keyword in user_input.lower() for keyword in ['emergency', 'crisis', 'help', 'contact', 'number', 'regulation', 'standard']):
        with st.spinner("🌍 Getting location-specific information..."):
            location_web_info = get_location_info_from_web(user_location['country'], user_location['region'])

    with st.spinner("🔍 Searching for relevant information..."):
        results = retriever.get_relevant_documents(user_input)

    if results:
        context_chunks = "\n\n".join([doc.page_content for doc in results])
        system_prompt = create_localized_system_prompt(context_chunks, user_location, location_web_info)
    else:
        with st.spinner("🌐 Searching the web..."):
            # Include location in web search for more relevant results
            location_context = f"{user_location['country']} {user_location['region']}"
            search_query = f"{user_input} {location_context} aged care healthcare"
            web_results = search_tool.run(search_query)

        system_prompt = (
            f"You are an expert AI assistant trained to support care workers, clinicians and aged care staff in "
            f"{user_location['city']}, {user_location['region']}, {user_location['country']}.\n\n"
            
            f"Always provide detailed, accurate and well-structured answers that are relevant to "
            f"{user_location['country']} healthcare and aged care systems. Include local emergency contacts, "
            f"regulations, and standards when relevant.\n\n"

            f" Always provide local services and community services information and contact details\n\n"

            f"Always provide detailed care plans when asked.\n\n"
            f"Always focus on finer details of nutrition recommedations when appropriate.\n\n"


            
            f"Always explain key terms clearly and include step-by-step instructions, Bullet-point lists and definitions or examples when appropriate.\n\n"
        )
        
        if location_web_info:
            system_prompt += f"Location-specific information:\n{location_web_info}\n\n"
        
        system_prompt += f"Web search results:\n{web_results}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

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


# ========== Load Environment ==========
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")  # add this to your .env

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
    st.image("C:\\Users\\narta\\OneDrive\\Pictures\\Camera Roll\\Glorious Picture.png", width=100)
with col2:
    st.title("Glorious AI Assistant")

# ========== Location Detection and Display ==========
if 'user_location' not in st.session_state:
    with st.spinner("🌍 Detecting your location..."):
        st.session_state.user_location = get_user_location()

user_location = st.session_state.user_location

# Display location info
st.markdown("Ask a question about mental health assessments, aged care workflows, or general topics.")

# ========== Chat Session State ==========
for key in ["chat_history", "messages", "timestamps", "feedback"]:
    if key not in st.session_state:
        st.session_state[key] = []

# Add flag for clearing chat
if "clear_chat" not in st.session_state:
    st.session_state["clear_chat"] = False

# Initialize voice input flag
if "process_voice_input" not in st.session_state:
    st.session_state["process_voice_input"] = False

# ========== Sidebar: Settings & Voice Input ==========
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Location settings
    st.subheader("📍 Location")
    st.write(f"**Country:** {user_location['country']}")
    st.write(f"**Region:** {user_location['region']}")
    st.write(f"**City:** {user_location['city']}")
    
    # Manual location override
    if st.button("🔄 Refresh Location"):
        st.session_state.user_location = get_user_location()
        st.rerun()

    if st.button("🔄 Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.timestamps = []
        st.session_state.feedback = []
        st.rerun()

    if st.session_state.get("messages"):
        chat_text = f"Chat History - {user_location['city']}, {user_location['country']}\n"
        chat_text += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for msg, time in zip(st.session_state.messages, st.session_state.timestamps):
            role = "User" if isinstance(msg, HumanMessage) else "Bot" if isinstance(msg, AIMessage) else "System"
            chat_text += f"{time} - {role}:\n{msg.content}\n\n"

        st.download_button("📥 Download Chat", data=chat_text, file_name=f"chat_history_{user_location['country_code']}.txt")

    st.subheader("🎙️ Voice Input")
    record = st.button("🔴 Record Voice")
    record_duration = st.slider("Recording duration (seconds)", 5, 60, 10)

    if record:
        audio_path = record_audio(record_duration)
        transcription = transcribe_audio(audio_path)
        if transcription:
            st.session_state.voice_input = transcription
            st.session_state.process_voice_input = True
            st.success("📝 Transcription: " + transcription)
            st.rerun()

# ========== Handle clear chat flag and force rerun ==========
if st.session_state.get("clear_chat", False):
    st.session_state["clear_chat"] = False
    if RerunException:
        raise RerunException(st.script_runner.RerunData())
    else:
        st.error("Unable to rerun app automatically. Please refresh the page manually.")

# ========== Display Chat History ==========
for msg in st.session_state.chat_history:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ========== Process Voice Input Automatically ==========
if st.session_state.get("process_voice_input", False) and st.session_state.get("voice_input"):
    voice_text = st.session_state.voice_input
    st.session_state.process_voice_input = False
    # Clean up the voice input from session state
    del st.session_state.voice_input
    
    # Process the voice input automatically
    process_user_input(voice_text)
    st.rerun()

# ========== Chat Input ==========
user_input = st.chat_input("Type your question here...")

if user_input:
    process_user_input(user_input)