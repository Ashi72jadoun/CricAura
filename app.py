import streamlit as st
import json
import os
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

# 1. Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# 2. Initialize the AI and Search clients
client = Groq(api_key=groq_api_key)
tavily_client = TavilyClient(api_key=tavily_api_key)

# 3. Load local mock cricket data from the JSON file
try:
    with open("mock_cricket_data.json", "r") as file:
        cricket_data = json.load(file)
except FileNotFoundError:
    cricket_data = {"error": "Local data file not found!"}

# 4. Define the robust Web Search Tool for the Agent using Tavily
def get_live_search_data(query):
    """Fetches real-time text and images from Tavily Search API."""
    try:
        # Optimized query for sports context
        optimized_query = f"{query} cricket news updates"
        
        # Tavily search fetching both content and images
        response = tavily_client.search(
            query=optimized_query, 
            search_depth="basic", 
            include_images=True, 
            max_results=3
        )
        
        # Format text results
        search_text = "\n\n".join([f"Title: {res['title']}\nInfo: {res['content']}" for res in response.get('results', [])])
        if not search_text:
            search_text = "TOOL_FAILURE: No recent news found."
            
        # Get the first relevant image if available
        images = response.get('images', [])
        image_url = images[0] if images else None
        
        return search_text, image_url
    except Exception as e:
        return f"TOOL_FAILURE: Tavily Search failed ({e})", None

# 5. Configure the Streamlit user interface
st.set_page_config(page_title="CricAura", page_icon="🏏", layout="centered")

custom_css = """
<style>
    .stApp {
        background-image: linear-gradient(rgba(11, 19, 43, 0.88), rgba(11, 19, 43, 0.95)), url('https://images.unsplash.com/photo-1531415074968-036ba1b575da?q=80&w=2070&auto=format&fit=crop');
        background-size: cover;
        background-attachment: fixed;
        color: #ffffff;
    }
    div[data-testid="stChatInput"] textarea, .stChatInputContainer textarea, .stChatInput textarea {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 600 !important;
    }
    .stChatMessage, .stChatMessage p, .stChatMessage li, .stChatMessage span, .stChatMessage div, .stChatMessage ul {
        color: #ffffff !important;
    }
    .stChatMessage {
        background-color: rgba(28, 37, 65, 0.85);
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #3a506b;
        backdrop-filter: blur(5px);
    }
    h1 {
        color: #4facfe !important;
        font-family: 'Trebuchet MS', sans-serif;
        text-align: center;
    }
    img {
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("🏏 CricAura")
st.markdown("<p style='text-align: center; color: #e0e0e0;'>Your Autonomous Sports Assistant</p>", unsafe_allow_html=True)
st.write("") 

# 6. Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if "image" in message and message["image"]:
            st.image(message["image"], width=300)
        st.markdown(message["content"])

# 7. Capture user input
final_query = None

# Move the large audio box to the sidebar to keep the center clean
with st.sidebar:
    st.markdown("### 🎤 Voice Search")
    voice_input = st.audio_input("Tap to speak")

# Keep the text input at the bottom
text_input = st.chat_input("Type your question here...")

if text_input:
    final_query = text_input
elif voice_input:
    with st.spinner("Processing voice input..."):
        try:
            transcription = client.audio.transcriptions.create(
                file=("audio.wav", voice_input.read()),
                model="whisper-large-v3"
            )
            final_query = transcription.text
        except Exception as e:
            st.error(f"Voice transcription failed: {e}")

# 8. Process the final extracted query
if final_query:
    st.chat_message("user").markdown(final_query)
    st.session_state.chat_history.append({"role": "user", "content": final_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data and fetching media..."):
            try:
                # Fetch text and image from Tavily
                live_web_data, fetched_image_url = get_live_search_data(final_query)
                
                st.info(f"Agent Web Tool Status: {live_web_data[:100]}...")
                
                agent_system_instruction = f"""
                You are CricAura, an autonomous sports expert. 
                
                LIVE WEB DATA: 
                {live_web_data}
                
                LOCAL DATA: 
                {json.dumps(cricket_data)}
                
                CRITICAL RULES:
                1. Analyze the LIVE WEB DATA carefully. If it starts with "TOOL_FAILURE", you MUST reply exactly with: "I apologize, but my live search tool is currently unable to fetch real-time data from the web for this query." Do not attempt to guess or hallucinate the answer.
                2. NEVER use the words "internal dashboard", "database", "system", "memory", or "World Cricket Dashboard". 
                3. NEVER invent match schedules if the live data is missing.
                4. Answer directly using ONLY the facts provided in the data above.
                5. DO NOT output text like "(Displaying photo)" or mention images.
                """
                
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": agent_system_instruction},
                        {"role": "user", "content": final_query}
                    ],
                    model="qwen/qwen3.8-27b", 
                )
                answer = chat_completion.choices[0].message.content
                
                if fetched_image_url:
                    st.image(fetched_image_url, width=300)
                
                st.markdown(answer)
                
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer, 
                    "image": fetched_image_url
                })
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")