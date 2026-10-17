import streamlit as st
import json
import os
from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS

# 1. Load environment variables and retrieve Groq API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# 2. Initialize the Groq AI client
client = Groq(api_key=api_key)

# 3. Load local mock cricket data from the JSON file
try:
    with open("mock_cricket_data.json", "r") as file:
        cricket_data = json.load(file)
except FileNotFoundError:
    cricket_data = {"error": "Local data file not found!"}

# 4. Define the Web Search Tools for the Agent (Text and Images)
def get_live_search_results(query):
    """Fetches real-time text data from DuckDuckGo using multiple fallback backends."""
    optimized_query = f"{query} cricket news"
    
    # Try different backends one by one to bypass rate limits
    for backend in ["api", "html", "lite"]:
        try:
            results = DDGS().text(optimized_query, max_results=3, backend=backend)
            if results:
                search_text = "\n\n".join([f"Title: {res['title']}\nInfo: {res['body']}" for res in results])
                return search_text
        except Exception:
            continue  # If this backend fails, try the next one in the loop
            
    return "TOOL_FAILURE: Search API is temporarily rate-limited."

def get_image_url(query):
    """Fetches a relevant image URL from DuckDuckGo Images."""
    try:
        search_query = f"{query} cricket"
        results = DDGS().images(search_query, max_results=1)
        if results and len(results) > 0:
            return results[0].get("image")
        return None
    except Exception:
        return None

# 5. Configure the Streamlit user interface with a custom modern layout
st.set_page_config(page_title="CricAura", page_icon="🏏", layout="centered")

# Inject Custom CSS for dark background image, fixed contrast, and chat colors
custom_css = """
<style>
    /* Dark background with Cricket image overlay */
    .stApp {
        background-image: linear-gradient(rgba(11, 19, 43, 0.88), rgba(11, 19, 43, 0.95)), url('https://images.unsplash.com/photo-1531415074968-036ba1b575da?q=80&w=2070&auto=format&fit=crop');
        background-size: cover;
        background-attachment: fixed;
        color: #ffffff;
    }
    
    /* Fix Chat Input Text Color (Force Black text for visibility inside the light box) */
    div[data-testid="stChatInput"] textarea, .stChatInputContainer textarea, .stChatInput textarea {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Fix Markdown Lists (Stats) Color to White for high contrast */
    .stChatMessage, .stChatMessage p, .stChatMessage li, .stChatMessage span, .stChatMessage div, .stChatMessage ul {
        color: #ffffff !important;
    }
    
    /* Message Box Styling with blur effect */
    .stChatMessage {
        background-color: rgba(28, 37, 65, 0.85);
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #3a506b;
        backdrop-filter: blur(5px);
    }
    
    /* Headers */
    h1 {
        color: #4facfe !important;
        font-family: 'Trebuchet MS', sans-serif;
        text-align: center;
    }
    
    /* Image styling inside chat */
    img {
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# UI Headers
st.title("🏏 CricAura")
st.markdown("<p style='text-align: center; color: #e0e0e0;'>Your Autonomous Sports Assistant</p>", unsafe_allow_html=True)
st.write("") # Spacer

# 6. Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 7. Display previous conversation history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        # Display image if it exists in the message history
        if "image" in message and message["image"]:
            st.image(message["image"], width=300)
        st.markdown(message["content"])

# 8. Capture user input (Both Voice and Text options)
final_query = None

# Audio and Text inputs
voice_input = st.audio_input("🎤 Tap to use Voice Search")
text_input = st.chat_input("Type your question here...")

if text_input:
    final_query = text_input
elif voice_input:
    with st.spinner("Processing voice input..."):
        try:
            # Transcribe audio using Groq's Whisper API
            transcription = client.audio.transcriptions.create(
                file=("audio.wav", voice_input.read()),
                model="whisper-large-v3"
            )
            final_query = transcription.text
        except Exception as e:
            st.error(f"Voice transcription failed: {e}")

# 9. Process the final extracted query
if final_query:
    # Display the user message in the UI
    st.chat_message("user").markdown(final_query)
    st.session_state.chat_history.append({"role": "user", "content": final_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data and fetching media..."):
            try:
                # Fetch live text context and relevant image from the web
                live_web_data = get_live_search_results(final_query)
                fetched_image_url = get_image_url(final_query)
                
                # Visual debugger for the user to see the exact tool output
                st.info(f"Agent Web Tool Status: {live_web_data[:100]}...")
                
                # ULTRA-STRICT SYSTEM INSTRUCTION TO PREVENT HALLUCINATIONS
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
                
                # Generate the final response using the Qwen model
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": agent_system_instruction},
                        {"role": "user", "content": final_query}
                    ],
                    model="qwen/qwen3.8-27b", 
                )
                answer = chat_completion.choices[0].message.content
                
                # Render the actual image file (if DDGS found one), then the text answer
                if fetched_image_url:
                    st.image(fetched_image_url, width=300)
                
                st.markdown(answer)
                
                # Save both the text answer and the image URL to session state
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer, 
                    "image": fetched_image_url
                })
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")