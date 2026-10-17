import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Gemini ke available models yeh hain:\n")
try:
    # Google GenAI SDK se models list karna
    for model in client.models.list():
        print(model.name)
except Exception as e:
    print(f"Error aaya bhai: {e}")