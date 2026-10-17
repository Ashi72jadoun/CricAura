import os
from dotenv import load_dotenv
from groq import Groq

# API key load kar rahe hain
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# API se directly active models ki list maang rahe hain
print("Tere account par abhi yeh models completely supported aur active hain:\n")
try:
    models = client.models.list()
    for model in models.data:
        print(f"- {model.id}")
except Exception as e:
    print(f"Error: {e}")