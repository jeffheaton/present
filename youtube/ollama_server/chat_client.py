from openai import OpenAI
import os

SERVER_URL = "http://10.0.4.205:8080/"  # Replace with your server URL
API_KEY = os.getenv("OLLAMA_KEY")
MODEL = "gpt-oss:20b"  # Replace with your desired model

# Point to your Open WebUI instance
client = OpenAI(
    base_url=f"{SERVER_URL}api",
    api_key=API_KEY,
)

# Ask a simple question
response = client.chat.completions.create(
    model=MODEL,  # or any model shown in your WebUI
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(response.choices[0].message.content)

