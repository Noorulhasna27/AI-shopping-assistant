import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

print("Flash models available to your key:")
for m in client.models.list():
    name = m.name.replace("models/", "")
    if "gemini" in name and "flash" in name:
        print(" ", name)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
r = client.models.generate_content(model=MODEL, contents="Say hello in 5 words.")
print("\n", MODEL, "->", r.text)
