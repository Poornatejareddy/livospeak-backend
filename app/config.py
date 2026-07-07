import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017/livospeak")

# Temporal folder for uploaded audio processing
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "temp_uploads")

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)
