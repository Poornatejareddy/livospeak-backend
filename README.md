# LivoSpeak AI — Speech Coaching Backend API

This is the FastAPI-powered speech analysis and coaching backend for LivoSpeak AI. It processes browser-recorded audio, performs transcription and linguistic analytics via Groq's high-speed LPU APIs, and manages student history with MongoDB.

---

## 🚀 Features

*   **Audio Validation & Remuxing**: Normalizes incoming media streams (like browser WebM/Opus) into standard 16kHz WAV streams via FFmpeg/FFprobe.
*   **Speech-to-Text (STT)**: Uses **Groq Whisper Large V3** to retrieve high-resolution word-level timestamp alignments and token confidences.
*   **Linguistic Pronunciation Analysis**: Uses **Groq Llama 3.3 70B** to detect phonetic errors, syllable stress gaps, WPM speed ratings, and generate personalized tongue twisters and a 5-minute practice plan.
*   **Strict Scoring Metrics**: Computes customized scores for Pronunciation, Clarity, and Fluency based on real LLM-detected phonetic mistakes and pause duration evaluations.
*   **Compliance-First Privacy**: Ensures no audio data or personal voice biometric markers are stored permanently, adhering to India's DPDP Act 2023.

---

## 🛠️ Prerequisites

Before running the backend, ensure you have the following installed on your host system:
1.  **Python 3.10+**
2.  **FFmpeg / FFprobe**: The binaries must be present in the system path (`$PATH`) for audio duration extraction and remuxing to function.
3.  **MongoDB**: A running local MongoDB instance or a MongoDB Atlas connection string.

---

## 📦 Installation & Setup

1.  **Navigate to the backend directory**:
    ```bash
    cd backend
    ```

2.  **Create and activate a Python virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create environment configuration**:
    Create a file named `.env` in the root of the `backend/` directory:
    ```ini
    GROQ_API_KEY=your_groq_api_key_here
    PORT=8000
    HOST=127.0.0.1
    MONGODB_URI=mongodb://127.0.0.1:27017/livospeak
    ```

---

## 🚦 Running the Application

To start the FastAPI development server with hot-reload enabled:

```bash
uvicorn app.main:app --reload --port 8000
```

The documentation UI will be accessible at:
- Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🔗 Core API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/analyze` | Receives multipart form data audio files, validates duration, transcribes, analyzes pronunciation, and immediately wipes audio. |
| `GET` | `/api/history` | Fetches a summary of past speaking runs sorted chronologically. |
| `GET` | `/api/history/{id}` | Fetches full, detailed analytics and practice plans for a specific session. |
| `DELETE` | `/api/history/{id}` | Permanently deletes a history record from the database (DPDP Compliance). |

---

## 🔒 DPDP Privacy Safeguards
- **In-Memory / Instant Cleanup**: Audio files are deleted inside a `finally` block instantly upon completing the API request (or upon raising a validation error).
- **No Voice Biometrics**: The system never maps vocal prints or biometric markers. Only the textual transcription metrics are saved to MongoDB.
