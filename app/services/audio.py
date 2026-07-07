import os
import subprocess
from typing import Tuple

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".ogg"}
SUPPORTED_MIME_TYPES = {
    "audio/mpeg", 
    "audio/mp3",
    "audio/wav", 
    "audio/x-wav", 
    "audio/mp4", 
    "audio/m4a", 
    "audio/x-m4a",
    "audio/webm",
    "audio/ogg"
}

def validate_audio_file(file_path: str, filename: str, mime_type: str) -> Tuple[bool, str, float]:
    """
    Validates the uploaded audio file:
    1. Check if the file extension is supported (MP3, WAV, M4A).
    2. Check file size (max 15MB).
    3. Run ffprobe to check if it's a valid audio file and verify the duration is between 30 and 45 seconds.
    
    Returns:
        (is_valid: bool, error_message: str, duration: float)
    """
    # 1. Extension check
    ext = os.path.splitext(filename.lower())[1]
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Only MP3, WAV, and M4A are supported.", 0.0

    # 2. File size check (e.g., max 15MB)
    try:
        size_bytes = os.path.getsize(file_path)
        max_size = 15 * 1024 * 1024  # 15MB
        if size_bytes > max_size:
            return False, f"File size ({size_bytes / (1024*1024):.2f}MB) exceeds the 15MB limit.", 0.0
    except OSError as e:
        return False, f"Failed to access audio file for size check: {str(e)}", 0.0

    # 3. Duration and format validation via ffprobe
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        duration_str = result.stdout.strip()
        
        if not duration_str:
            return False, "Could not extract duration. The file might be corrupted or not a valid audio format.", 0.0
        
        duration = float(duration_str)
        
        # English learners recording length condition: between 30 and 45 seconds.
        # We allow a tiny 0.5-second buffer (e.g., 29.5 to 45.5 seconds) to be user-friendly,
        # but let's stick to the 30-45 second range.
        if duration < 30.0 or duration > 45.0:
            return False, f"Audio duration must be between 30 and 45 seconds. Your recording is {duration:.1f} seconds long.", duration
            
        return True, "", duration
    except subprocess.CalledProcessError as e:
        return False, f"Failed to parse audio file. Ensure it is a valid MP3, WAV, or M4A file. Error: {e.stderr.strip()}", 0.0
    except Exception as e:
        return False, f"An unexpected error occurred during audio validation: {str(e)}", 0.0
