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
        
        # If duration is "N/A" (common for browser-native WebM/OGG streams), remux to WAV to compute duration
        if not duration_str or duration_str.upper() == "N/A":
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Duration reported as N/A. Remuxing to standard WAV to calculate length.")
            
            temp_wav = file_path + ".wav"
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", file_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                result_wav = subprocess.run([
                    "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", temp_wav
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                duration_str = result_wav.stdout.strip()
                
                # Replace original file with standardized WAV
                os.replace(temp_wav, file_path)
            except Exception as remux_err:
                logger.error(f"Failed to remux file for duration parsing: {str(remux_err)}")
                if os.path.exists(temp_wav):
                    try:
                        os.remove(temp_wav)
                    except:
                        pass
        
        if not duration_str or duration_str.upper() == "N/A":
            return False, "Could not extract duration. The file might be corrupted or not a valid audio format.", 0.0
        
        duration = float(duration_str)
        
        # English learners recording length condition: between 30 and 45 seconds.
        if duration < 30.0 or duration > 45.0:
            return False, f"Audio duration must be between 30 and 45 seconds. Your recording is {duration:.1f} seconds long.", duration
            
        return True, "", duration
    except subprocess.CalledProcessError as e:
        return False, f"Failed to parse audio file. Ensure it is a valid MP3, WAV, WebM or M4A file. Error: {e.stderr.strip()}", 0.0
    except Exception as e:
        return False, f"An unexpected error occurred during audio validation: {str(e)}", 0.0
