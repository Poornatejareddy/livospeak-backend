import logging
from typing import Dict, Any, List
from app.services.groq_service import transcribe_audio, generate_coaching_feedback

logger = logging.getLogger(__name__)

def analyze_speech(file_path: str, duration: float) -> Dict[str, Any]:
    """
    Orchestrates the speech analysis pipeline:
    1. Transcribes speech using Groq Whisper.
    2. Validates language (must be English).
    3. Calculates speech metrics (WPM, pauses, confidence).
    4. Calculates scores (pronunciation, fluency, clarity, overall).
    5. Detects low confidence words.
    6. Calls LLM to generate pronunciation feedback and a practice plan.
    """
    # 1. Transcribe audio
    transcription_result = transcribe_audio(file_path)
    
    # 2. Check language
    detected_lang = transcription_result.get("language", "english").lower()
    # Support "en", "english", etc.
    if detected_lang not in ["english", "en"] and not detected_lang.startswith("en"):
        raise ValueError(
            f"The speech must be in English. Detected language: '{detected_lang.capitalize()}'. "
            "Please record your speech in English."
        )

    transcript_text = transcription_result.get("text", "").strip()
    if not transcript_text:
        raise ValueError("Could not detect any speech in the audio. Please make sure you speak clearly and your microphone is working.")

    # 3. Process words and find timestamps
    segments = transcription_result.get("segments") or []
    all_words: List[Dict[str, Any]] = []
    
    # Try root-level words first (often populated when requesting word-level timestamps)
    root_words = transcription_result.get("words")
    if root_words:
        for sw in root_words:
            word_text = sw.get("word", "").strip()
            if not word_text:
                continue
            conf = sw.get("probability", sw.get("confidence", 0.95))
            all_words.append({
                "word": word_text,
                "start": sw.get("start", 0.0),
                "end": sw.get("end", 0.0),
                "confidence": conf
            })
    else:
        # Accumulate all words from all segments
        for segment in segments:
            if not segment:
                continue
            segment_words = segment.get("words") or []
            for sw in segment_words:
                word_text = sw.get("word", "").strip()
                if not word_text:
                    continue
                conf = sw.get("probability", sw.get("confidence", 0.95))
                all_words.append({
                    "word": word_text,
                    "start": sw.get("start", 0.0),
                    "end": sw.get("end", 0.0),
                    "confidence": conf
                })

    # If no word-level timestamps were returned, we simulate them based on transcript split
    if not all_words:
        words_list = transcript_text.split()
        if not words_list:
            raise ValueError("Could not parse words from the transcript.")
            
        logger.warning("Word timestamps not found. Simulating word timings.")
        time_per_word = duration / len(words_list)
        for idx, word in enumerate(words_list):
            all_words.append({
                "word": word,
                "start": round(idx * time_per_word, 2),
                "end": round((idx + 1) * time_per_word, 2),
                "confidence": 0.90  # Default confidence
            })

    total_words = len(all_words)
    
    # 4. Calculate Speech Rate (Words Per Minute)
    # duration is validated, WPM = (total_words / duration_seconds) * 60
    wpm = int((total_words / duration) * 60)
    
    # Label speed
    if wpm < 100:
        speed_label = "Slow"
    elif wpm > 155:
        speed_label = "Fast"
    else:
        speed_label = "Normal"

    # 5. Detect pauses and hesitations
    # A pause is marked if the gap between consecutive words is > 1.2 seconds
    pauses_count = 0
    long_pauses = []
    for i in range(1, len(all_words)):
        prev_end = all_words[i-1]["end"]
        curr_start = all_words[i]["start"]
        gap = curr_start - prev_end
        if gap > 1.2:
            pauses_count += 1
            long_pauses.append({
                "after_word": all_words[i-1]["word"],
                "duration": round(gap, 2),
                "start": prev_end,
                "end": curr_start
            })

    # 6. Identify low confidence words (Mistakes)
    # Words with confidence score < 0.88 are flagged as candidate pronunciation issues
    low_confidence_threshold = 0.88
    low_confidence_words = []
    for w in all_words:
        # Strip punctuation for cleaner display and comparison
        clean_word = w["word"].strip(".,!?;:\"()[]{}")
        if w["confidence"] < low_confidence_threshold and len(clean_word) > 1:
            w["is_mistake"] = True
            low_confidence_words.append(w)
        else:
            w["is_mistake"] = False

    # Sort low confidence words by confidence ascending, and pick the top 5 lowest for LLM analysis
    low_confidence_words_sorted = sorted(low_confidence_words, key=lambda x: x["confidence"])
    words_for_llm = low_confidence_words_sorted[:5]

    # 7. Call Groq LLM for mistakes details (IPA, why it matters, practice) and coaching
    # Passing the words_for_llm list (top 5 worst pronounced) to avoid token bloat and keep it highly relevant
    coaching_result = generate_coaching_feedback(
        transcript=transcript_text,
        wpm=wpm,
        low_confidence_words=words_for_llm
    )
    
    actual_mistakes = coaching_result.get("mistakes", [])
    num_mistakes = len(actual_mistakes)

    # 8. Mark words as mistakes in the main transcript if they match LLM mistakes
    mistake_words_set = {m["word"].lower().strip(".,!?;:\"()[]{}") for m in actual_mistakes}
    for w in all_words:
        clean_w = w["word"].lower().strip(".,!?;:\"()[]{}")
        if clean_w in mistake_words_set:
            w["is_mistake"] = True

    # 9. Calculate Scores (rigorous and adjusted by identified mistakes)
    # Pronunciation score: average confidence of all words
    confidences = [w["confidence"] for w in all_words]
    avg_confidence = sum(confidences) / total_words if total_words > 0 else 0.85
    
    # Scale: a confidence of 0.90 -> 90%, a confidence of 0.60 -> 60%
    # We apply a scaling function and then deduct for actual mistakes to match learner expectations
    pronunciation_score = int(45 + (avg_confidence * 53))
    pronunciation_score -= min(30, num_mistakes * 6)  # Deduct 6 points per mistake
    pronunciation_score = max(35, min(99, pronunciation_score))

    # Fluency score: based on WPM and pauses
    # Optimal WPM is 110 - 145. Deduct points for deviations.
    fluency_score = 95
    if wpm < 110:
        fluency_score -= int((110 - wpm) * 0.7)  # Deduct for slow speech
    elif wpm > 145:
        fluency_score -= int((wpm - 145) * 0.5)  # Deduct for fast speech
        
    # Deduct for excessive pauses (each pause > 1.2s deducts 4 points, max 20)
    fluency_score -= min(20, pauses_count * 4)
    fluency_score = max(30, min(99, fluency_score))

    # Clarity score: based on the percentage of words pronounced with high confidence
    low_conf_count = len(low_confidence_words)
    clarity_ratio = (total_words - low_conf_count) / total_words if total_words > 0 else 0.8
    clarity_score = int(40 + (clarity_ratio * 58))
    clarity_score -= min(25, num_mistakes * 5)  # Deduct 5 points per mistake
    clarity_score = max(35, min(99, clarity_score))

    # Confidence score: average of the confidences directly mapped to percentage
    confidence_score = int(avg_confidence * 100)
    confidence_score -= min(20, num_mistakes * 4)
    confidence_score = max(30, min(99, confidence_score))

    # Overall Score: weighted average of the core metrics
    overall_score = int(
        0.4 * pronunciation_score + 
        0.35 * clarity_score + 
        0.25 * fluency_score
    )
    overall_score = max(35, min(99, overall_score))

    # 10. Assemble final response
    return {
        "success": True,
        "transcript": transcript_text,
        "duration": round(duration, 2),
        "scores": {
            "overall": overall_score,
            "pronunciation": pronunciation_score,
            "fluency": fluency_score,
            "clarity": clarity_score,
            "confidence": confidence_score
        },
        "speech_rate": {
            "wpm": wpm,
            "label": speed_label
        },
        "words": all_words,
        "mistakes": actual_mistakes,
        "coaching": coaching_result.get("coaching", {
            "strengths": "Good voice projection and steady pacing.",
            "weaknesses": "Some unclear vowel sounds in multi-syllabic words.",
            "advice": "Practice slowing down on longer words and ensure you pronounce each syllable clearly."
        }),
        "practice_plan": coaching_result.get("practice_plan", {
            "practice_words": ["comfortable", "important", "pronunciation"],
            "practice_sentences": ["I want to make my pronunciation comfortable and natural."],
            "tongue_twisters": ["She sells seashells by the seashore."],
            "five_minute_plan": [
                "Minute 1: Practice deep breathing and relax your jaw.",
                "Minutes 2-3: Say the practice words slowly, focusing on each syllable.",
                "Minutes 4-5: Read the practice sentences aloud."
            ]
        })
    }
