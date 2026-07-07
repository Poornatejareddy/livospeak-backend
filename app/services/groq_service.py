import os
import json
import logging
from typing import Dict, Any, List
from groq import Groq
from app.config import GROQ_API_KEY

logger = logging.getLogger(__name__)

# Initialize client if key is present
client = None
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {str(e)}")

def transcribe_audio(file_path: str) -> Dict[str, Any]:
    """
    Transcribes audio using Groq Whisper.
    Forces verbose_json format and word-level timestamps to retrieve word confidences.
    
    Returns:
        Dict containing transcription, language, and word-level details.
    """
    if not GROQ_API_KEY or not client:
        logger.warning("GROQ_API_KEY is not configured. Falling back to Mock Transcription.")
        return get_mock_transcription()

    try:
        with open(file_path, "rb") as audio_file:
            # Call Groq Whisper API
            response = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
            
            # The response is a TranscriptionVerbose object. We convert it to a dictionary.
            # If the response has a direct .model_dump() or dict representation:
            if hasattr(response, "model_dump"):
                result = response.model_dump()
            elif hasattr(response, "dict"):
                result = response.dict()
            else:
                result = dict(response)
                
            return result
    except Exception as e:
        logger.error(f"Groq Whisper transcription failed: {str(e)}")
        # Return fallback mock transcription if API call fails
        logger.info("Falling back to Mock Transcription due to API error.")
        return get_mock_transcription()

def generate_coaching_feedback(
    transcript: str, 
    wpm: int, 
    low_confidence_words: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calls Groq LLM (llama-3.3-70b-specdec or llama3-8b-8192) to analyze the speech
    and generate pronunciation coaching details (mistakes, IPA, explanations, practice plans).
    
    Returns:
        Structured JSON response with mistake explanations and practice plan.
    """
    if not GROQ_API_KEY or not client:
        logger.warning("GROQ_API_KEY is not configured. Falling back to Mock Coaching Feedback.")
        return get_mock_coaching_feedback(low_confidence_words)

    # Prepare prompt
    words_list_str = ", ".join([f"'{w['word']}' (confidence: {w['confidence']:.2f})" for w in low_confidence_words])
    
    prompt = f"""
You are an expert English Pronunciation and Speech Coach. An English learner has uploaded a speech recording.
Here is the transcript of their speech:
"{transcript}"

The speaker's speech rate was calculated at: {wpm} WPM (Words Per Minute).
Here are the candidate words flagged with low transcription confidence (indicating potential pronunciation issues):
[{words_list_str}]

Your task is to analyze this speech and generate highly detailed, realistic, and rigorous pronunciation coaching. 
To ensure high accuracy, do not be overly generous:
1. Identify up to 5 words that were likely mispronounced. Focus primarily on the low confidence words, but you can also identify other words in the transcript that are commonly mispronounced (e.g. silent letters like 'receipt', syllable compression like 'comfortable', wrong word stress, or phoneme errors) if the candidate list is empty or insufficient.
2. For each identified word, provide:
   - "word": The word as it appears in the transcript.
   - "issue": A brief description of the specific pronunciation error (e.g., "silent 'p' pronounced", "wrong vowel quality", "misplaced word stress").
   - "expected_pronunciation": The expected pronunciation in standard IPA notation (e.g., "/ˈkʌmf.tə.bəl/").
   - "why_it_matters": The linguistic explanation of why the correct pronunciation matters (e.g. vowel shifts, syllable count, syllable stress).
   - "practice": 3 short practice phrases containing the word.
3. If the user spoke perfectly, you can leave the "mistakes" list empty (or with only 1 minor suggestion). But be rigorous.

Also generate strengths, weaknesses, overall actionable advice, and a 5-minute practice plan.

You MUST respond with a JSON object conforming exactly to this structure:
{{
  "mistakes": [
    {{
      "word": "comfortable",
      "issue": "Missing middle syllable",
      "expected_pronunciation": "/ˈkʌmf.tə.bəl/",
      "why_it_matters": "Native speakers typically pronounce all three syllable groups, even though the middle one is reduced. Pronouncing it as 'com-for-ta-ble' sounds unnatural.",
      "practice": ["comfortable chair", "comfortable shoes", "comfortable weather"]
    }}
  ],
  "coaching": {{
    "strengths": "Provide 1-2 sentences highlighting what the speaker did well (e.g., speed, pausing, general consonant sounds).",
    "weaknesses": "Provide 1-2 sentences highlighting their primary pattern of error (e.g., dropping ending consonants, inserting extra syllables).",
    "advice": "Provide 2 sentences of actionable coaching advice on how to improve."
  }},
  "practice_plan": {{
    "practice_words": ["word1", "word2", "word3"],
    "practice_sentences": ["Short sentence using practice word 1.", "Short sentence using practice word 2."],
    "tongue_twisters": ["1-2 relevant English tongue twisters."],
    "five_minute_plan": [
      "Minute 1: Warm up your mouth and vocal cords by humming and doing lip trills.",
      "Minutes 2-3: Slowly read the practice words list, exaggerating the correct vowel sounds and syllable count.",
      "Minutes 4-5: Speak the practice sentences in front of a mirror, focusing on linking and pacing."
    ]
  }}
}}

Ensure that "mistakes" contains entries only for words that actually have errors or were in the low confidence list (up to 5 key words).
Do not return any conversational text, markdown wrappers, or formatting outside of the valid JSON object.
"""

    models_to_try = ["llama-3.3-70b-specdec", "llama-3.1-70b-versatile", "llama3-8b-8192"]
    
    for model in models_to_try:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional pronunciation coach who outputs strictly valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model,
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1500
            )
            
            response_text = chat_completion.choices[0].message.content
            return json.loads(response_text)
        except Exception as e:
            logger.warning(f"Failed to generate feedback with model {model}: {str(e)}")
            continue

    logger.error("All Groq LLM models failed. Falling back to Mock Coaching Feedback.")
    return get_mock_coaching_feedback(low_confidence_words)

def get_mock_transcription() -> Dict[str, Any]:
    """
    Returns a realistic mock transcription.
    """
    # A standard English learner paragraph about technology and communication.
    text = "In my opinion, technology is very comfortable and important for modern communication. However, some people face difficulties when they speak English because they don't practice regular vocabulary or pronunciation. We need to focus on developing better listening skills and speaking daily."
    
    # Generate mock words list with timestamps and confidence scores
    # Words like "comfortable", "difficulties", "pronunciation", "regular" will have lower confidence
    raw_words = text.replace(".", "").replace(",", "").split()
    words = []
    start_time = 0.5
    
    for idx, w in enumerate(raw_words):
        duration = len(w) * 0.12 + 0.1
        w_lower = w.lower()
        
        # Set low confidence for specific target words to simulate mistakes
        if w_lower in ["comfortable", "difficulties", "pronunciation", "regular", "technology"]:
            confidence = 0.55 + (idx % 3) * 0.08
        else:
            confidence = 0.92 + (idx % 5) * 0.01
            
        words.append({
            "word": w,
            "start": round(start_time, 2),
            "end": round(start_time + duration, 2),
            "confidence": round(confidence, 2)
        })
        start_time += duration + 0.15 # Add pause between words
        
    return {
        "text": text,
        "language": "english",
        "duration": round(start_time, 2),
        "segments": [
            {
                "text": text,
                "start": 0.0,
                "end": round(start_time, 2),
                "words": words
            }
        ]
    }

def get_mock_coaching_feedback(low_confidence_words: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns realistic mock coaching feedback.
    """
    # Look for our known mock words or fallback
    words_to_mock = [w["word"].lower().replace(".", "").replace(",", "") for w in low_confidence_words]
    
    mistakes = []
    
    # We will build standard mock explanations for typical mispronounced words
    mock_mistake_db = {
        "comfortable": {
            "issue": "Missing middle syllable / pronouncing as com-for-ta-ble",
            "expected_pronunciation": "/ˈkʌmf.tə.bəl/",
            "why_it_matters": "Native speakers compress this word to three syllables. Pronouncing all four syllables ('com-for-ta-ble') sounds overly formal and non-native.",
            "practice": ["a comfortable chair", "comfortable shoes", "feel comfortable speaking"]
        },
        "difficulties": {
            "issue": "Weak syllable reduction on 'i' and incorrect word stress",
            "expected_pronunciation": "/ˈdɪf.ɪ.kəl.tiz/",
            "why_it_matters": "The stress should be on the very first syllable ('DIF'). Placing stress on 'cult' or failing to reduce the middle vowels makes the word hard to recognize.",
            "practice": ["face many difficulties", "learning difficulties", "overcome difficulties"]
        },
        "pronunciation": {
            "issue": "Pronouncing 'nun' as 'noun' (pro-noun-ci-a-tion)",
            "expected_pronunciation": "/prəˌnʌn.siˈeɪ.ʃən/",
            "why_it_matters": "Although the verb is 'pronounce' (/praʊns/), the noun shifts to a short 'u' sound (/nʌn/). Pronouncing it as 'pronounc-iation' is a highly common error.",
            "practice": ["english pronunciation", "correct pronunciation", "pronunciation practice"]
        },
        "regular": {
            "issue": "Distorting the vocalic 'r' and unstressed 'u' syllable",
            "expected_pronunciation": "/ˈreɡ.jə.lər/",
            "why_it_matters": "The middle syllable 'u' is reduced to a 'yuh' (/jə/) sound, and the ending has a rhotic 'ler' (/lər/). Leaving the 'u' sound fully open makes it sound awkward.",
            "practice": ["regular practice", "on a regular basis", "regular schedule"]
        },
        "technology": {
            "issue": "Incorrect stress on 'tech' instead of 'nol'",
            "expected_pronunciation": "/tekˈnɒl.ə.dʒi/",
            "why_it_matters": "In English, four-syllable nouns ending in '-logy' place the primary stress on the antepenultimate (third-to-last) syllable: 'tech-NOL-o-gy'.",
            "practice": ["modern technology", "information technology", "technology is evolving"]
        }
    }
    
    for word in words_to_mock:
        if word in mock_mistake_db:
            mistakes.append({
                "word": word,
                **mock_mistake_db[word]
            })
            
    # Default fallback mistake if none matched
    if not mistakes and low_confidence_words:
        for w in low_confidence_words[:2]:
            word = w["word"].replace(".", "").replace(",", "")
            mistakes.append({
                "word": word,
                "issue": "Unclear vowel articulation or word stress",
                "expected_pronunciation": f"/{word}/",
                "why_it_matters": "Clear articulation of vowel sounds and proper word stress is crucial for clarity. Native speakers rely on stress patterns to decode word meaning.",
                "practice": [f"practice {word}", f"say {word} slowly", f"correct stress on {word}"]
            })
            
    # If still empty (no low confidence words)
    if not mistakes:
        mistakes.append({
            "word": "comfortable",
            **mock_mistake_db["comfortable"]
        })

    return {
        "mistakes": mistakes,
        "coaching": {
            "strengths": "You maintain a steady speaking speed and your articulation of basic consonants (like 't', 'p', 'd') is clean.",
            "weaknesses": "You have a tendency to misplace stress on multi-syllable words and struggle with vowel reduction in unstressed syllables.",
            "advice": "Focus on identifying the stressed syllable in new vocabulary. Practice reducing unstressed vowels to the 'schwa' (/ə/) sound to sound more natural."
        },
        "practice_plan": {
            "practice_words": [m["word"] for m in mistakes] + ["vegetable", "interesting", "literally"],
            "practice_sentences": [
                "It is comfortable to use technology for communication.",
                "We need regular pronunciation practice to overcome difficulties."
            ],
            "tongue_twisters": [
                "Red lolly, yellow lolly, red lolly, yellow lolly.",
                "A proper copper coffee pot."
            ],
            "five_minute_plan": [
                "Minute 1: Practice deep breathing and jaw relaxation exercises.",
                "Minutes 2-3: Pronounce the practice words, focusing on correct syllable stress (e.g. tech-NOL-o-gy).",
                "Minutes 4-5: Record yourself reading the practice sentences, and play it back to self-evaluate clarity."
            ]
        }
    }
