from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ScoreBreakdown(BaseModel):
    overall: int = Field(..., description="Overall pronunciation score (35-99)")
    pronunciation: int = Field(..., description="Accuracy of phonetic pronunciation (35-99)")
    fluency: int = Field(..., description="Flow, pausing, and speech speed score (35-99)")
    clarity: int = Field(..., description="Articulatory precision and intelligibility score (35-99)")
    confidence: int = Field(..., description="Whisper transcription confidence score (30-99)")

class SpeechRate(BaseModel):
    wpm: int = Field(..., description="Words Per Minute")
    label: str = Field(..., description="Speech speed label ('Slow', 'Normal', 'Fast')")

class WordDetail(BaseModel):
    word: str = Field(..., description="The word text")
    start: float = Field(..., description="Start time of the word in seconds")
    end: float = Field(..., description="End time of the word in seconds")
    confidence: float = Field(..., description="Confidence score for this word (0.0-1.0)")
    is_mistake: bool = Field(..., description="True if flagged as potential mispronunciation")

class MistakeDetail(BaseModel):
    word: str = Field(..., description="The word name")
    issue: str = Field(..., description="Concise description of the pronunciation mistake")
    expected_pronunciation: str = Field(..., description="Expected IPA notation (e.g. /ˈkʌmf.tə.bəl/)")
    why_it_matters: str = Field(..., description="Phonetic explanation of why it matters / how to correct it")
    practice: List[str] = Field(..., description="Phrases to practice this word")

class CoachingFeedback(BaseModel):
    strengths: str = Field(..., description="What the speaker did well")
    weaknesses: str = Field(..., description="Common error patterns identified")
    advice: str = Field(..., description="Actionable coaching tips")

class PracticePlan(BaseModel):
    practice_words: List[str] = Field(..., description="List of vocabulary words to practice")
    practice_sentences: List[str] = Field(..., description="Practice sentences using the target words")
    tongue_twisters: List[str] = Field(..., description="1-2 recommended tongue twisters")
    five_minute_plan: List[str] = Field(..., description="Step-by-step daily practice routine")

class AnalysisResponse(BaseModel):
    success: bool = Field(True, description="Indicates if processing was successful")
    transcript: str = Field(..., description="Full speech transcription text")
    duration: float = Field(..., description="Calculated duration of the audio in seconds")
    scores: ScoreBreakdown = Field(..., description="Pronunciation score category breakdown")
    speech_rate: SpeechRate = Field(..., description="Speech rate metrics")
    words: List[WordDetail] = Field(..., description="Timeline breakdown of all words in the speech")
    mistakes: List[MistakeDetail] = Field(..., description="List of detected pronunciation mistakes with explanations")
    coaching: CoachingFeedback = Field(..., description="Personalized AI coach feedback")
    practice_plan: PracticePlan = Field(..., description="Structured pronunciation practice plan")
