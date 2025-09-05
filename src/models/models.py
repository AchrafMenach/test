from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime

class StudentProfile(BaseModel):
    student_id: str
    name: Optional[str] = None
    level: int = 1
    current_objective: Optional[str] = None
    learning_history: List[Dict] = Field(default_factory=list)
    objectives_completed: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_session: Optional[str] = None

class Exercise(BaseModel):
    exercise: str = Field(description="Une question unique et précise adaptée à l\'objectif")
    solution: str = Field(description="Solution mathématique détaillée et rigoureuse")
    hints: List[str] = Field(
        description="Indice principal pour guider l\'élève",
        default_factory=list
    )
    difficulty: str
    concept: str

class EvaluationResult(BaseModel):
    is_correct: bool = Field(..., description="Indique si la réponse est correcte")
    error_type: Optional[str] = Field(None, description="Type d\'erreur identifié") 
    feedback: str = Field(..., description="Feedback pédagogique détaillé s l\'erreur")
    detailed_explanation: str = Field(..., description="Explication mathématique complète")
    step_by_step_correction: str = Field(..., description="Correction étape par étape")
    recommendations: List[str] = Field(..., description="Recommandations personnalisées")

class CoachPersonal(BaseModel):
    motivation: str = Field(..., description="message motivant")
    strategy: str = Field(..., description="stratégie concrète") 
    tip: str = Field(..., description="astuce pratique")
    encouragement: List[str] = Field(..., description="phrase positive")

from pydantic import BaseModel
from typing import Optional, List, Dict

class PersonalizedCoachMessage(BaseModel):
    motivation: str
    strategy: str
    tip: str
    encouragement: List[str]
    next_steps: List[str]   # Nouveau : étapes recommandées
