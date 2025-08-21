from crewai import Task
from crewai import Agent
from src.models.models import EvaluationResult, Exercise
from typing import Union
from pathlib import Path

class EvaluationTask:
    def __init__(self, agent: Agent):
        self.agent = agent

    def create_task(self, exercise: Exercise, answer: Union[str, Path], extracted_text: str) -> Task:
        prompt = f"""
        Évalue la réponse de l\"étudiant à l\"exercice suivant:
        Exercice: {exercise.exercise}
        Solution attendue: {exercise.solution}
        Réponse de l\"étudiant (texte extrait): {extracted_text}

        Fournis une évaluation précise et pédagogique, identifie clairement les erreurs et propose des explications détaillées.
        """
        return Task(
            description=prompt,
            agent=self.agent,
            expected_output="Un objet EvaluationResult complet avec is_correct, error_type, feedback, detailed_explanation, step_by_step_correction et recommendations",
            output_pydantic=EvaluationResult
        )


