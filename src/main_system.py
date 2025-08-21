import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from rich.console import Console
console = Console()
from crewai import Crew, Process
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from crewai import Task

from src.models.models import Exercise, EvaluationResult, StudentProfile, CoachPersonal
from src.agents.exercise_creator_agent import ExerciseCreatorAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.personal_coach_agent import PersonalCoachAgent
from src.tasks.exercise_creation_task import ExerciseCreationTask
from src.tasks.evaluation_task import EvaluationTask
from src.tools.file_processor import FileProcessor
from src.config.learning_objectives import LearningObjectives
from src.student_manager import StudentManager

load_dotenv()

class MathTutoringSystem:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.llm = self._initialize_llm()
        
        self.file_processor = FileProcessor()
        self.student_manager = StudentManager()
        self.learning_objectives = LearningObjectives()
        self.current_student: Optional[StudentProfile] = None

        self.exercise_creator_agent = ExerciseCreatorAgent(self.llm).create_agent() if self.llm else None
        self.evaluator_agent = EvaluatorAgent(self.llm).create_agent() if self.llm else None
        self.personal_coach_agent = PersonalCoachAgent(self.llm).create_agent() if self.llm else None

        self.exercise_creation_task = ExerciseCreationTask(self.exercise_creator_agent) if self.exercise_creator_agent else None
        self.evaluation_task = EvaluationTask(self.evaluator_agent) if self.evaluator_agent else None

    def _initialize_llm(self):
        try:
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=f"groq/llama-3.3-70b-versatile",
                temperature=0.7
            )
        except Exception as e:
            console.print(f"Mode hors ligne activé: {str(e)}")
            return None

    def set_current_student(self, student_id: str):
        self.current_student = self.student_manager.load_student(student_id)
        if self.current_student and self.student_manager.long_term_memory:
            self._load_initial_memories()

    def _load_initial_memories(self):
        if not self.current_student or not self.student_manager.long_term_memory:
            return
                    # Ajouter les objectifs complétés comme mémoires
        for obj in self.current_student.objectives_completed:
            self.student_manager.long_term_memory.add_memory(
                content=f"Objectif complété: {obj}",
                metadata={"type": "achievement", "objective": obj}
            )
                # Ajouter l'historique d'apprentissage
        for item in self.current_student.learning_history:
            self.student_manager.long_term_memory.add_memory(
                content=f"Exercice: {item['exercise']} - Réponse: {item['answer']}",
                metadata={
                    "type": "exercise",
                    "correct": str(item["evaluation"]),
                    "timestamp": item["timestamp"]
                }
            )

    def get_current_objective_info(self):
        if not self.current_student:
            return None
        
        objective = self.learning_objectives.objectives.get(self.current_student.current_objective or "", {})
        if not objective:
            return None
        
        level_info = objective["niveaux"].get(str(self.current_student.level), {})
        return {
            "description": objective.get("description", ""),
            "level_name": level_info.get("name", ""),
            "total_levels": len(objective["niveaux"]),
            "objectives": level_info.get("objectives", [])
        }

    def get_student_progress(self):
        if not self.current_student:
            return None
        
        return {
            "level": self.current_student.level,
            "completed": len(self.current_student.objectives_completed),
            "history": pd.DataFrame(self.current_student.learning_history)
        }

    def generate_exercise(self) -> Optional[Exercise]:
        if not self.current_student or not self.current_student.current_objective:
            console.print("Aucun étudiant ou objectif défini")
            return None

        objective_data = self.learning_objectives.objectives.get(self.current_student.current_objective)
        if not objective_data:
            console.print(f"Objectif non trouvé: {self.current_student.current_objective}")
            return None

        level_info = objective_data["niveaux"].get(str(self.current_student.level))
        if not level_info:
            console.print(f"Niveau non trouvé: {self.current_student.level}")
            return None

        default_exercise = Exercise(
            exercise=f"Résoudre: {level_info['example_functions'][0]}",
            solution=f"Solution: {level_info['objectives'][0]}",
            hints=["Appliquez les méthodes appropriées"],
            difficulty=level_info["name"],
            concept=self.current_student.current_objective
        )

        if not self.exercise_creator_agent or not self.exercise_creation_task:
            return default_exercise

        try:
            task = self.exercise_creation_task.create_task(
                objective_description=objective_data["description"],
                level_name=level_info["name"],
                objective_type=self.current_student.current_objective,
                example_function=level_info["example_functions"][0]
            )

            crew = Crew(
                agents=[self.exercise_creator_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )

            result = crew.kickoff()
            
            # CORRECTION: Extraire l'objet Exercise du CrewOutput
            if hasattr(result, 'pydantic') and result.pydantic:
                return result.pydantic
            elif hasattr(result, 'raw') and result.raw:
                # Si le résultat est déjà un objet Exercise
                if isinstance(result.raw, Exercise):
                    return result.raw
                # Sinon, retourner l'exercice par défaut
                return default_exercise
            else:
                return default_exercise

        except Exception as e:
            console.print(f"Erreur génération exercice: {str(e)}")
            return default_exercise

    def generate_similar_exercise(self, original_exercise: Exercise) -> Optional[Exercise]:
        if not self.current_student:
            console.print("Aucun étudiant défini pour générer un exercice similaire.")
            return None

        if not self.exercise_creator_agent or not self.exercise_creation_task:
            console.print("Agents ou tâches de création d'exercices non initialisés.")
            return None

        try:
            task = self.exercise_creation_task.create_similar_exercise_task(
                original_exercise=original_exercise,
                student_level=self.current_student.level
            )
            crew = Crew(
                agents=[self.exercise_creator_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            result = crew.kickoff()
            
            # CORRECTION: Extraire l'objet Exercise du CrewOutput
            if hasattr(result, 'pydantic') and result.pydantic:
                return result.pydantic
            elif hasattr(result, 'raw') and result.raw:
                if isinstance(result.raw, Exercise):
                    return result.raw
                return None
            else:
                return None
        except Exception as e:
            console.print(f"Erreur lors de la génération d'un exercice similaire: {str(e)}")
            return None

    def evaluate_response(self, exercise: Exercise, answer: Union[str, Path]) -> EvaluationResult:
        extracted_text = ""
        if isinstance(answer, (Path, str)) and Path(answer).exists():
            extracted_text = self.file_processor.extract_text_from_file(str(answer))
            if not extracted_text:
                console.print("Aucun texte extrait du fichier")
                return self._create_fallback_evaluation(exercise)
        else:
            extracted_text = str(answer)

        if not self.evaluator_agent or not self.evaluation_task:
            return self._create_fallback_evaluation(exercise)

        try:
            task = self.evaluation_task.create_task(exercise, answer, extracted_text)
            crew = Crew(
                agents=[self.evaluator_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            result = crew.kickoff()
            
            # CORRECTION: Extraire l'objet EvaluationResult du CrewOutput
            if hasattr(result, 'pydantic') and result.pydantic:
                return result.pydantic
            elif hasattr(result, 'raw') and result.raw:
                if isinstance(result.raw, EvaluationResult):
                    return result.raw
                return self._create_fallback_evaluation(exercise)
            else:
                return self._create_fallback_evaluation(exercise)
        except Exception as e:
            console.print(f"Erreur évaluation réponse: {str(e)}")
            return self._create_fallback_evaluation(exercise)

    def _create_fallback_evaluation(self, exercise: Exercise) -> EvaluationResult:
        return EvaluationResult(
            is_correct=False,
            error_type="Générique",
            feedback="Impossible d'évaluer la réponse. Veuillez réessayer ou fournir une réponse plus claire.",
            detailed_explanation=f"La solution attendue était: {exercise.solution}",
            step_by_step_correction="Aucune correction détaillée disponible en raison de l'erreur.",
            recommendations=["Vérifiez votre saisie", "Contactez le support si le problème persiste"]
        )

    def _create_fallback_coaching(self) -> CoachPersonal:
        return CoachPersonal(
            motivation="Accroche-toi, chaque effort compte !",
            strategy="Essaie de décomposer le problème en étapes plus petites.",
            tip="N'hésite pas à demander de l'aide si tu es bloqué.",
            encouragement=["Tu es capable de grandes choses !", "La persévérance est la clé du succès."]
        )

    def get_personal_coach_message(self) -> Optional[CoachPersonal]:
        if not self.personal_coach_agent or not self.llm:
            return self._create_fallback_coaching()
        
        try:
            # Créer une tâche pour le coach personnel
            task = Task(
                description="Génère un message de motivation, une stratégie, une astuce et des encouragements pour un étudiant en mathématiques. Le message doit être positif et constructif.",
                agent=self.personal_coach_agent,
                expected_output="Un objet CoachPersonal complet avec motivation, strategy, tip et encouragement",
                output_pydantic=CoachPersonal
            )
            crew = Crew(
                agents=[self.personal_coach_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            result = crew.kickoff()
            
            # CORRECTION: Extraire l'objet CoachPersonal du CrewOutput
            if hasattr(result, 'pydantic') and result.pydantic:
                return result.pydantic
            elif hasattr(result, 'raw') and result.raw:
                if isinstance(result.raw, CoachPersonal):
                    return result.raw
                return self._create_fallback_coaching()
            else:
                return self._create_fallback_coaching()
        except Exception as e:
            console.print(f"Erreur génération message coach: {str(e)}")
            return self._create_fallback_coaching()


if __name__ == "__main__":
    system = MathTutoringSystem()
    
    # Example Usage
    # Create a student
    student = system.student_manager.create_student(name="Alice")
    system.set_current_student(student.student_id)
    console.print(f"Étudiant créé: {system.current_student.name} (ID: {system.current_student.student_id})")

    # Set a current objective for the student (assuming 'Fonctions' is a valid objective)
    if system.learning_objectives.objectives_order:
        system.current_student.current_objective = system.learning_objectives.objectives_order[0]
        system.student_manager.save_student(system.current_student)
        console.print(f"Objectif actuel défini: {system.current_student.current_objective}")

    # Generate an exercise
    exercise = system.generate_exercise()
    if exercise:
        console.print(f"Exercice généré: {exercise.exercise}")
        console.print(f"Solution: {exercise.solution}")
        console.print(f"Indices: {exercise.hints}")

        # Generate a similar exercise
        similar_exercise = system.generate_similar_exercise(exercise)
        if similar_exercise:
            console.print(f"Exercice similaire généré: {similar_exercise.exercise}")

        # Simulate student answer
        student_answer = "La réponse est 42."
        evaluation = system.evaluate_response(exercise, student_answer)
        console.print(f"Évaluation: {evaluation.feedback}")
        console.print(f"Correction détaillée: {evaluation.detailed_explanation}")

    # Get a personal coach message
    coach_message = system.get_personal_coach_message()
    if coach_message:
        console.print(f"Message du coach: {coach_message.motivation}")
        console.print(f"Stratégie: {coach_message.strategy}")
        console.print(f"Astuce: {coach_message.tip}")
        console.print(f"Encouragements: {', '.join(coach_message.encouragement)}")

    # Get student progress
    progress = system.get_student_progress()
    if progress:
        console.print(f"Progression de l'étudiant: Niveau {progress['level']}")

    # Example of loading an existing student
    # loaded_student = system.student_manager.load_student(student.student_id)
    # if loaded_student:
    #     console.print(f"Étudiant chargé: {loaded_student.name}")