from crewai import Task
from crewai import Agent
from src.models.models import Exercise

class ExerciseCreationTask:
    def __init__(self, agent: Agent):
        self.agent = agent

    def create_task(self, objective_description: str, level_name: str, objective_type: str, example_function: str) -> Task:
        prompt = f"""
        Tu es un professeur de mathématiques expert. Crée un exercice avec:
        - Objectif: {objective_description}
        - Niveau: {level_name} 
        - Type: {objective_type}
        - Basé sur: {example_function}

        L\"exercice doit:
        1. Être clair et précis
        2. Avoir une solution détaillée
        3. Inclure 2-3 indices pédagogiques
        4. Correspondre au niveau de difficulté
        """
        return Task(
            description=prompt,
            agent=self.agent,
            expected_output="Un objet Exercise complet avec exercise, solution, hints, difficulty et concept",
            output_pydantic=Exercise
        )

    def create_similar_exercise_task(self, original_exercise: Exercise, student_level: int) -> Task:
        prompt = f"""
        En tant que professeur de mathématiques expert, crée un nouvel exercice qui est SIMILAIRE à l\"exercice original suivant, mais avec des valeurs ou un contexte légèrement différents pour offrir une nouvelle pratique. L\"exercice doit être adapté au niveau de l\"étudiant ({student_level}).

        Exercice original:
        {original_exercise.exercise}
        Solution originale: {original_exercise.solution}
        Concept: {original_exercise.concept}
        Difficulté: {original_exercise.difficulty}

        Le nouvel exercice doit:
        1. Être similaire en concept et difficulté à l\"original.
        2. Présenter des variations suffisantes pour ne pas être une simple répétition.
        3. Avoir une solution détaillée.
        4. Inclure 2-3 indices pédagogiques.
        """
        return Task(
            description=prompt,
            agent=self.agent,
            expected_output="Un objet Exercise complet avec exercise, solution, hints, difficulty et concept pour l\"exercice similaire",
            output_pydantic=Exercise
        )


