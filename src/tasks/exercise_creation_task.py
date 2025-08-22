from crewai import Task
from crewai import Agent
from src.models.models import Exercise
import json

class ExerciseCreationTask:
    def __init__(self, agent: Agent):
        self.agent = agent

    def create_task(self, objective_description: str, level_name: str, objective_type: str, example_function: str) -> Task:
        prompt = f"""
Crée un exercice de mathématiques avec les spécifications suivantes:

CONTEXTE:
- Objectif: {objective_description}
- Niveau: {level_name} 
- Type: {objective_type}
- Basé sur: {example_function}

RÈGLES:
1. Énoncé clair et adapté au niveau {level_name}
2. Solution détaillée avec étapes progressives
3. 2-3 indices pédagogiques utiles
4. Utiliser LaTeX pour les mathématiques: $x^2$, $\\frac{{a}}{{b}}$, etc.
5. Concept mathématique précis

Tu dois créer un exercice complet qui respecte exactement ces critères.
        """
        
        return Task(
            description=prompt,
            agent=self.agent,
            expected_output="Un exercice mathématique structuré avec énoncé, solution étape par étape, indices, niveau de difficulté et concept",
            output_pydantic=Exercise
        )

    def create_similar_exercise_task(self, original_exercise: Exercise, student_level: int) -> Task:
        # Convertir le niveau numérique en nom de niveau
        level_mapping = {
            1: "Débutant",
            2: "Intermédiaire", 
            3: "Avancé",
            4: "Expert"
        }
        level_name = level_mapping.get(student_level, "Intermédiaire")
        
        prompt = f"""
Crée un exercice SIMILAIRE basé sur l'exercice original ci-dessous:

EXERCICE ORIGINAL:
Énoncé: {original_exercise.exercise}
Concept: {original_exercise.concept}
Difficulté: {original_exercise.difficulty}

OBJECTIF:
- Créer un nouvel exercice du même type avec des valeurs différentes
- Niveau cible: {level_name}
- Garder le même concept mathématique: {original_exercise.concept}

EXIGENCES:
1. Varier les nombres/valeurs mais garder la même structure
2. Solution complète avec étapes détaillées
3. 2-3 indices progressifs
4. Formatage LaTeX pour les expressions mathématiques
5. Même niveau de difficulté que l'original

L'exercice doit permettre une pratique supplémentaire du même concept.
        """
        
        return Task(
            description=prompt,
            agent=self.agent,
            expected_output="Un exercice similaire avec structure identique mais valeurs différentes",
            output_pydantic=Exercise
        )

    def create_adaptive_exercise_task(self, concept: str, difficulty_level: str, student_weaknesses: list = None) -> Task:
        """Crée un exercice adaptatif basé sur les faiblesses identifiées de l'étudiant"""
        
        weaknesses_text = ""
        if student_weaknesses:
            weaknesses_text = f"\nFaiblesses identifiées de l'étudiant: {', '.join(student_weaknesses)}"
            weaknesses_text += "\nAdapter l'exercice pour aider l'étudiant à travailler sur ces points faibles."
        
        prompt = f"""
        Crée un exercice de mathématiques personnalisé avec les spécifications suivantes:
        
        - Concept principal: {concept}
        - Niveau de difficulté: {difficulty_level}
        {weaknesses_text}

        INSTRUCTIONS:
        1. L'exercice doit cibler spécifiquement le concept {concept}
        2. Le niveau de difficulté doit être exactement {difficulty_level}
        3. Si des faiblesses sont identifiées, l'exercice doit aider à les corriger
        4. Utiliser le formatage LaTeX pour toutes les expressions mathématiques
        5. Fournir une solution pédagogique détaillée
        6. Inclure des indices progressifs pour guider l'étudiant

        FORMAT DE SORTIE REQUIS:
        Retourne UNIQUEMENT un objet JSON valide avec les clés suivantes:
        - "exercise": l'énoncé de l'exercice
        - "solution": la solution complète étape par étape
        - "hints": liste de 2-3 indices progressifs
        - "difficulty": "{difficulty_level}"
        - "concept": "{concept}"
        
        Ne pas inclure d'explications supplémentaires en dehors de cet objet JSON.
        """
        
        return Task(
            description=prompt,
            agent=self.agent,
            expected_output="Un objet Exercise complet au format JSON adaptatif",
            output_pydantic=Exercise
        )