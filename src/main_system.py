import os
import json
import re
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
                model="groq/llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=2048
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

    def _clean_latex_escapes(self, json_str: str) -> str:
        """Nettoie les échappements LaTeX dans une chaîne JSON de manière robuste"""
        try:
            # Méthode 1: Échapper tous les backslashes qui ne sont pas déjà échappés
            # D'abord, protéger les backslashes déjà échappés
            json_str = json_str.replace('\\\\', '__DOUBLE_BACKSLASH_PLACEHOLDER__')
            
            # Ensuite, échapper tous les backslashes simples restants
            json_str = json_str.replace('\\', '\\\\')
            
            # Enfin, restaurer les doubles backslashes protégés
            json_str = json_str.replace('__DOUBLE_BACKSLASH_PLACEHOLDER__', '\\\\')
            
            return json_str
            
        except Exception as e:
            console.print(f"Erreur nettoyage LaTeX: {str(e)}")
            return json_str

    def _clean_latex_escapes_alternative(self, json_str: str) -> str:
        """Version alternative avec expressions régulières pour les commandes LaTeX"""
        try:
            # Pattern pour les expressions LaTeX dans les chaînes JSON
            # Cherche les backslashes suivis de lettres (commandes LaTeX) qui ne sont pas échappés
            latex_pattern = r'(?<!\\)\\([a-zA-Z]+(?:\{[^}]*\})*)'
            
            def escape_latex_match(match):
                return '\\\\' + match.group(1)
            
            # Appliquer le remplacement
            json_str = re.sub(latex_pattern, escape_latex_match, json_str)
            
            # Gérer les cas spéciaux
            special_patterns = {
                r'(?<!\\)\\mathbb\{([^}]+)\}': r'\\\\mathbb{{\1}}',
                r'(?<!\\)\\frac\{([^}]+)\}\{([^}]+)\}': r'\\\\frac{{\1}}{{\2}}',
                r'(?<!\\)\\sqrt\{([^}]+)\}': r'\\\\sqrt{{\1}}',
                r'(?<!\\)\\infty': r'\\\\infty',
                r'(?<!\\)\\pi': r'\\\\pi'
            }
            
            for pattern, replacement in special_patterns.items():
                json_str = re.sub(pattern, replacement, json_str)
            
            return json_str
            
        except Exception as e:
            console.print(f"Erreur nettoyage LaTeX alternatif: {str(e)}")
            return json_str

    def _manual_parse_exercise(self, text: str) -> Optional[Exercise]:
        """Parsing manuel en cas d'échec du JSON"""
        try:
            # Extraire les informations avec des expressions régulières
            exercise_match = re.search(r'"exercise":\s*"(.*?)"(?=\s*,)', text, re.DOTALL)
            solution_match = re.search(r'"solution":\s*"(.*?)"(?=\s*,)', text, re.DOTALL)
            hints_match = re.search(r'"hints":\s*\[(.*?)\]', text, re.DOTALL)
            difficulty_match = re.search(r'"difficulty":\s*"([^"]+)"', text)
            concept_match = re.search(r'"concept":\s*"([^"]+)"', text)
            
            if not all([exercise_match, solution_match, difficulty_match, concept_match]):
                console.print("Impossible d'extraire toutes les informations nécessaires")
                return None
            
            # Extraire les hints
            hints = []
            if hints_match:
                hints_content = hints_match.group(1)
                # Extraire chaque hint entre guillemets
                hint_matches = re.findall(r'"(.*?)"(?=\s*[,\]])', hints_content, re.DOTALL)
                hints = hint_matches
            
            return Exercise(
                exercise=exercise_match.group(1).strip(),
                solution=solution_match.group(1).strip(), 
                hints=hints if hints else ["Aucun indice disponible"],
                difficulty=difficulty_match.group(1).strip(),
                concept=concept_match.group(1).strip()
            )
            
        except Exception as e:
            console.print(f"Erreur parsing manuel: {str(e)}")
            return None

    def _parse_exercise_from_text(self, text: str) -> Optional[Exercise]:
        """Parse un exercice depuis le texte de sortie avec méthodes robustes"""
        try:
            # Chercher un JSON dans le texte
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                console.print("Aucun JSON trouvé dans le texte")
                return self._manual_parse_exercise(text)
                
            json_str = json_match.group()
            
            # Essayer plusieurs méthodes de nettoyage
            cleaning_methods = [self._clean_latex_escapes, self._clean_latex_escapes_alternative]
            
            for clean_method in cleaning_methods:
                try:
                    cleaned_json = clean_method(json_str)
                    exercise_data = json.loads(cleaned_json)
                    
                    # Valider que les champs requis sont présents
                    required_fields = ['exercise', 'solution', 'hints', 'difficulty', 'concept']
                    if all(field in exercise_data for field in required_fields):
                        return Exercise(**exercise_data)
                        
                except json.JSONDecodeError as e:
                    console.print(f"Erreur JSON avec méthode {clean_method.__name__}: {str(e)}")
                    continue
                except Exception as e:
                    console.print(f"Erreur autre avec méthode {clean_method.__name__}: {str(e)}")
                    continue
            
            # Si toutes les méthodes JSON échouent, essayer le parsing manuel
            return self._manual_parse_exercise(text)
            
        except Exception as e:
            console.print(f"Erreur parsing exercice: {str(e)}")
            console.print(f"Texte problématique: {text[:500]}...")
            return None

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

        # Exercice par défaut en cas d'échec
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
            # Prompt amélioré avec instructions très claires
            task = Task(
                description=f"""
Tu dois créer un exercice de mathématiques au format JSON STRICT sans erreur.

CONTEXTE:
- Objectif: {objective_data["description"]}
- Niveau: {level_info["name"]} 
- Type: {self.current_student.current_objective}
- Exemple basé sur: {level_info["example_functions"][0] if level_info.get("example_functions") else "fonctions de base"}

RÈGLES STRICTES POUR LE JSON:
1. UTILISE TOUJOURS des doubles backslashes pour LaTeX: \\\\frac{{a}}{{b}}, \\\\sqrt{{x}}, \\\\mathbb{{R}}
2. PAS de backslashes simples dans le JSON
3. Utilise des accolades doubles pour les paramètres LaTeX: {{a}}, {{b}}
4. Teste mentalement que le JSON est valide

FORMAT EXACT REQUIS:
{{
  "exercise": "Énoncé de l'exercice avec LaTeX correctement échappé",
  "solution": "Solution étape par étape détaillée",
  "hints": ["Indice 1", "Indice 2", "Indice 3"],
  "difficulty": "{level_info["name"]}",
  "concept": "{self.current_student.current_objective}"
}}

EXEMPLE CORRECT:
{{
  "exercise": "Calculer la dérivée de f(x) = \\\\frac{{x^2 + 1}}{{x - 2}}",
  "solution": "Utilisons la règle du quotient: si f(x) = \\\\frac{{u(x)}}{{v(x)}}, alors f'(x) = \\\\frac{{u'(x)v(x) - u(x)v'(x)}}{{[v(x)]^2}}. Ici u(x) = x^2 + 1, donc u'(x) = 2x. Et v(x) = x - 2, donc v'(x) = 1. Par conséquent: f'(x) = \\\\frac{{2x(x-2) - (x^2+1)(1)}}{{(x-2)^2}} = \\\\frac{{2x^2 - 4x - x^2 - 1}}{{(x-2)^2}} = \\\\frac{{x^2 - 4x - 1}}{{(x-2)^2}}",
  "hints": ["Utilisez la règle du quotient", "Identifiez u(x) et v(x)", "Calculez u'(x) et v'(x) séparément"],
  "difficulty": "Intermédiaire",
  "concept": "Dérivées"
}}

IMPORTANT: Réponds UNIQUEMENT avec le JSON, aucun texte supplémentaire.
                """,
                agent=self.exercise_creator_agent,
                expected_output="JSON valide uniquement"
            )

            crew = Crew(
                agents=[self.exercise_creator_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )

            result = crew.kickoff()
            
            # Extraire le texte du résultat
            result_text = ""
            if hasattr(result, 'raw'):
                result_text = str(result.raw)
            else:
                result_text = str(result)
            
            # Parser l'exercice depuis le texte
            exercise = self._parse_exercise_from_text(result_text)
            return exercise if exercise else default_exercise

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
            # Convertir le niveau numérique en nom de niveau
            level_mapping = {
                1: "Débutant",
                2: "Intermédiaire", 
                3: "Avancé",
                4: "Expert"
            }
            level_name = level_mapping.get(self.current_student.level, "Intermédiaire")
            
            task = Task(
                description=f"""
Crée un exercice SIMILAIRE basé sur l'exercice original ci-dessous.

EXERCICE ORIGINAL:
Énoncé: {original_exercise.exercise}
Concept: {original_exercise.concept}
Difficulté: {original_exercise.difficulty}

OBJECTIF:
- Créer un nouvel exercice du même type avec des valeurs/paramètres différents
- Niveau cible: {level_name}
- Garder le même concept mathématique: {original_exercise.concept}

RÈGLES STRICTES POUR LE JSON:
1. UTILISE TOUJOURS des doubles backslashes pour LaTeX: \\\\frac{{a}}{{b}}, \\\\sqrt{{x}}
2. Accolades doubles pour paramètres: {{a}}, {{b}}
3. JSON parfaitement valide

FORMAT EXACT:
{{
  "exercise": "Nouvel énoncé avec valeurs différentes",
  "solution": "Solution détaillée étape par étape",
  "hints": ["Indice 1", "Indice 2", "Indice 3"],
  "difficulty": "{original_exercise.difficulty}",
  "concept": "{original_exercise.concept}"
}}

IMPORTANT: Réponds UNIQUEMENT avec le JSON valide, aucun autre texte.
                """,
                agent=self.exercise_creator_agent,
                expected_output="JSON valide d'exercice similaire"
            )
            
            crew = Crew(
                agents=[self.exercise_creator_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Extraire le texte du résultat
            result_text = ""
            if hasattr(result, 'raw'):
                result_text = str(result.raw)
            else:
                result_text = str(result)
            
            # Parser l'exercice depuis le texte
            return self._parse_exercise_from_text(result_text)
            
        except Exception as e:
            console.print(f"Erreur lors de la génération d'un exercice similaire: {str(e)}")
            return None

    def _parse_evaluation_from_text(self, text: str) -> Optional[EvaluationResult]:
        """Parse une évaluation depuis le texte de sortie"""
        try:
            # Chercher un JSON dans le texte
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                
                # Nettoyer les échappements LaTeX problématiques
                json_str = self._clean_latex_escapes(json_str)
                
                eval_data = json.loads(json_str)
                return EvaluationResult(**eval_data)
            
            return None
            
        except Exception as e:
            console.print(f"Erreur parsing évaluation: {str(e)}")
            console.print(f"Texte problématique: {text[:500]}...")
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
            task = Task(
                description=f"""
Évalue la réponse de l'étudiant à cet exercice de mathématiques.

EXERCICE: {exercise.exercise}
SOLUTION ATTENDUE: {exercise.solution}
RÉPONSE ÉTUDIANT: {extracted_text}

ÉVALUATION REQUISE:
1. La réponse est-elle correcte ? (true/false)
2. Quel type d'erreur si incorrecte ?
3. Feedback pédagogique constructif
4. Explication détaillée de la solution
5. Correction étape par étape
6. Recommandations pour l'amélioration

FORMAT JSON EXACT:
{{
  "is_correct": true/false,
  "error_type": "Type d'erreur spécifique ou null si correct",
  "feedback": "Feedback pédagogique détaillé et constructif",
  "detailed_explanation": "Explication mathématique complète de la solution",
  "step_by_step_correction": "Correction détaillée étape par étape",
  "recommendations": ["Recommandation 1", "Recommandation 2", "Recommandation 3"]
}}

IMPORTANT: Réponds UNIQUEMENT avec le JSON valide, aucun texte supplémentaire.
                """,
                agent=self.evaluator_agent,
                expected_output="Évaluation détaillée au format JSON"
            )
            
            crew = Crew(
                agents=[self.evaluator_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Extraire le texte du résultat
            result_text = ""
            if hasattr(result, 'raw'):
                result_text = str(result.raw)
            else:
                result_text = str(result)
            
            # Parser l'évaluation depuis le texte
            evaluation = self._parse_evaluation_from_text(result_text)
            return evaluation if evaluation else self._create_fallback_evaluation(exercise)
            
        except Exception as e:
            console.print(f"Erreur évaluation réponse: {str(e)}")
            return self._create_fallback_evaluation(exercise)

    def _create_fallback_evaluation(self, exercise: Exercise) -> EvaluationResult:
        return EvaluationResult(
            is_correct=False,
            error_type="Évaluation impossible",
            feedback="Impossible d'évaluer la réponse automatiquement. Veuillez réessayer ou fournir une réponse plus claire.",
            detailed_explanation=f"La solution attendue était: {exercise.solution}",
            step_by_step_correction="Aucune correction détaillée disponible en raison de l'erreur d'évaluation.",
            recommendations=["Vérifiez votre saisie", "Reformulez votre réponse", "Contactez le support si le problème persiste"]
        )

    def _create_fallback_coaching(self) -> CoachPersonal:
        return CoachPersonal(
            motivation="Continue tes efforts, chaque étape compte dans ton apprentissage !",
            strategy="Essaie de décomposer les problèmes complexes en étapes plus simples et gérables.",
            tip="N'hésite pas à refaire les exercices pour bien maîtriser les concepts.",
            encouragement=["Tu progresses bien !", "La persévérance est la clé du succès en mathématiques.", "Chaque erreur est une opportunité d'apprendre."]
        )

    def _parse_coaching_from_text(self, text: str) -> Optional[CoachPersonal]:
        """Parse un message de coaching depuis le texte de sortie"""
        try:
            # Chercher un JSON dans le texte
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                
                # Nettoyer les échappements LaTeX problématiques
                json_str = self._clean_latex_escapes(json_str)
                
                coaching_data = json.loads(json_str)
                return CoachPersonal(**coaching_data)
            
            return None
            
        except Exception as e:
            console.print(f"Erreur parsing coaching: {str(e)}")
            console.print(f"Texte problématique: {text[:500]}...")
            return None

    def get_personal_coach_message(self) -> Optional[CoachPersonal]:
        if not self.personal_coach_agent or not self.llm:
            return self._create_fallback_coaching()
        
        try:
            task = Task(
                description="""
Génère un message de motivation personnalisé pour un étudiant en mathématiques.

OBJECTIF: Créer un message positif et constructif qui aide l'étudiant à rester motivé et engagé.

COMPOSANTS REQUIS:
1. Message de motivation inspirant et personnel
2. Stratégie concrète d'apprentissage applicable
3. Astuce pratique pour améliorer les performances en mathématiques
4. Phrases d'encouragement positives et motivantes

FORMAT JSON EXACT:
{
  "motivation": "Message motivant et inspirant personnalisé",
  "strategy": "Stratégie concrète et applicable pour l'apprentissage",
  "tip": "Astuce pratique et utile pour les mathématiques",
  "encouragement": ["Phrase positive 1", "Phrase positive 2", "Phrase positive 3"]
}

IMPORTANT: Réponds UNIQUEMENT avec le JSON valide, aucun texte supplémentaire.
                """,
                agent=self.personal_coach_agent,
                expected_output="Message de coaching complet au format JSON"
            )
            
            crew = Crew(
                agents=[self.personal_coach_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Extraire le texte du résultat
            result_text = ""
            if hasattr(result, 'raw'):
                result_text = str(result.raw)
            else:
                result_text = str(result)
            
            # Parser le coaching depuis le texte
            coaching = self._parse_coaching_from_text(result_text)
            return coaching if coaching else self._create_fallback_coaching()
            
        except Exception as e:
            console.print(f"Erreur génération message coach: {str(e)}")
            return self._create_fallback_coaching()


if __name__ == "__main__":
    system = MathTutoringSystem()
    
    # Example Usage
    if system.llm:  # Vérifier que l'API est disponible
        console.print("=== Initialisation du Système de Tutorat Mathématique ===")
        
        # Create a student
        student = system.student_manager.create_student(name="Alice")
        system.set_current_student(student.student_id)
        console.print(f"✅ Étudiant créé: {system.current_student.name} (ID: {system.current_student.student_id})")

        # Set a current objective for the student
        if system.learning_objectives.objectives_order:
            system.current_student.current_objective = system.learning_objectives.objectives_order[0]
            system.student_manager.save_student(system.current_student)
            console.print(f"✅ Objectif actuel défini: {system.current_student.current_objective}")

        console.print("\n=== Génération d'Exercice ===")
        # Generate an exercise
        exercise = system.generate_exercise()
        if exercise:
            console.print(f"📝 Exercice généré:")
            console.print(f"   Énoncé: {exercise.exercise}")
            console.print(f"   Difficulté: {exercise.difficulty}")
            console.print(f"   Concept: {exercise.concept}")
            console.print(f"   Solution: {exercise.solution}")
            console.print(f"   Indices: {exercise.hints}")

            console.print("\n=== Génération d'Exercice Similaire ===")
            # Generate a similar exercise
            similar_exercise = system.generate_similar_exercise(exercise)
            if similar_exercise:
                console.print(f"📝 Exercice similaire généré:")
                console.print(f"   Énoncé: {similar_exercise.exercise}")

            console.print("\n=== Évaluation de Réponse ===")
            # Simulate student answer
            student_answer = "Je pense que la réponse est x = 5, mais je ne suis pas sûr des étapes."
            evaluation = system.evaluate_response(exercise, student_answer)
            console.print(f"📊 Évaluation de la réponse:")
            console.print(f"   Correcte: {evaluation.is_correct}")
            console.print(f"   Type d'erreur: {evaluation.error_type}")
            console.print(f"   Feedback: {evaluation.feedback}")
            console.print(f"   Explication: {evaluation.detailed_explanation[:100]}...")

        console.print("\n=== Message du Coach Personnel ===")
        # Get a personal coach message
        coach_message = system.get_personal_coach_message()
        if coach_message:
            console.print(f"💪 Message du coach:")
            console.print(f"   Motivation: {coach_message.motivation}")
            console.print(f"   Stratégie: {coach_message.strategy}")
            console.print(f"   Astuce: {coach_message.tip}")
            console.print(f"   Encouragements: {', '.join(coach_message.encouragement)}")

        console.print("\n=== Progression de l'Étudiant ===")
        # Get student progress
        progress = system.get_student_progress()
        if progress:
            console.print(f"📈 Progression: Niveau {progress['level']}, {progress['completed']} objectifs complétés")
    else:
        console.print("⚠️  Mode hors ligne - API non disponible")
        console.print("Pour tester le système complet, configurez votre clé API GROQ dans le fichier .env")