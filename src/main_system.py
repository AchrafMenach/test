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

from src.models.models import Exercise, EvaluationResult, StudentProfile, CoachPersonal,PersonalizedCoachMessage
from src.agents.exercise_creator_agent import ExerciseCreatorAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.personal_coach_agent import PersonalCoachAgent
from src.tasks.exercise_creation_task import ExerciseCreationTask
from src.tasks.evaluation_task import EvaluationTask
from src.tools.file_processor import FileProcessor
from src.config.learning_objectives import LearningObjectives
from src.student_manager import StudentManager
from langchain_community.chat_models import ChatOllama
from src.session_manager import SessionManager

load_dotenv()

class MathTutoringSystem:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.llm = self._initialize_llm()
        
        self.file_processor = FileProcessor()
        self.session_manager = SessionManager()
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

    def set_current_student(self, student_id: str, student_name: Optional[str] = None):
        """Utilise le SessionManager au lieu de charger directement"""
        session = self.session_manager.get_or_create_session(student_id, student_name)
        self.current_student = session.student_profile
        
        # Charger les mémoires initiales si disponible
        if session.memory:
            self._load_initial_memories_from_session(session)

    def _load_initial_memories_from_session(self, session):
        """Charge les mémoires depuis la session (déjà synchronisées)"""
        if not session.memory:
            return
        
        # Les mémoires sont déjà synchronisées automatiquement par le SessionManager
        # Ici on peut optionnellement récupérer des mémoires spécifiques si nécessaire
        try:
            # Exemple : récupérer les dernières réussites pour contextualiser
            recent_achievements = session.memory.query_memory(
                query_texts=["objectif complété", "achievement"],
                n_results=5
            )
            if recent_achievements:
                console.print(f"📚 {len(recent_achievements)} mémoires chargées pour contextualisation")
        except Exception as e:
            console.print(f"⚠️ Erreur chargement mémoires contextuelles: {str(e)}")
            
    def save_current_student(self) -> bool:
        """Sauvegarde l'étudiant actuel via le SessionManager"""
        if not self.current_student:
            return False
        
        return self.session_manager.save_session(self.current_student.student_id)

    def add_exercise_to_history(self, exercise: Exercise, answer: str, is_correct: bool):
        """Ajoute un exercice à l'historique de l'étudiant actuel"""
        if not self.current_student:
            return
        
        history_item = {
            "exercise": exercise.exercise,
            "answer": answer,
            "evaluation": is_correct,
            "timestamp": datetime.now().isoformat(),
            "concept": exercise.concept
        }
        
        self.current_student.learning_history.append(history_item)
        
        # Sauvegarder automatiquement
        self.save_current_student()
        
    def get_current_objective_info(self):
        if not self.current_student:
            return None
        
        objective = self.learning_objectives.objectives.get(self.current_student.current_objective or "", {})
        if not objective:
            return None
        
        return {
            "description": objective.get("description", ""),
            "level_name": objective.get("level_name", ""),
            "total_levels": sum(
                len(theme.get("niveaux", {})) for cycle in self.learning_objectives.objectives.values()
                for theme in cycle.values()
            ),
            "objectives": objective.get("objectives", [])
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

        objective_data = self.learning_objectives.objectives.get(self.current_student.current_objective, {})

        if not objective_data:
            return None

        # Plus de "niveaux", les infos sont déjà au bon niveau
        level_info = {
            "level": objective_data.get("level"),
            "level_name": objective_data.get("level_name"),
            "objectives": objective_data.get("objectives", []),
            "example_exercises": objective_data.get("example_exercises", []),
            "example_functions": objective_data.get("example_functions", [])
        }

        # Exercice par défaut en cas d'échec avec délimiteurs mathématiques
        default_exercise = Exercise(
            exercise=f"Résoudre l'équation suivante: $x + 5 = 12$",
            solution=f"Pour résoudre $x + 5 = 12$, on soustrait 5 des deux côtés: $x = 12 - 5 = 7$",
            hints=["Isolez la variable x", "Utilisez les opérations inverses"],
            difficulty=level_info["level_name"],
            concept=self.current_student.current_objective
        )

        if not self.exercise_creator_agent or not self.exercise_creation_task:
            return default_exercise

        try:
            # Prompt amélioré avec instructions très claires pour les délimiteurs mathématiques
            task = Task(
                description=f"""
Tu dois créer un exercice de mathématiques au format JSON STRICT avec expressions mathématiques correctement délimitées.

CONTEXTE:
- Objectif: {objective_data["description"]}
- Niveau: {level_info["level_name"]} 
- Type: {self.current_student.current_objective}
- Exemple basé sur: {level_info["example_functions"][0] if level_info.get("example_functions") else "fonctions de base"}

RÈGLES STRICTES POUR LES EXPRESSIONS MATHÉMATIQUES:
1. TOUJOURS encadrer les expressions mathématiques avec des délimiteurs:
   - Pour les expressions INLINE: $expression$ (un seul dollar de chaque côté)
   - Pour les expressions EN BLOC: $$expression$$ (deux dollars de chaque côté)
2. Exemples corrects:
   - Inline: "Résoudre $x^2 + 3x - 4 = 0$"
   - Bloc: "La dérivée est: $$f'(x) = 2x + 3$$"
3. UTILISE des doubles backslashes pour LaTeX: \\\\frac{{a}}{{b}}, \\\\sqrt{{x}}
4. PAS de backslashes simples dans le JSON
5. Accolades doubles pour paramètres: {{a}}, {{b}}

FORMAT EXACT REQUIS:
{{
  "exercise": "Énoncé avec expressions mathématiques délimitées par $ ou $$",
  "solution": "Solution détaillée avec expressions mathématiques délimitées",
  "hints": ["Indice 1 avec $math$ si nécessaire", "Indice 2", "Indice 3"],
  "difficulty": "{level_info["level_name"]}",
  "concept": "{self.current_student.current_objective}"
}}

EXEMPLE CORRECT:
{{
  "exercise": "Calculer la dérivée de la fonction $f(x) = \\\\frac{{x^2 + 1}}{{x - 2}}$",
  "solution": "Utilisons la règle du quotient: si $f(x) = \\\\frac{{u(x)}}{{v(x)}}$, alors $$f'(x) = \\\\frac{{u'(x)v(x) - u(x)v'(x)}}{{[v(x)]^2}}$$ Ici $u(x) = x^2 + 1$, donc $u'(x) = 2x$. Et $v(x) = x - 2$, donc $v'(x) = 1$. Par conséquent: $$f'(x) = \\\\frac{{2x(x-2) - (x^2+1)(1)}}{{(x-2)^2}} = \\\\frac{{x^2 - 4x - 1}}{{(x-2)^2}}$$",
  "hints": ["Utilisez la règle du quotient: $\\\\frac{{d}}{{dx}}[\\\\frac{{u}}{{v}}] = \\\\frac{{u'v - uv'}}{{v^2}}$", "Identifiez $u(x) = x^2 + 1$ et $v(x) = x - 2$", "Calculez $u'(x)$ et $v'(x)$ séparément"],
  "difficulty": "Intermédiaire",
  "concept": "Dérivées"
}}

IMPORTANT: 
- Réponds UNIQUEMENT avec le JSON valide
- N'oublie JAMAIS les délimiteurs $ pour les expressions mathématiques
- Teste mentalement que chaque expression mathématique est bien encadrée
                """,
                agent=self.exercise_creator_agent,
                expected_output="JSON valide avec expressions mathématiques correctement délimitées"
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
Crée un exercice SIMILAIRE basé sur l'exercice original ci-dessous, avec expressions mathématiques correctement délimitées.

EXERCICE ORIGINAL:
Énoncé: {original_exercise.exercise}
Concept: {original_exercise.concept}
Difficulté: {original_exercise.difficulty}

OBJECTIF:
- Créer un nouvel exercice du même type avec des valeurs/paramètres différents
- Niveau cible: {level_name}
- Garder le même concept mathématique: {original_exercise.concept}

RÈGLES STRICTES POUR LES EXPRESSIONS MATHÉMATIQUES:
1. TOUJOURS encadrer les expressions mathématiques:
   - Expressions INLINE: $expression$
   - Expressions EN BLOC: $$expression$$
2. UTILISE des doubles backslashes pour LaTeX: \\\\frac{{a}}{{b}}, \\\\sqrt{{x}}
3. Accolades doubles pour paramètres: {{a}}, {{b}}
4. JSON parfaitement valide

FORMAT EXACT:
{{
  "exercise": "Nouvel énoncé avec expressions mathématiques délimitées par $ ou $$",
  "solution": "Solution détaillée avec expressions mathématiques délimitées",
  "hints": ["Indice 1 avec $math$ si nécessaire", "Indice 2", "Indice 3"],
  "difficulty": "{original_exercise.difficulty}",
  "concept": "{original_exercise.concept}"
}}

EXEMPLE:
Si l'original était: "Dériver $f(x) = x^2 + 3x$"
Le similaire pourrait être: "Dériver $g(x) = 2x^3 - 5x + 1$"

IMPORTANT: 
- Réponds UNIQUEMENT avec le JSON valide
- N'oublie JAMAIS les délimiteurs $ pour les expressions mathématiques
                """,
                agent=self.exercise_creator_agent,
                expected_output="JSON valide d'exercice similaire avec expressions mathématiques délimitées"
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
3. Feedback pédagogique constructif avec expressions mathématiques délimitées
4. Explication détaillée de la solution avec expressions mathématiques délimitées
5. Correction étape par étape avec expressions mathématiques délimitées
6. Recommandations pour l'amélioration

RÈGLES POUR LES EXPRESSIONS MATHÉMATIQUES:
- TOUJOURS encadrer les expressions mathématiques:
  - Expressions INLINE: $expression$
  - Expressions EN BLOC: $$expression$$
- Doubles backslashes pour LaTeX: \\\\frac{{a}}{{b}}, \\\\sqrt{{x}}

FORMAT JSON EXACT:
{{
  "is_correct": true/false,
  "error_type": "Type d'erreur spécifique ou null si correct",
  "feedback": "Feedback pédagogique avec expressions mathématiques délimitées par $ ou $$",
  "detailed_explanation": "Explication mathématique complète avec expressions délimitées",
  "step_by_step_correction": "Correction détaillée avec expressions mathématiques délimitées",
  "recommendations": ["Recommandation 1", "Recommandation 2", "Recommandation 3"]
}}

EXEMPLE de feedback avec math:
"feedback": "Votre approche est correcte mais vous avez fait une erreur dans le calcul de $\\\\frac{{d}}{{dx}}[x^2] = 2x$. La dérivée finale devrait être $f'(x) = 2x + 3$."

IMPORTANT: Réponds UNIQUEMENT avec le JSON valide avec expressions mathématiques correctement délimitées.
                """,
                agent=self.evaluator_agent,
                expected_output="Évaluation détaillée au format JSON avec expressions mathématiques délimitées"
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

Si tu inclus des expressions mathématiques, encadre-les avec $ pour inline ou $$ pour bloc.

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
        
    def check_objective_completion(self) -> bool:
        """
        Vérifie si l'étudiant a terminé son objectif actuel
        Critères : nombre d'exercices réussis, progression dans les niveaux, etc.
        """
        if not self.current_student or not self.current_student.current_objective:
            return False
        
        # Analyser l'historique d'apprentissage récent
        recent_exercises = self.current_student.learning_history[-10:]  # 10 derniers exercices
        if len(recent_exercises) < 5:  # Pas assez d'exercices pour évaluer
            return False
        
        # Calculer le taux de réussite récent
        correct_answers = sum(1 for ex in recent_exercises if ex.get('evaluation', False))
        success_rate = correct_answers / len(recent_exercises)
        
        # Critères de completion (ajustables)
        if success_rate >= 0.8:  # 80% de réussite sur les derniers exercices
            return True
        
        return False
    
    
    def get_session_stats(self) -> dict:
        """Récupère les statistiques des sessions actives"""
        return self.session_manager.get_sessions_info()

    def cleanup_expired_sessions(self) -> int:
        """Nettoie les sessions expirées (peut être appelé manuellement)"""
        return self.session_manager.cleanup_expired_sessions()

    def shutdown(self):
        """Arrêt propre du système avec sauvegarde des sessions"""
        console.print("🛑 Arrêt du système de tutorat...")
        if hasattr(self, 'session_manager'):
            self.session_manager.shutdown()
        console.print("✅ Système arrêté proprement")

    
    
    def advance_to_next_objective(self) -> bool:
        """Version mise à jour avec sauvegarde via SessionManager"""
        if not self.current_student or not self.current_student.current_objective:
            return False
        
        # Ajouter l'objectif actuel aux objectifs complétés
        current_obj = self.current_student.current_objective
        if current_obj not in self.current_student.objectives_completed:
            self.current_student.objectives_completed.append(current_obj)
        
        # Trouver l'index de l'objectif actuel
        try:
            current_index = self.learning_objectives.objectives_order.index(current_obj)
        except ValueError:
            console.print(f"Objectif actuel non trouvé dans l'ordre: {current_obj}")
            return False
        
        # Vérifier s'il y a un objectif suivant
        if current_index + 1 < len(self.learning_objectives.objectives_order):
            next_objective = self.learning_objectives.objectives_order[current_index + 1]
            self.current_student.current_objective = next_objective
            
            # Optionnel : augmenter le niveau si approprié
            if self.current_student.level < 4:  # Maximum niveau 4
                self.current_student.level += 1
            
            # Sauvegarder les changements via SessionManager
            self.save_current_student()
            
            console.print(f"✅ Progression vers: {next_objective} (Niveau {self.current_student.level})")
            return True
        else:
            # Tous les objectifs sont terminés
            self.current_student.current_objective = None
            self.save_current_student()
            console.print("🎉 Tous les objectifs ont été complétés !")
            return False
        
    def get_progression_status(self) -> dict:
        """
        Retourne le statut de progression détaillé
        """
        if not self.current_student:
            return {"error": "Aucun étudiant sélectionné"}
        
        total_objectives = len(self.learning_objectives.objectives_order)
        completed_count = len(self.current_student.objectives_completed)
        current_obj = self.current_student.current_objective
        
        # Calculer le pourcentage de progression
        if current_obj and current_obj in self.learning_objectives.objectives_order:
            current_index = self.learning_objectives.objectives_order.index(current_obj)
            progress_percentage = (completed_count / total_objectives) * 100
        else:
            progress_percentage = (completed_count / total_objectives) * 100
        
        # Vérifier si prêt pour la progression
        ready_to_advance = self.check_objective_completion()
        
        return {
            "total_objectives": total_objectives,
            "completed_objectives": completed_count,
            "current_objective": current_obj,
            "progress_percentage": round(progress_percentage, 1),
            "current_level": self.current_student.level,
            "ready_to_advance": ready_to_advance,
            "next_objective": self._get_next_objective(),
            "recent_success_rate": self._calculate_recent_success_rate()
        }

    def _get_next_objective(self) -> str:
        """Retourne le prochain objectif ou None si terminé"""
        if not self.current_student or not self.current_student.current_objective:
            return None
        
        try:
            current_index = self.learning_objectives.objectives_order.index(
                self.current_student.current_objective
            )
            if current_index + 1 < len(self.learning_objectives.objectives_order):
                return self.learning_objectives.objectives_order[current_index + 1]
        except (ValueError, IndexError):
            pass
        
        return None

    def _calculate_recent_success_rate(self) -> float:
        """Calcule le taux de réussite récent"""
        if not self.current_student or not self.current_student.learning_history:
            return 0.0
        
        recent_exercises = self.current_student.learning_history[-10:]
        if not recent_exercises:
            return 0.0
        
        correct_count = sum(1 for ex in recent_exercises if ex.get('evaluation', False))
        return round((correct_count / len(recent_exercises)) * 100, 1)

    def auto_check_and_advance(self) -> dict:
        """
        Vérifie automatiquement et fait progresser l'étudiant si les critères sont remplis
        Utilisé après chaque évaluation d'exercice
        """
        result = {
            "progression_occurred": False,
            "message": "",
            "new_objective": None,
            "new_level": None
        }
        
        if self.check_objective_completion():
            if self.advance_to_next_objective():
                result["progression_occurred"] = True
                result["message"] = "Félicitations ! Vous avez terminé cet objectif et progressé vers le suivant."
                result["new_objective"] = self.current_student.current_objective
                result["new_level"] = self.current_student.level
            else:
                result["message"] = "Félicitations ! Vous avez terminé tous les objectifs du programme !"
        
        return result

    # Nouvelles méthodes pour l'interface web
    def generate_exercise_for_api(self, student_id: str) -> dict:
        """
        Version API de génération d'exercice qui retourne un dictionnaire compatible avec l'interface web
        """
        self.set_current_student(student_id)
        exercise = self.generate_exercise()
        
        if not exercise:
            return {
                "error": "Impossible de générer un exercice",
                "exercise": None
            }
        
        # Retourner dans le format attendu par l'interface
        return {
            "exercise": exercise.exercise,
            "solution": exercise.solution,
            "hints": exercise.hints,
            "difficulty": exercise.difficulty,
            "concept": exercise.concept,
            "context": None,  # Peut être ajouté plus tard si nécessaire
            "objective": self.current_student.current_objective if self.current_student else None
        }

    def generate_similar_exercise_for_api(self, original_exercise_data: dict) -> dict:
        """
        Version API de génération d'exercice similaire
        """
        if not self.current_student:
            return {
                "error": "Aucun étudiant sélectionné",
                "exercise": None
            }
        
        # Créer un objet Exercise à partir des données
        original_exercise = Exercise(
            exercise=original_exercise_data.get("exercise", ""),
            solution=original_exercise_data.get("solution", ""),
            hints=original_exercise_data.get("hints", []),
            difficulty=original_exercise_data.get("difficulty", ""),
            concept=original_exercise_data.get("concept", "")
        )
        
        similar_exercise = self.generate_similar_exercise(original_exercise)
        
        if not similar_exercise:
            return {
                "error": "Impossible de générer un exercice similaire",
                "exercise": None
            }
        
        return {
            "exercise": similar_exercise.exercise,
            "solution": similar_exercise.solution,
            "hints": similar_exercise.hints,
            "difficulty": similar_exercise.difficulty,
            "concept": similar_exercise.concept,
            "context": None,
            "objective": self.current_student.current_objective if self.current_student else None
        }

    def evaluate_answer_for_api(self, exercise_data: dict, answer: str, student_id: str) -> dict:
        """
        Version API d'évaluation d'une réponse textuelle avec gestion de la progression
        """
        self.set_current_student(student_id)
        
        # Créer un objet Exercise à partir des données
        exercise = Exercise(
            exercise=exercise_data.get("exercise", ""),
            solution=exercise_data.get("solution", ""),
            hints=exercise_data.get("hints", []),
            difficulty=exercise_data.get("difficulty", ""),
            concept=exercise_data.get("concept", "")
        )
        
        # Évaluer la réponse
        evaluation = self.evaluate_response(exercise, answer)
        
        # Enregistrer dans l'historique de l'étudiant
        if self.current_student:
            self.current_student.learning_history.append({
                "exercise": exercise.exercise,
                "answer": answer,
                "evaluation": evaluation.is_correct,
                "timestamp": datetime.now().isoformat(),
                "concept": exercise.concept
            })
            self.student_manager.save_student(self.current_student)
        
        # Vérifier la progression
        progression_result = self.auto_check_and_advance()
        
        # Adapter le format de retour pour l'interface
        api_result = {
            "evaluation": {
                "is_correct": evaluation.is_correct,
                "feedback": evaluation.feedback if hasattr(evaluation, 'feedback') else evaluation.detailed_explanation,
                "explanation": evaluation.detailed_explanation if hasattr(evaluation, 'detailed_explanation') else evaluation.feedback,
                "correct_answer": exercise.solution,
                "error_type": evaluation.error_type if hasattr(evaluation, 'error_type') else None,
                "recommendations": evaluation.recommendations if hasattr(evaluation, 'recommendations') else []
            }
        }
        
        # Ajouter les informations de progression si applicable
        if progression_result["progression_occurred"]:
            api_result["progression"] = {
                "level_up": True,
                "new_objective": progression_result["new_objective"],
                "new_level": progression_result["new_level"],
                "message": progression_result["message"]
            }
        
        return api_result

    def evaluate_file_answer_for_api(self, exercise_data: dict, file_path: str, student_id: str) -> dict:
        """
        Version API d'évaluation d'une réponse fichier avec gestion de la progression
        """
        self.set_current_student(student_id)
        
        # Créer un objet Exercise à partir des données
        exercise = Exercise(
            exercise=exercise_data.get("exercise", ""),
            solution=exercise_data.get("solution", ""),
            hints=exercise_data.get("hints", []),
            difficulty=exercise_data.get("difficulty", ""),
            concept=exercise_data.get("concept", "")
        )
        
        # Évaluer la réponse à partir du fichier
        evaluation = self.evaluate_response(exercise, Path(file_path))
        
        # Enregistrer dans l'historique de l'étudiant
        if self.current_student:
            self.current_student.learning_history.append({
                "exercise": exercise.exercise,
                "answer": f"Fichier: {Path(file_path).name}",
                "evaluation": evaluation.is_correct,
                "timestamp": datetime.now().isoformat(),
                "concept": exercise.concept
            })
            self.student_manager.save_student(self.current_student)
        
        # Vérifier la progression
        progression_result = self.auto_check_and_advance()
        
        # Adapter le format de retour pour l'interface
        api_result = {
            "evaluation": {
                "is_correct": evaluation.is_correct,
                "feedback": evaluation.feedback if hasattr(evaluation, 'feedback') else evaluation.detailed_explanation,
                "explanation": evaluation.detailed_explanation if hasattr(evaluation, 'detailed_explanation') else evaluation.feedback,
                "correct_answer": exercise.solution,
                "error_type": evaluation.error_type if hasattr(evaluation, 'error_type') else None,
                "recommendations": evaluation.recommendations if hasattr(evaluation, 'recommendations') else []
            }
        }
        
        # Ajouter les informations de progression si applicable
        if progression_result["progression_occurred"]:
            api_result["progression"] = {
                "level_up": True,
                "new_objective": progression_result["new_objective"],
                "new_level": progression_result["new_level"],
                "message": progression_result["message"]
            }
        
        return api_result
    
    def get_personalized_coach_message(self, exercise: Optional[Exercise] = None, 
                                    student_answer: Optional[str] = None,
                                    evaluation: Optional[EvaluationResult] = None) -> Optional[PersonalizedCoachMessage]:
        """
        Génère un message de coaching personnalisé basé sur l'exercice et la réponse de l'étudiant
        """
        if not self.personal_coach_agent or not self.llm:
            return self._create_fallback_personalized_coaching()
        
        try:
            # Construire le contexte pour le coach
            context_parts = []
            
            if exercise:
                context_parts.append(f"EXERCICE: {exercise.exercise}")
                context_parts.append(f"SOLUTION ATTENDUE: {exercise.solution}")
                context_parts.append(f"CONCEPT: {exercise.concept}")
                context_parts.append(f"DIFFICULTÉ: {exercise.difficulty}")
            
            if student_answer:
                context_parts.append(f"RÉPONSE DE L'ÉTUDIANT: {student_answer}")
            
            if evaluation:
                context_parts.append(f"ÉVALUATION: {'Correcte' if evaluation.is_correct else 'Incorrecte'}")
                if evaluation.error_type:
                    context_parts.append(f"TYPE D'ERREUR: {evaluation.error_type}")
            
            # Ajouter l'historique de l'étudiant pour plus de contexte
            if self.current_student and self.current_student.learning_history:
                recent_history = self.current_student.learning_history[-5:]  # 5 derniers exercices
                success_rate = sum(1 for h in recent_history if h.get('evaluation', False)) / len(recent_history)
                context_parts.append(f"TAUX DE RÉUSSITE RÉCENT: {success_rate:.1%}")
                context_parts.append(f"NIVEAU ÉTUDIANT: {self.current_student.level}")
                context_parts.append(f"OBJECTIF ACTUEL: {self.current_student.current_objective}")
            
            context_str = "\n".join(context_parts)
            
            task = Task(
                description=f"""
    Tu es un coach mathématique IA personnalisé. Génère un message de coaching adapté à la situation spécifique de l'étudiant.

    CONTEXTE DE L'ÉTUDIANT:
    {context_str}

    ANALYSE REQUISE:
    1. Analyse la réponse de l'étudiant par rapport à l'exercice
    2. Identifie ses forces et faiblesses spécifiques
    3. Adapte le coaching à son niveau et ses besoins

    COACHING PERSONNALISÉ REQUIS:
    1. **motivation**: Message motivant basé sur sa performance actuelle
    2. **strategy**: Stratégie spécifique pour améliorer ses points faibles identifiés
    3. **tip**: Astuce ciblée pour le concept mathématique en question
    4. **encouragement**: Liste de phrases positives adaptées à sa situation
    5. **next_steps**: Liste d'étapes concrètes recommandées pour progresser

    RÈGLES POUR LES EXPRESSIONS MATHÉMATIQUES:
    - Encadrer avec $ pour inline ou $$ pour bloc
    - Doubles backslashes: \\\\frac{{a}}{{b}}

    EXEMPLES D'ADAPTATION:
    - Si l'étudiant a fait une erreur de calcul → Focus sur la méthode, pas juste la motivation
    - Si l'étudiant a la bonne approche mais mauvaise exécution → Encourager l'approche, corriger l'exécution
    - Si l'étudiant est complètement perdu → Décomposer en étapes plus simples
    - Si l'étudiant réussit bien → Défis plus avancés

    FORMAT JSON EXACT:
    {{
    "motivation": "Message motivant personnalisé basé sur sa performance",
    "strategy": "Stratégie spécifique pour ses besoins identifiés",
    "tip": "Astuce ciblée pour le concept avec expressions mathématiques délimitées",
    "encouragement": ["Encouragement spécifique 1", "Encouragement spécifique 2", "Encouragement spécifique 3"],
    "next_steps": ["Étape concrète 1", "Étape concrète 2", "Étape concrète 3"]
    }}

    IMPORTANT: Réponds UNIQUEMENT avec le JSON valide, sois spécifique et personnalisé.
                """,
                agent=self.personal_coach_agent,
                expected_output="Message de coaching personnalisé au format JSON"
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
            coaching = self._parse_personalized_coaching_from_text(result_text)
            return coaching if coaching else self._create_fallback_personalized_coaching()
            
        except Exception as e:
            console.print(f"Erreur génération coaching personnalisé: {str(e)}")
            return self._create_fallback_personalized_coaching()

    def _parse_personalized_coaching_from_text(self, text: str) -> Optional[PersonalizedCoachMessage]:
        """Parse un message de coaching personnalisé depuis le texte de sortie"""
        try:
            # Chercher un JSON dans le texte
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                
                # Nettoyer les échappements LaTeX problématiques
                json_str = self._clean_latex_escapes(json_str)
                
                coaching_data = json.loads(json_str)
                return PersonalizedCoachMessage(**coaching_data)
            
            return None
            
        except Exception as e:
            console.print(f"Erreur parsing coaching personnalisé: {str(e)}")
            console.print(f"Texte problématique: {text[:500]}...")
            return None

    def _create_fallback_personalized_coaching(self) -> PersonalizedCoachMessage:
        return PersonalizedCoachMessage(
            motivation="Continue tes efforts, chaque étape compte dans ton apprentissage !",
            strategy="Essaie de décomposer les problèmes complexes en étapes plus simples et gérables.",
            tip="N'hésite pas à refaire les exercices pour bien maîtriser les concepts.",
            encouragement=["Tu progresses bien !", "La persévérance est la clé du succès en mathématiques.", "Chaque erreur est une opportunité d'apprendre."],
            next_steps=["Révise les concepts de base", "Pratique avec des exercices similaires", "N'hésite pas à demander de l'aide"]
        )
        
        
        
    def evaluate_answer_for_api_with_coaching(self, exercise_data: dict, answer: str, student_id: str) -> dict:
        """Version API d'évaluation avec coaching personnalisé et gestion de session"""
        # Utiliser set_current_student qui gère maintenant les sessions
        self.set_current_student(student_id)
        
        # Créer un objet Exercise à partir des données
        exercise = Exercise(
            exercise=exercise_data.get("exercise", ""),
            solution=exercise_data.get("solution", ""),
            hints=exercise_data.get("hints", []),
            difficulty=exercise_data.get("difficulty", ""),
            concept=exercise_data.get("concept", "")
        )
        
        # Évaluer la réponse
        evaluation = self.evaluate_response(exercise, answer)
        
        # Générer le coaching personnalisé
        personalized_coaching = self.get_personalized_coach_message(
            exercise=exercise,
            student_answer=answer,
            evaluation=evaluation
        )
        
        # Ajouter à l'historique avec sauvegarde automatique
        self.add_exercise_to_history(exercise, answer, evaluation.is_correct)
        
        # Vérifier la progression
        progression_result = self.auto_check_and_advance()
        
        # Retourner le résultat complet avec coaching personnalisé
        api_result = {
            "evaluation": {
                "is_correct": evaluation.is_correct,
                "feedback": evaluation.feedback if hasattr(evaluation, 'feedback') else evaluation.detailed_explanation,
                "explanation": evaluation.detailed_explanation if hasattr(evaluation, 'detailed_explanation') else evaluation.feedback,
                "correct_answer": exercise.solution,
                "error_type": evaluation.error_type if hasattr(evaluation, 'error_type') else None,
                "recommendations": evaluation.recommendations if hasattr(evaluation, 'recommendations') else []
            },
            "personalized_coaching": {
                "motivation": personalized_coaching.motivation,
                "strategy": personalized_coaching.strategy,
                "tip": personalized_coaching.tip,
                "encouragement": personalized_coaching.encouragement,
                "next_steps": personalized_coaching.next_steps
            } if personalized_coaching else None,
            "session_info": {
                "session_active": True,
                "last_activity": datetime.now().isoformat(),
                "exercises_completed": len(self.current_student.learning_history)
            }
        }
        
        # Ajouter les informations de progression si applicable
        if progression_result["progression_occurred"]:
            api_result["progression"] = {
                "level_up": True,
                "new_objective": progression_result["new_objective"],
                "new_level": progression_result["new_level"],
                "message": progression_result["message"]
            }
        
        return api_result
    
    

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

        console.print("\n=== Test API - Génération d'Exercice ===")
        # Test API exercise generation
        exercise_result = system.generate_exercise_for_api(student.student_id)
        if not exercise_result.get("error"):
            console.print(f"📝 Exercice généré via API:")
            console.print(f"   Énoncé: {exercise_result['exercise']}")
            console.print(f"   Difficulté: {exercise_result['difficulty']}")
            console.print(f"   Concept: {exercise_result['concept']}")

            console.print("\n=== Test API - Évaluation de Réponse ===")
            # Test API answer evaluation
            student_answer = "Je pense que la réponse est $x = 5$, mais je ne suis pas sûr des étapes."
            eval_result = system.evaluate_answer_for_api(exercise_result, student_answer, student.student_id)
            console.print(f"📊 Évaluation via API:")
            console.print(f"   Correcte: {eval_result['evaluation']['is_correct']}")
            console.print(f"   Feedback: {eval_result['evaluation']['feedback'][:100]}...")
            
            if eval_result.get("progression"):
                console.print(f"🎉 Progression détectée: {eval_result['progression']['message']}")

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