from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Union, List
import uvicorn
from pathlib import Path
import tempfile
import os
from datetime import datetime  # Import manquant ajouté
from typing import List, Dict

# Importez votre système existant
from src.main_system import MathTutoringSystem
from src.models.models import Exercise, EvaluationResult, StudentProfile, CoachPersonal

from src.session_manager import SessionManager
import atexit
# Modèles Pydantic pour l'API
class StudentCreate(BaseModel):
    name: str

class StudentResponse(BaseModel):
    student_id: str
    name: str
    level: int
    current_objective: Optional[str] = None
    objectives_completed: List[str] = []

class ExerciseRequest(BaseModel):
    student_id: str

class AnswerSubmission(BaseModel):
    exercise: Exercise
    answer: str
    student_id: str

class ProgressResponse(BaseModel):
    level: int
    completed: int
    current_objective: Optional[str] = None
    objectives_completed: List[str] = []
class ProgressionStatus(BaseModel):
    total_objectives: int
    completed_objectives: int
    current_objective: Optional[str]
    progress_percentage: float
    current_level: int
    ready_to_advance: bool
    next_objective: Optional[str]
    recent_success_rate: float

class ProgressionResult(BaseModel):
    progression_occurred: bool
    message: str
    new_objective: Optional[str] = None
    new_level: Optional[int] = None
# Modèles pour le test de niveau
class LevelTestQuestion(BaseModel):
    id: str
    objective: str
    level: int
    question: str
    options: List[str]
    correct_answer: int  # Index de la bonne réponse
    explanation: str

class LevelTestResponse(BaseModel):
    question_id: str
    selected_answer: int

class LevelTestSubmission(BaseModel):
    student_id: str
    responses: List[LevelTestResponse]

class LevelTestResult(BaseModel):
    student_id: str
    total_questions: int
    correct_answers: int
    score_percentage: float
    recommended_level: int
    objective_scores: Dict[str, Dict[str, int]]  # objective -> {correct, total}
    detailed_feedback: str

class LevelTestStart(BaseModel):
    objectives: List[str]  # Objectifs à tester
    questions_per_objective: int = 2
    max_level_per_objective: int = 3

# Initialisation FastAPI
app = FastAPI(
    title="Math Tutoring System API",
    description="API pour le système de tutorat en mathématiques avec CrewAI",
    version="1.0.0"
)

# Configuration CORS pour permettre les requêtes depuis React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001",
                   "http://localhost:5173" ,
                   "http://localhost:3000",
                   "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instance globale du système de tutorat
math_system = MathTutoringSystem()
session_manager = math_system.session_manager

# Classe pour générer les tests de niveau
class LevelTestGenerator:
    def __init__(self, math_system: MathTutoringSystem):
        self.math_system = math_system
        self.test_questions = self._initialize_test_questions()
    
    def _initialize_test_questions(self) -> Dict[str, Dict[int, List[LevelTestQuestion]]]:
        """
        Initialise une banque de questions organisée par objectif et niveau
        Structure: {objective: {level: [questions...]}}
        """
        questions = {}
        
        # Questions pour "Domaine de définition"
        questions["Domaine de définition"] = {
            1: [
                LevelTestQuestion(
                    id="dd_1_1",
                    objective="Domaine de définition",
                    level=1,
                    question="Quel est le domaine de définition de la fonction $f(x) = x^2 + 3x - 1$ ?",
                    options=["$\\mathbb{R}$", "$\\mathbb{R}^*$", "$[0, +∞[$", "$]-∞, 0]$"],
                    correct_answer=0,
                    explanation="Une fonction polynomiale est définie sur tous les réels."
                ),
                LevelTestQuestion(
                    id="dd_1_2",
                    objective="Domaine de définition",
                    level=1,
                    question="Pour une fonction polynomiale $g(x) = ax^3 + bx^2 + cx + d$, le domaine de définition est:",
                    options=["$\\mathbb{R}^+$", "$\\mathbb{R}$", "Cela dépend des coefficients", "$\\mathbb{R}^*$"],
                    correct_answer=1,
                    explanation="Toute fonction polynomiale est définie sur l'ensemble des réels."
                )
            ],
            2: [
                LevelTestQuestion(
                    id="dd_2_1",
                    objective="Domaine de définition",
                    level=2,
                    question="Quel est le domaine de définition de $f(x) = \\frac{x+1}{x-3}$ ?",
                    options=["$\\mathbb{R}$", "$\\mathbb{R} \\setminus \\{3\\}$", "$\\mathbb{R} \\setminus \\{-1\\}$", "$\\mathbb{R} \\setminus \\{-1, 3\\}$"],
                    correct_answer=1,
                    explanation="La fonction n'est pas définie quand le dénominateur s'annule, c'est-à-dire quand $x = 3$."
                ),
                LevelTestQuestion(
                    id="dd_2_2",
                    objective="Domaine de définition",
                    level=2,
                    question="Pour quelle valeur de $x$ la fonction $h(x) = \\frac{2x-1}{x+5}$ n'est-elle pas définie ?",
                    options=["$x = \\frac{1}{2}$", "$x = -5$", "$x = 5$", "$x = -\\frac{1}{2}$"],
                    correct_answer=1,
                    explanation="La fonction n'est pas définie quand $x + 5 = 0$, soit $x = -5$."
                )
            ],
            3: [
                LevelTestQuestion(
                    id="dd_3_1",
                    objective="Domaine de définition",
                    level=3,
                    question="Quel est le domaine de définition de $f(x) = \\frac{x}{x^2-4}$ ?",
                    options=["$\\mathbb{R} \\setminus \\{2\\}$", "$\\mathbb{R} \\setminus \\{-2, 2\\}$", "$\\mathbb{R} \\setminus \\{4\\}$", "$\\mathbb{R}$"],
                    correct_answer=1,
                    explanation="$x^2 - 4 = (x-2)(x+2) = 0$ quand $x = 2$ ou $x = -2$."
                ),
                LevelTestQuestion(
                    id="dd_3_2",
                    objective="Domaine de définition",
                    level=3,
                    question="Pour la fonction $g(x) = \\frac{1}{x^2+x-6}$, quelles sont les valeurs interdites ?",
                    options=["$x = 2$ et $x = 3$", "$x = -3$ et $x = 2$", "$x = -2$ et $x = 3$", "$x = 1$ et $x = 6$"],
                    correct_answer=1,
                    explanation="$x^2 + x - 6 = (x+3)(x-2) = 0$ quand $x = -3$ ou $x = 2$."
                )
            ]
        }
        
        # Questions pour "Calcul des limites"
        questions["Calcul des limites"] = {
            1: [
                LevelTestQuestion(
                    id="cl_1_1",
                    objective="Calcul des limites",
                    level=1,
                    question="Quelle est $\\lim_{x \\to 2} (3x + 1)$ ?",
                    options=["6", "7", "5", "La limite n'existe pas"],
                    correct_answer=1,
                    explanation="Pour une fonction polynomiale, la limite en un point est la valeur de la fonction en ce point: $3(2) + 1 = 7$."
                ),
                LevelTestQuestion(
                    id="cl_1_2",
                    objective="Calcul des limites",
                    level=1,
                    question="Que vaut $\\lim_{x \\to +∞} (x^2 - x + 1)$ ?",
                    options=["$+∞$", "$-∞$", "0", "1"],
                    correct_answer=0,
                    explanation="Le terme dominant $x^2$ tend vers $+∞$ quand $x \\to +∞$."
                )
            ],
            2: [
                LevelTestQuestion(
                    id="cl_2_1",
                    objective="Calcul des limites",
                    level=2,
                    question="Quelle est $\\lim_{x \\to +∞} \\frac{2x^2 + x}{x^2 + 3}$ ?",
                    options=["0", "1", "2", "$+∞$"],
                    correct_answer=2,
                    explanation="En divisant par $x^2$ : $\\lim_{x \\to +∞} \\frac{2 + \\frac{1}{x}}{1 + \\frac{3}{x^2}} = \\frac{2}{1} = 2$."
                ),
                LevelTestQuestion(
                    id="cl_2_2",
                    objective="Calcul des limites",
                    level=2,
                    question="Que vaut $\\lim_{x \\to +∞} \\frac{x + 1}{3x - 2}$ ?",
                    options=["$\\frac{1}{3}$", "0", "3", "$+∞$"],
                    correct_answer=0,
                    explanation="Les degrés sont égaux, la limite est le rapport des coefficients dominants: $\\frac{1}{3}$."
                )
            ],
            3: [
                LevelTestQuestion(
                    id="cl_3_1",
                    objective="Calcul des limites",
                    level=3,
                    question="Quelle est $\\lim_{x \\to 1} \\frac{x^2 - 1}{x - 1}$ ?",
                    options=["0", "1", "2", "La limite n'existe pas"],
                    correct_answer=2,
                    explanation="Forme $\\frac{0}{0}$ : $\\frac{x^2-1}{x-1} = \\frac{(x-1)(x+1)}{x-1} = x+1 \\to 1+1 = 2$."
                ),
                LevelTestQuestion(
                    id="cl_3_2",
                    objective="Calcul des limites",
                    level=3,
                    question="Que vaut $\\lim_{x \\to 0} \\frac{\\sin(x)}{x}$ ?",
                    options=["0", "1", "$+∞$", "La limite n'existe pas"],
                    correct_answer=1,
                    explanation="C'est une limite remarquable fondamentale : $\\lim_{x \\to 0} \\frac{\\sin(x)}{x} = 1$."
                )
            ]
        }
        
        return questions
    
    def generate_test(self, objectives: List[str], questions_per_objective: int = 2, 
                     max_level_per_objective: int = 3) -> List[LevelTestQuestion]:
        """
        Génère un test avec des questions pour les objectifs spécifiés
        """
        test_questions = []
        
        for objective in objectives:
            if objective not in self.test_questions:
                continue
            
            # Prendre des questions de différents niveaux
            for level in range(1, min(max_level_per_objective + 1, 
                                    len(self.test_questions[objective]) + 1)):
                if level in self.test_questions[objective]:
                    available_questions = self.test_questions[objective][level]
                    # Prendre jusqu'à questions_per_objective questions par niveau
                    for i, question in enumerate(available_questions[:questions_per_objective]):
                        test_questions.append(question)
        
        return test_questions
    
    def evaluate_test(self, questions: List[LevelTestQuestion], 
                     responses: List[LevelTestResponse]) -> LevelTestResult:
        """
        Évalue le test et détermine le niveau recommandé
        """
        # Créer un mapping des réponses
        response_map = {resp.question_id: resp.selected_answer for resp in responses}
        
        correct_count = 0
        objective_scores = {}
        
        # Évaluer chaque question
        for question in questions:
            if question.objective not in objective_scores:
                objective_scores[question.objective] = {"correct": 0, "total": 0}
            
            objective_scores[question.objective]["total"] += 1
            
            if question.id in response_map:
                if response_map[question.id] == question.correct_answer:
                    correct_count += 1
                    objective_scores[question.objective]["correct"] += 1
        
        # Calculer le score
        total_questions = len(questions)
        score_percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0
        
        # Déterminer le niveau recommandé
        recommended_level = self._calculate_recommended_level(objective_scores, score_percentage)
        
        # Générer le feedback détaillé
        detailed_feedback = self._generate_detailed_feedback(objective_scores, score_percentage)
        
        return LevelTestResult(
            student_id="",  # Sera rempli par l'API
            total_questions=total_questions,
            correct_answers=correct_count,
            score_percentage=round(score_percentage, 1),
            recommended_level=recommended_level,
            objective_scores=objective_scores,
            detailed_feedback=detailed_feedback
        )
    
    def _calculate_recommended_level(self, objective_scores: Dict[str, Dict[str, int]], 
                                   overall_score: float) -> int:
        """
        Calcule le niveau recommandé basé sur les scores par objectif
        """
        if overall_score >= 80:
            return 3  # Niveau avancé
        elif overall_score >= 60:
            return 2  # Niveau intermédiaire
        else:
            return 1  # Niveau débutant
    
    def _generate_detailed_feedback(self, objective_scores: Dict[str, Dict[str, int]], 
                                  overall_score: float) -> str:
        """
        Génère un feedback détaillé basé sur les résultats
        """
        feedback_parts = []
        
        feedback_parts.append(f"Score global: {overall_score:.1f}%")
        
        for objective, scores in objective_scores.items():
            percentage = (scores["correct"] / scores["total"]) * 100
            feedback_parts.append(f"{objective}: {scores['correct']}/{scores['total']} ({percentage:.1f}%)")
        
        if overall_score >= 80:
            feedback_parts.append("Excellent ! Vous maîtrisez bien les concepts de base.")
        elif overall_score >= 60:
            feedback_parts.append("Bon niveau ! Quelques révisions vous aideront à progresser.")
        else:
            feedback_parts.append("Il serait bénéfique de revoir les concepts fondamentaux.")
        
        return " | ".join(feedback_parts)

# Instance globale du générateur de test
test_generator = LevelTestGenerator(math_system)

# Nouveaux endpoints à ajouter dans fastapi_main.py

@app.post("/level-test/generate")
async def generate_level_test(test_config: LevelTestStart):
    """Générer un test de niveau pour les objectifs spécifiés"""
    try:
        questions = test_generator.generate_test(
            objectives=test_config.objectives,
            questions_per_objective=test_config.questions_per_objective,
            max_level_per_objective=test_config.max_level_per_objective
        )
        
        if not questions:
            raise HTTPException(status_code=404, detail="Aucune question trouvée pour les objectifs spécifiés")
        
        return {
            "test_id": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "questions": questions,
            "total_questions": len(questions),
            "estimated_duration": len(questions) * 2  # 2 minutes par question
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur génération test: {str(e)}")

@app.post("/level-test/submit", response_model=LevelTestResult)
async def submit_level_test(submission: LevelTestSubmission):
    """Soumettre les réponses du test de niveau et obtenir l'évaluation"""
    try:
        # Récupérer les questions du test (dans un vrai système, on les stockerait)
        # Pour cette démonstration, on génère toutes les questions disponibles
        all_questions = []
        for objective in test_generator.test_questions:
            for level in test_generator.test_questions[objective]:
                all_questions.extend(test_generator.test_questions[objective][level])
        
        # Filtrer les questions qui correspondent aux réponses
        test_questions = [q for q in all_questions if any(r.question_id == q.id for r in submission.responses)]
        
        if not test_questions:
            raise HTTPException(status_code=404, detail="Questions du test non trouvées")
        
        # Évaluer le test
        result = test_generator.evaluate_test(test_questions, submission.responses)
        result.student_id = submission.student_id
        
        # Mettre à jour le niveau de l'étudiant
        math_system.set_current_student(submission.student_id)
        if math_system.current_student:
            math_system.current_student.level = result.recommended_level
            # Définir le premier objectif si pas encore défini
            if not math_system.current_student.current_objective and math_system.learning_objectives.objectives_order:
                math_system.current_student.current_objective = math_system.learning_objectives.objectives_order[0]
            math_system.student_manager.save_student(math_system.current_student)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur soumission test: {str(e)}")

@app.get("/level-test/objectives")
async def get_available_test_objectives():
    """Récupérer les objectifs disponibles pour le test de niveau"""
    try:
        available_objectives = list(test_generator.test_questions.keys())
        return {
            "objectives": available_objectives,
            "total_objectives": len(available_objectives),
            "description": "Objectifs disponibles pour le test de niveau"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur récupération objectifs test: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage de l'application"""
    print("🚀 Math Tutoring System API démarrée")

@app.get("/")
async def root():
    """Point d'entrée principal de l'API"""
    return {
        "message": "Math Tutoring System API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {"status": "healthy", "llm_available": math_system.llm is not None}

# ===================== GESTION DES ÉTUDIANTS =====================

@app.post("/students/", response_model=StudentResponse)
async def create_student(student_data: StudentCreate):
    """Créer un nouveau profil étudiant avec session automatique"""
    try:
        student_id = datetime.now().strftime("%Y%m%d%H%M%S%f")[:16]
        
        # Utiliser le SessionManager au lieu de l'ancien StudentManager
        math_system.set_current_student(student_id, student_data.name)
        
        # Définir le premier objectif si disponible
        if math_system.learning_objectives.objectives_order:
            math_system.current_student.current_objective = math_system.learning_objectives.objectives_order[0]
            # Sauvegarde automatique via SessionManager
            math_system.save_current_student()
        
        return StudentResponse(
            student_id=student_id,
            name=math_system.current_student.name,
            level=math_system.current_student.level,
            current_objective=math_system.current_student.current_objective,
            objectives_completed=math_system.current_student.objectives_completed
        )
    except Exception as e:
        print(f"Erreur création étudiant: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur création étudiant: {str(e)}")
@app.get("/students/{student_id}/detailed", response_model=dict)
async def get_detailed_student_info(student_id: str):
    """Récupérer les informations détaillées d'un étudiant avec stats de session"""
    try:
        math_system.set_current_student(student_id)
        
        if not math_system.current_student:
            raise HTTPException(status_code=404, detail="Étudiant non trouvé")
        
        # Récupérer les infos de session
        session_info = math_system.session_manager.get_sessions_info()
        current_session = session_info.get("sessions", {}).get(student_id, {})
        
        return {
            "student": {
                "student_id": math_system.current_student.student_id,
                "name": math_system.current_student.name,
                "level": math_system.current_student.level,
                "current_objective": math_system.current_student.current_objective,
                "objectives_completed": math_system.current_student.objectives_completed,
                "created_at": math_system.current_student.created_at,
                "exercises_completed": len(math_system.current_student.learning_history),
                "recent_success_rate": math_system._calculate_recent_success_rate()
            },
            "session": {
                "is_active": student_id in session_info.get("sessions", {}),
                "last_activity": current_session.get("last_activity"),
                "expires_at": current_session.get("expires_at"),
                "has_memory": current_session.get("has_memory", False)
            },
            "progress": math_system.get_progression_status() if math_system.current_student else {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur récupération infos détaillées: {str(e)}")


@app.get("/students/{student_id}", response_model=StudentResponse)
async def get_student(student_id: str):
    """Récupérer les informations d'un étudiant"""
    try:
        student = math_system.student_manager.load_student(student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Étudiant non trouvé")
        
        return StudentResponse(
            student_id=student.student_id,
            name=student.name,
            level=student.level,
            current_objective=student.current_objective,
            objectives_completed=student.objectives_completed
        )
    except Exception as e:
        print(f"Erreur récupération étudiant: {str(e)}")  # Log pour debug
        raise HTTPException(status_code=500, detail=f"Erreur récupération étudiant: {str(e)}")

@app.get("/students/{student_id}/progress", response_model=ProgressResponse)
async def get_student_progress(student_id: str):
    """Récupérer la progression d'un étudiant"""
    try:
        math_system.set_current_student(student_id)
        progress = math_system.get_student_progress()
        
        if not progress:
            raise HTTPException(status_code=404, detail="Progression non trouvée")
        
        return ProgressResponse(
            level=progress["level"],
            completed=progress["completed"],
            current_objective=math_system.current_student.current_objective if math_system.current_student else None,
            objectives_completed=math_system.current_student.objectives_completed if math_system.current_student else []
        )
    except Exception as e:
        print(f"Erreur récupération progression: {str(e)}")  # Log pour debug
        raise HTTPException(status_code=500, detail=f"Erreur récupération progression: {str(e)}")

# ===================== GESTION DES EXERCICES =====================

@app.post("/exercises/generate", response_model=Exercise)
async def generate_exercise(request: ExerciseRequest):
    """Générer un nouvel exercice pour un étudiant"""
    try:
        math_system.set_current_student(request.student_id)
        exercise = math_system.generate_exercise()
        
        if not exercise:
            raise HTTPException(status_code=500, detail="Impossible de générer un exercice")
        
        return exercise
    except Exception as e:
        print(f"Erreur génération exercice: {str(e)}")  # Log pour debug
        raise HTTPException(status_code=500, detail=f"Erreur génération exercice: {str(e)}")

@app.post("/exercises/similar", response_model=Exercise)
async def generate_similar_exercise(original_exercise: Exercise):
    """Générer un exercice similaire"""
    try:
        similar_exercise = math_system.generate_similar_exercise(original_exercise)
        
        if not similar_exercise:
            raise HTTPException(status_code=500, detail="Impossible de générer un exercice similaire")
        
        return similar_exercise
    except Exception as e:
        print(f"Erreur génération exercice similaire: {str(e)}")  # Log pour debug
        raise HTTPException(status_code=500, detail=f"Erreur génération exercice similaire: {str(e)}")

@app.post("/exercises/evaluate", response_model=dict)  # Changer le type de retour
async def evaluate_answer_with_progression(submission: AnswerSubmission):
    """Évaluer la réponse d'un étudiant avec vérification automatique de progression"""
    try:
        math_system.set_current_student(submission.student_id)
        evaluation = math_system.evaluate_response(submission.exercise, submission.answer)
        
        # Sauvegarder dans l'historique
        try:
            if math_system.current_student:
                if not hasattr(math_system.current_student, 'learning_history'):
                    math_system.current_student.learning_history = []
                
                history_item = {
                    "exercise": submission.exercise.exercise,
                    "answer": submission.answer,
                    "evaluation": evaluation.is_correct,
                    "timestamp": datetime.now().isoformat()
                }
                math_system.current_student.learning_history.append(history_item)
                math_system.student_manager.save_student(math_system.current_student)
        except Exception as history_error:
            print(f"Erreur sauvegarde historique: {str(history_error)}")
        
        # Vérifier la progression automatiquement
        progression_result = math_system.auto_check_and_advance()
        
        return {
            "evaluation": evaluation.dict(),
            "progression": progression_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur évaluation avec progression: {str(e)}")

# ===================== COACH PERSONNEL =====================

@app.get("/coach/message", response_model=CoachPersonal)
async def get_coach_message():
    """Obtenir un message du coach personnel"""
    try:
        coach_message = math_system.get_personal_coach_message()
        
        if not coach_message:
            raise HTTPException(status_code=500, detail="Impossible de générer un message du coach")
        
        return coach_message
    except Exception as e:
        print(f"Erreur message coach: {str(e)}")  # Log pour debug
        raise HTTPException(status_code=500, detail=f"Erreur message coach: {str(e)}")

# ===================== GESTION DES FICHIERS =====================

@app.post("/exercises/evaluate-file")
async def evaluate_file_answer(
    student_id: str,
    exercise: str,  # JSON string de l'exercice
    file: UploadFile = File(...)
):
    """Évaluer une réponse depuis un fichier uploadé"""
    try:
        import json
        
        # Parser l'exercice depuis JSON
        exercise_obj = Exercise.parse_raw(exercise)
        
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            math_system.set_current_student(student_id)
            evaluation = math_system.evaluate_response(exercise_obj, temp_path)
            
            # Sauvegarde dans l'historique avec gestion d'erreur
            try:
                if math_system.current_student:
                    if not hasattr(math_system.current_student, 'learning_history'):
                        math_system.current_student.learning_history = []
                    
                    history_item = {
                        "exercise": exercise_obj.exercise,
                        "answer": f"Fichier: {file.filename}",
                        "evaluation": evaluation.is_correct,
                        "timestamp": datetime.now().isoformat()
                    }
                    math_system.current_student.learning_history.append(history_item)
                    math_system.student_manager.save_student(math_system.current_student)
            except Exception as history_error:
                print(f"Erreur sauvegarde historique fichier: {str(history_error)}")
            
            return evaluation
        finally:
            # Nettoyer le fichier temporaire
            try:
                os.unlink(temp_path)
            except:
                pass  # Ignore les erreurs de nettoyage
    
    except Exception as e:
        print(f"Erreur évaluation fichier: {str(e)}")  # Log pour debug
        raise HTTPException(status_code=500, detail=f"Erreur évaluation fichier: {str(e)}")

# ===================== OBJECTIFS D'APPRENTISSAGE =====================

@app.get("/objectives")
async def get_learning_objectives():
    """Récupérer tous les objectifs d'apprentissage disponibles"""
    try:
        return {
            "objectives": math_system.learning_objectives.objectives,
            "order": math_system.learning_objectives.objectives_order
        }
    except Exception as e:
        print(f"Erreur récupération objectifs: {str(e)}")  # Log pour debug
        raise HTTPException(status_code=500, detail=f"Erreur récupération objectifs: {str(e)}")

@app.get("/objectives/{student_id}/current")
async def get_current_objective_info(student_id: str):
    """Récupérer les informations sur l'objectif actuel d'un étudiant"""
    try:
        math_system.set_current_student(student_id)
        objective_info = math_system.get_current_objective_info()
        
        if not objective_info:
            raise HTTPException(status_code=404, detail="Informations d'objectif non trouvées")
        
        return objective_info
    except Exception as e:
        print(f"Erreur récupération objectif actuel: {str(e)}")  # Log pour debug
        raise HTTPException(status_code=500, detail=f"Erreur récupération objectif actuel: {str(e)}")

# ===================== UTILITAIRES =====================

@app.get("/stats")
async def get_system_stats():
    """Récupérer les statistiques du système"""
    try:
        # Compter les étudiants avec gestion d'erreur
        students_count = 0
        try:
            if hasattr(math_system.student_manager, 'list_students'):
                students_count = len(math_system.student_manager.list_students())
        except:
            pass  # Ignore l'erreur si la méthode n'existe pas
        
        return {
            "students_count": students_count,
            "objectives_available": len(math_system.learning_objectives.objectives),
            "llm_status": "online" if math_system.llm else "offline",
            "system_version": "1.0.0"
        }
    except Exception as e:
        print(f"Erreur récupération statistiques: {str(e)}")  # Log pour debug
        raise HTTPException(status_code=500, detail=f"Erreur récupération statistiques: {str(e)}")

# ===================== GESTION D'ERREURS GLOBALE =====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestionnaire d'erreurs global pour capturer toutes les exceptions non gérées"""
    print(f"Erreur non gérée: {str(exc)}")
    import traceback
    traceback.print_exc()
    return HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.get("/students/{student_id}/progression-status", response_model=ProgressionStatus)
async def get_progression_status(student_id: str):
    """Récupérer le statut de progression détaillé d'un étudiant"""
    try:
        math_system.set_current_student(student_id)
        status = math_system.get_progression_status()
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        return ProgressionStatus(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur récupération statut: {str(e)}")

@app.post("/students/{student_id}/advance-objective", response_model=ProgressionResult)
async def advance_student_objective(student_id: str):
    """Faire progresser manuellement un étudiant vers l'objectif suivant"""
    try:
        math_system.set_current_student(student_id)
        
        # Vérifier si l'étudiant peut progresser
        if not math_system.check_objective_completion():
            return ProgressionResult(
                progression_occurred=False,
                message="L'étudiant n'a pas encore rempli les critères pour passer à l'objectif suivant."
            )
        
        # Faire progresser
        if math_system.advance_to_next_objective():
            return ProgressionResult(
                progression_occurred=True,
                message="Progression vers l'objectif suivant réussie !",
                new_objective=math_system.current_student.current_objective,
                new_level=math_system.current_student.level
            )
        else:
            return ProgressionResult(
                progression_occurred=False,
                message="Tous les objectifs ont été complétés !"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur progression: {str(e)}")

@app.get("/students/{student_id}/check-completion")
async def check_objective_completion(student_id: str):
    """Vérifier si un étudiant peut passer à l'objectif suivant"""
    try:
        math_system.set_current_student(student_id)
        can_advance = math_system.check_objective_completion()
        
        return {
            "can_advance": can_advance,
            "recent_success_rate": math_system._calculate_recent_success_rate(),
            "exercises_completed": len(math_system.current_student.learning_history) if math_system.current_student else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur vérification completion: {str(e)}")
@app.post("/exercises/evaluate-with-coaching", response_model=dict)
async def evaluate_answer_with_personalized_coaching(submission: AnswerSubmission):
    """Évaluer la réponse avec coaching personnalisé et gestion de session optimisée"""
    try:
        # La méthode utilise maintenant automatiquement le SessionManager
        result = math_system.evaluate_answer_for_api_with_coaching(
            exercise_data=submission.exercise.dict(),
            answer=submission.answer,
            student_id=submission.student_id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur évaluation avec coaching: {str(e)}")

# Endpoint pour obtenir un coaching personnalisé après coup
@app.post("/coach/personalized-message")
async def get_personalized_coaching_message(
    exercise: Exercise,
    student_answer: str,
    student_id: str,
    evaluation: Optional[dict] = None
):
    """Obtenir un message de coaching personnalisé basé sur un exercice et une réponse spécifiques"""
    try:
        math_system.set_current_student(student_id)
        
        # Convertir l'évaluation si fournie
        eval_result = None
        if evaluation:
            eval_result = EvaluationResult(**evaluation)
        
        coaching = math_system.get_personalized_coach_message(
            exercise=exercise,
            student_answer=student_answer,
            evaluation=eval_result
        )
        
        if not coaching:
            raise HTTPException(status_code=500, detail="Impossible de générer un coaching personnalisé")
        
        return coaching
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur coaching personnalisé: {str(e)}")
# Fonction de nettoyage pour FastAPI
def cleanup_on_shutdown():
    """Nettoyage lors de l'arrêt du serveur"""
    math_system.shutdown()

# Enregistrer le nettoyage
atexit.register(cleanup_on_shutdown)

# Ajouter ces nouveaux endpoints pour la gestion des sessions
@app.get("/sessions/stats")
async def get_sessions_stats():
    """Récupérer les statistiques des sessions actives"""
    try:
        return math_system.get_session_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur récupération stats sessions: {str(e)}")

@app.post("/sessions/cleanup")
async def cleanup_expired_sessions():
    """Nettoyer manuellement les sessions expirées"""
    try:
        cleaned_count = math_system.cleanup_expired_sessions()
        return {
            "cleaned_sessions": cleaned_count,
            "active_sessions": math_system.session_manager.get_active_sessions_count(),
            "message": f"{cleaned_count} sessions expirées nettoyées"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur nettoyage sessions: {str(e)}")

@app.delete("/sessions/{student_id}")
async def close_student_session(student_id: str):
    """Fermer explicitement la session d'un étudiant"""
    try:
        if math_system.session_manager.close_session(student_id):
            return {"message": f"Session fermée pour {student_id}"}
        else:
            raise HTTPException(status_code=404, detail="Session non trouvée")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur fermeture session: {str(e)}")

@app.post("/sessions/save-all")
async def save_all_sessions():
    """Forcer la sauvegarde de toutes les sessions actives"""
    try:
        saved_count = math_system.session_manager.force_save_all_sessions()
        return {
            "saved_sessions": saved_count,
            "message": f"{saved_count} sessions sauvegardées"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur sauvegarde sessions: {str(e)}")
    
@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage avec gestionnaire de sessions"""
    print("🚀 Math Tutoring System API démarrée avec SessionManager")
    print(f"📊 Sessions timeout: {math_system.session_manager.session_timeout_minutes} minutes")
    print(f"🧹 Nettoyage automatique: {math_system.session_manager.cleanup_interval_minutes} minutes")

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage à l'arrêt du serveur"""
    print("🛑 Arrêt du serveur - Sauvegarde des sessions...")
    math_system.shutdown()
    print("✅ Arrêt propre terminé")

# Middleware pour mettre à jour automatiquement l'activité des sessions
@app.middleware("http")
async def session_activity_middleware(request, call_next):
    """Middleware pour tracker l'activité des sessions automatiquement"""
    response = await call_next(request)
    
    # Extraire student_id de l'URL si présent
    import re
    student_id_match = re.search(r"/students/([^/]+)", str(request.url))
    if student_id_match:
        student_id = student_id_match.group(1)
        # Marquer l'activité de la session
        session = math_system.session_manager._active_sessions.get(student_id)
        if session:
            session.update_activity()
    
    return response

# Nouvel endpoint pour gérer les données de session temporaires
@app.post("/sessions/{student_id}/data")
async def update_session_data(student_id: str, data: dict):
    """Mettre à jour les données temporaires de session"""
    try:
        updated_count = 0
        for key, value in data.items():
            if math_system.session_manager.update_session_data(student_id, key, value):
                updated_count += 1
        
        return {
            "updated_fields": updated_count,
            "student_id": student_id,
            "message": f"{updated_count} champs mis à jour dans la session"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur mise à jour données session: {str(e)}")

@app.get("/sessions/{student_id}/data/{key}")
async def get_session_data(student_id: str, key: str):
    """Récupérer une donnée temporaire de session"""
    try:
        value = math_system.session_manager.get_session_data(student_id, key)
        if value is None:
            raise HTTPException(status_code=404, detail=f"Clé '{key}' non trouvée dans la session")
        
        return {
            "student_id": student_id,
            "key": key,
            "value": value
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur récupération donnée session: {str(e)}")

# Endpoint de santé amélioré avec infos sessions
@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API avec infos sessions"""
    try:
        sessions_info = math_system.get_session_stats()
        return {
            "status": "healthy",
            "llm_available": math_system.llm is not None,
            "active_sessions": sessions_info["active_sessions"],
            "session_manager_status": "running",
            "chroma_db_status": "connected" if math_system.session_manager.chroma_client else "disconnected"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "llm_available": math_system.llm is not None,
            "active_sessions": 0
        }

# Endpoint pour diagnostics avancés
@app.get("/diagnostics")
async def system_diagnostics():
    """Diagnostics détaillés du système"""
    try:
        sessions_info = math_system.get_session_stats()
        
        return {
            "system_status": "operational",
            "sessions": {
                "active_count": sessions_info["active_sessions"],
                "timeout_minutes": math_system.session_manager.session_timeout_minutes,
                "cleanup_interval_minutes": math_system.session_manager.cleanup_interval_minutes,
                "sessions_details": sessions_info["sessions"]
            },
            "storage": {
                "data_directory": str(math_system.session_manager.data_dir),
                "chroma_db_available": math_system.session_manager.chroma_client is not None
            },
            "ai_services": {
                "llm_model": math_system.model_name,
                "llm_available": math_system.llm is not None,
                "agents_initialized": {
                    "exercise_creator": math_system.exercise_creator_agent is not None,
                    "evaluator": math_system.evaluator_agent is not None,
                    "personal_coach": math_system.personal_coach_agent is not None
                }
            },
            "learning_objectives": {
                "total_objectives": len(math_system.learning_objectives.objectives),
                "objectives_order_count": len(math_system.learning_objectives.objectives_order)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur diagnostics: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "fastapi_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )