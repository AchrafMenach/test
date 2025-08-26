from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Union, List
import uvicorn
from pathlib import Path
import tempfile
import os
from datetime import datetime  # Import manquant ajouté

# Importez votre système existant
from src.main_system import MathTutoringSystem
from src.models.models import Exercise, EvaluationResult, StudentProfile, CoachPersonal

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
    """Créer un nouveau profil étudiant"""
    try:
        student = math_system.student_manager.create_student(name=student_data.name)
        math_system.set_current_student(student.student_id)
        
        # Définir le premier objectif si disponible
        if math_system.learning_objectives.objectives_order:
            student.current_objective = math_system.learning_objectives.objectives_order[0]
            math_system.student_manager.save_student(student)
        
        return StudentResponse(
            student_id=student.student_id,
            name=student.name,
            level=student.level,
            current_objective=student.current_objective,
            objectives_completed=student.objectives_completed
        )
    except Exception as e:
        print(f"Erreur création étudiant: {str(e)}")  # Log pour debug
        raise HTTPException(status_code=500, detail=f"Erreur création étudiant: {str(e)}")

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

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )