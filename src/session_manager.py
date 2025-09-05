import time
import threading
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import chromadb
from rich.console import Console

from src.models.models import StudentProfile
from src.tools.long_term_memory import LongTermMemory

console = Console()

@dataclass
class StudentSession:
    """Représente une session active d'un étudiant"""
    student_profile: StudentProfile
    last_activity: datetime
    memory: Optional[LongTermMemory] = None
    session_data: Dict = None
    
    def __post_init__(self):
        if self.session_data is None:
            self.session_data = {}
    
    def update_activity(self):
        """Met à jour l'horodatage de dernière activité"""
        self.last_activity = datetime.now()
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Vérifie si la session a expiré"""
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)

class SessionManager:
    """
    Gestionnaire de sessions pour les étudiants
    Maintient les sessions actives en mémoire et gère leur cycle de vie
    """
    
    def __init__(self, data_dir="students_data", session_timeout_minutes=30, cleanup_interval_minutes=10):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Sessions actives : {student_id: StudentSession}
        self._active_sessions: Dict[str, StudentSession] = {}
        self._session_lock = threading.RLock()
        
        # Configuration
        self.session_timeout_minutes = session_timeout_minutes
        self.cleanup_interval_minutes = cleanup_interval_minutes
        
        # ChromaDB client partagé et persistant
        self.chroma_client = self._initialize_persistent_chroma()
        
        # Démarrer le nettoyage automatique des sessions expirées
        self._start_cleanup_thread()
        
        console.print(f"✅ SessionManager initialisé (timeout: {session_timeout_minutes}min)")
    
    def _initialize_persistent_chroma(self):
        """Initialise ChromaDB avec un chemin persistant (pas unique)"""
        try:
            persistent_path = self.data_dir / "memory_db"
            return chromadb.PersistentClient(path=str(persistent_path))
        except Exception as e:
            console.print(f"⚠️ Erreur ChromaDB: {str(e)}")
            return None
    
    def _start_cleanup_thread(self):
        """Démarre un thread de nettoyage automatique des sessions expirées"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(self.cleanup_interval_minutes * 60)  # Convertir en secondes
                    self.cleanup_expired_sessions()
                except Exception as e:
                    console.print(f"⚠️ Erreur nettoyage sessions: {str(e)}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def get_or_create_session(self, student_id: str, student_name: Optional[str] = None) -> StudentSession:
        """
        Récupère une session existante ou en crée une nouvelle
        Thread-safe avec verrous
        """
        with self._session_lock:
            # Vérifier si la session existe et est valide
            if student_id in self._active_sessions:
                session = self._active_sessions[student_id]
                if not session.is_expired(self.session_timeout_minutes):
                    session.update_activity()
                    return session
                else:
                    # Session expirée, la nettoyer
                    console.print(f"🔄 Session expirée pour {student_id}, nettoyage...")
                    self._cleanup_session(student_id)
            
            # Créer une nouvelle session
            return self._create_new_session(student_id, student_name)
    
    def _create_new_session(self, student_id: str, student_name: Optional[str] = None) -> StudentSession:
        """Crée une nouvelle session pour un étudiant"""
        try:
            # Charger ou créer le profil étudiant
            student_profile = self._load_or_create_student(student_id, student_name)
            
            # Initialiser la mémoire long terme si ChromaDB disponible
            memory = None
            if self.chroma_client:
                try:
                    memory = LongTermMemory(f"student_{student_id}", client=self.chroma_client)
                    if not memory.test_connection():
                        memory = None
                except Exception as e:
                    console.print(f"⚠️ Erreur mémoire pour {student_id}: {str(e)}")
                    memory = None
            
            # Créer la session
            session = StudentSession(
                student_profile=student_profile,
                last_activity=datetime.now(),
                memory=memory,
                session_data={}
            )
            
            # Ajouter aux sessions actives
            self._active_sessions[student_id] = session
            
            console.print(f"✅ Nouvelle session créée pour {student_profile.name or 'Anonymous'} (ID: {student_id})")
            return session
            
        except Exception as e:
            console.print(f"❌ Erreur création session {student_id}: {str(e)}")
            raise
    
    def _load_or_create_student(self, student_id: str, student_name: Optional[str] = None) -> StudentProfile:
        """Charge un étudiant existant ou en crée un nouveau"""
        student_file = self.data_dir / f"{student_id}.json"
        
        if student_file.exists():
            try:
                import json
                with open(student_file, 'r', encoding='utf-8') as f:
                    return StudentProfile(**json.load(f))
            except Exception as e:
                console.print(f"⚠️ Erreur chargement {student_id}: {str(e)}")
        
        # Créer un nouveau profil si pas trouvé ou erreur
        return StudentProfile(
            student_id=student_id,
            name=student_name,
            last_session=datetime.now().isoformat()
        )
    
    def save_session(self, student_id: str) -> bool:
        """Sauvegarde la session d'un étudiant sur disque"""
        with self._session_lock:
            if student_id not in self._active_sessions:
                console.print(f"⚠️ Session {student_id} non trouvée pour sauvegarde")
                return False
            
            session = self._active_sessions[student_id]
            session.update_activity()
            
            try:
                import json
                student_file = self.data_dir / f"{student_id}.json"
                
                with open(student_file, 'w', encoding='utf-8') as f:
                    json.dump(session.student_profile.model_dump(), f, indent=4)
                
                # Synchroniser avec la mémoire long terme si disponible
                if session.memory:
                    self._sync_to_memory(session)
                
                return True
                
            except Exception as e:
                console.print(f"❌ Erreur sauvegarde session {student_id}: {str(e)}")
                return False
    
    def _sync_to_memory(self, session: StudentSession):
        """Synchronise les données de session avec la mémoire long terme"""
        if not session.memory:
            return
        
        try:
            student = session.student_profile
            
            # Ajouter les objectifs complétés
            for obj in student.objectives_completed:
                session.memory.upsert_memory(
                    content=f"Objectif complété: {obj}",
                    metadata={
                        "type": "achievement",
                        "objective": obj,
                        "timestamp": datetime.now().isoformat()
                    },
                    id=f"achievement_{student.student_id}_{obj}"
                )
            
            # Ajouter l'historique récent (derniers 5 exercices)
            recent_history = student.learning_history[-5:] if student.learning_history else []
            for i, item in enumerate(recent_history):
                session.memory.upsert_memory(
                    content=f"Exercice: {item['exercise']} - Réponse: {item['answer']}",
                    metadata={
                        "type": "exercise",
                        "correct": str(item.get("evaluation", False)),
                        "timestamp": item.get("timestamp", datetime.now().isoformat()),
                        "concept": item.get("concept", "")
                    },
                    id=f"exercise_{student.student_id}_{len(student.learning_history)-5+i}"
                )
            
        except Exception as e:
            console.print(f"⚠️ Erreur sync mémoire: {str(e)}")
    
    def update_session_data(self, student_id: str, key: str, value) -> bool:
        """Met à jour les données temporaires de session"""
        with self._session_lock:
            if student_id in self._active_sessions:
                session = self._active_sessions[student_id]
                session.session_data[key] = value
                session.update_activity()
                return True
            return False
    
    def get_session_data(self, student_id: str, key: str, default=None):
        """Récupère une donnée temporaire de session"""
        with self._session_lock:
            if student_id in self._active_sessions:
                session = self._active_sessions[student_id]
                return session.session_data.get(key, default)
            return default
    
    def cleanup_expired_sessions(self) -> int:
        """Nettoie les sessions expirées et retourne le nombre nettoyé"""
        with self._session_lock:
            expired_sessions = [
                student_id for student_id, session in self._active_sessions.items()
                if session.is_expired(self.session_timeout_minutes)
            ]
            
            for student_id in expired_sessions:
                self._cleanup_session(student_id)
            
            if expired_sessions:
                console.print(f"🧹 {len(expired_sessions)} sessions expirées nettoyées")
            
            return len(expired_sessions)
    
    def _cleanup_session(self, student_id: str):
        """Nettoie une session spécifique"""
        if student_id in self._active_sessions:
            # Sauvegarder avant de nettoyer
            self.save_session(student_id)
            # Supprimer de la mémoire
            del self._active_sessions[student_id]
    
    def get_active_sessions_count(self) -> int:
        """Retourne le nombre de sessions actives"""
        with self._session_lock:
            return len(self._active_sessions)
    
    def get_sessions_info(self) -> Dict:
        """Retourne des informations sur les sessions actives"""
        with self._session_lock:
            return {
                "active_sessions": len(self._active_sessions),
                "sessions": {
                    student_id: {
                        "name": session.student_profile.name,
                        "level": session.student_profile.level,
                        "last_activity": session.last_activity.isoformat(),
                        "expires_at": (session.last_activity + timedelta(minutes=self.session_timeout_minutes)).isoformat(),
                        "has_memory": session.memory is not None
                    }
                    for student_id, session in self._active_sessions.items()
                }
            }
    
    def force_save_all_sessions(self) -> int:
        """Force la sauvegarde de toutes les sessions actives"""
        with self._session_lock:
            saved_count = 0
            for student_id in self._active_sessions.keys():
                if self.save_session(student_id):
                    saved_count += 1
            
            console.print(f"💾 {saved_count} sessions sauvegardées")
            return saved_count
    
    def close_session(self, student_id: str) -> bool:
        """Ferme explicitement une session (avec sauvegarde)"""
        with self._session_lock:
            if student_id in self._active_sessions:
                self.save_session(student_id)
                del self._active_sessions[student_id]
                console.print(f"🔚 Session fermée pour {student_id}")
                return True
            return False
    
    def shutdown(self):
        """Arrêt propre du gestionnaire de sessions"""
        console.print("🛑 Arrêt du SessionManager...")
        self.force_save_all_sessions()
        if self.chroma_client:
            try:
                # ChromaDB ne nécessite pas de fermeture explicite
                pass
            except Exception as e:
                console.print(f"⚠️ Erreur fermeture ChromaDB: {str(e)}")
        console.print("✅ SessionManager arrêté proprement")

# Décorateur pour faciliter l'usage avec les endpoints FastAPI
def with_session(func):
    """
    Décorateur pour injecter automatiquement la session dans les endpoints
    Usage: @with_session puis ajouter session: StudentSession comme premier paramètre
    """
    def wrapper(*args, **kwargs):
        # Récupérer student_id depuis les kwargs
        student_id = kwargs.get('student_id')
        if not student_id:
            raise ValueError("student_id requis pour with_session")
        
        # Récupérer le session_manager depuis l'instance globale (à définir dans FastAPI)
        from fastapi_main import session_manager  # Import spécifique
        
        # Obtenir la session
        session = session_manager.get_or_create_session(student_id)
        
        # Injecter la session comme premier argument
        return func(session, *args, **kwargs)
    
    return wrapper