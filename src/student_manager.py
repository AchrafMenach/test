import json
import shutil
from datetime import datetime, time
from pathlib import Path
from typing import Optional, Dict, List
import chromadb
from rich.console import Console

from src.models.models import StudentProfile
from src.tools.long_term_memory import LongTermMemory

console = Console()

class StudentManager:
    def __init__(self, data_dir="students_data", enable_memory: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # ✅ DB unique à chaque exécution
        unique_db_path = self.data_dir / f"memory_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.client = chromadb.PersistentClient(path=str(unique_db_path))

        self.long_term_memory = self._initialize_memory(enable_memory)

    def _initialize_memory(self, enable_memory):
        if not enable_memory:
            return None

        try:
            if hasattr(self, 'client'):
                try:
                    self.client.reset()
                except:
                    pass

            memory = LongTermMemory("global_memory", client=self.client)

            if not memory.test_connection():
                raise ConnectionError("Échec test connexion mémoire")

            return memory
        except Exception as e:
            console.print(f"⚠️ Initialisation mémoire échouée : {str(e)}")
            return None

    def create_student(self, name=None):
        student_id = datetime.now().strftime("%Y%m%d%H%M%S%f")[:16]
        profile = StudentProfile( 
            student_id=student_id,
            name=name,
            last_session=datetime.now().isoformat()
        )
        self.save_student(profile)
        return profile

    def load_student(self, student_id):
        student_file = self.data_dir / f"{student_id}.json"
        if not student_file.exists():
            return None
        try:
            with open(student_file, 'r', encoding='utf-8') as f:
                return StudentProfile(**json.load(f))
        except Exception as e:
            console.print(f"Erreur de chargement: {str(e)}")
            return None

    def save_student(self, student):
        student_file = self.data_dir / f"{student.student_id}.json"
        try:
            with open(student_file, 'w', encoding='utf-8') as f:
                json.dump(student.model_dump(), f, indent=4)

            self._sync_to_long_term_memory(student)
        except Exception as e:
            console.print(f"Erreur de sauvegarde: {str(e)}")

    def _sync_to_long_term_memory(self, student: StudentProfile) -> None:
        if not self.long_term_memory:
            if not hasattr(self, '_warned_memory'):
                console.print("ℹ️ Mémoire désactivée - mode dégradé activé")
                self._warned_memory = True
            return

        metadata = {
            "type": "level_update",
            "student_id": student.student_id,
            "new_level": str(student.level),
            "timestamp": datetime.now().isoformat()
        }

        for attempt in range(3):
            try:
                self.long_term_memory.upsert_memory(
                    content=f"Niveau {student.level} atteint par {student.name or 'anonyme'}",
                    metadata=metadata,
                    id=f"student_{student.student_id}_level_{student.level}"
                )
                return
            except Exception as e:
                if attempt == 2:
                    self._handle_sync_error(e, student)
                else:
                    print(f"⚠️ Tentative {attempt + 1} échouée, nouvelle tentative...")
                    time.sleep(1)

    def _handle_sync_error(self, error: Exception, student: StudentProfile):
        error_msg = f"❌ Erreur synchronisation mémoire: {str(error)}"
        console.print(error_msg)

        backup_dir = self.data_dir / "backups"
        try:
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f"{timestamp}_{student.student_id}.json"

            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "student": student.model_dump(),
                    "error": str(error),
                    "timestamp": datetime.now().isoformat(),
                    "attempt": "memory_sync"
                }, f, indent=4)

            console.print(f"✅ Sauvegarde secours créée: {backup_file}")
        except Exception as backup_error:
            console.print(f"❌ Échec sauvegarde secours: {str(backup_error)}")
