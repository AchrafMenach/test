import json
from pathlib import Path
from rich.console import Console

# Crée une instance globale de Console
console = Console()
class LearningObjectives:
    def __init__(self, objectives_file=None):
        base_path = Path(__file__).resolve().parent  # dossier du fichier actuel
        if objectives_file is None:
            objectives_file = base_path / "objectifs.json"
        self.objectives_file = Path(objectives_file)
        self._load_objectives()



    def _load_objectives(self):
        try:
            with open(self.objectives_file, 'r', encoding='utf-8') as f:
                self.objectives = json.load(f)
                self.objectives_order = list(self.objectives.keys())
        except Exception as e:
            console.print(f"Erreur de chargement des objectifs: {str(e)}")
            self.objectives = {}
            self.objectives_order = []


