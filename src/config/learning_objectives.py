import json
from pathlib import Path


class LearningObjectives:
    def __init__(self, json_path: str = "objectifs.json"):
        base_dir = Path(__file__).resolve().parent
        self.json_path = base_dir / json_path
        self.objectives = {}
        self.objectives_order = []
        self._load_json()


    def _load_json(self):
        if not self.json_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        structure = data.get("structure", {})

        # Aplatir la hiérarchie en dictionnaire plat {objectif_id: contenu}
        for cycle, cycle_data in structure.items():
            themes = cycle_data.get("themes", {})
            for theme_name, theme_data in themes.items():
                niveaux = theme_data.get("niveaux", {})
                for niveau_id, niveau_data in niveaux.items():
                    objective_id = f"{cycle}::{theme_name}::{niveau_id}"
                    self.objectives[objective_id] = {
                        "cycle": cycle,
                        "theme": theme_name,
                        "description": theme_data.get("description", ""),
                        "level": niveau_id,
                        "level_name": niveau_data.get("name", ""),
                        "objectives": niveau_data.get("objectives", []),
                        "example_exercises": niveau_data.get("example_exercises", []),
                        "example_functions": niveau_data.get("example_functions", [])
                    }
                    self.objectives_order.append(objective_id)
