import os
from pathlib import Path
from rich.console import Console

# Crée une instance globale de Console
console = Console()

class FileProcessor:
    def extract_text_from_file(self, file_path: str) -> str:
        # Affichage avec Rich
        console.print(f"[bold green]Processing file:[/bold green] {file_path}")

        # Exemple de logique selon type de fichier
        if Path(file_path).suffix.lower() == ".pdf":
            return "Extracted text from PDF: This is a dummy text for a math problem."
        elif Path(file_path).suffix.lower() in [".png", ".jpg", ".jpeg"]:
            return "Extracted text from image: This is a dummy text for an image of a math problem."
        else:
            return "Extracted text from unknown file type."
