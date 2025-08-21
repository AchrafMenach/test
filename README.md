# Projet Math Tutor - Architecture CrewAI

Ce projet a été restructuré pour adopter une architecture scalable et modulaire, inspirée par le framework CrewAI. L'objectif est de fournir un système de tutorat en mathématiques intelligent, capable de générer des exercices, d'évaluer les réponses des étudiants et de fournir un accompagnement personnalisé.

## Table des matières
- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Composants Clés](#composants-clés)
  - [Modèles Pydantic](#modèles-pydantic)
  - [Outils (Tools)](#outils-tools)
  - [Agents](#agents)
  - [Tâches (Tasks)](#tâches-tasks)
  - [Gestionnaire d'Étudiants (StudentManager)](#gestionnaire-détudiants-studentmanager)
  - [Système Principal (MathTutoringSystem)](#système-principal-mathtutoringsystem)
- [Flux de Travail (Crew)](#flux-de-travail-crew)

## Structure du Projet
Le projet est organisé de manière logique pour faciliter la compréhension et la maintenance:

```
math_tutor_crewai/
├── src/
│   ├── agents/                 # Définitions des agents CrewAI
│   │   ├── exercise_creator_agent.py
│   │   ├── evaluator_agent.py
│   │   └── personal_coach_agent.py
│   ├── tasks/                  # Définitions des tâches CrewAI
│   │   ├── exercise_creation_task.py
│   │   └── evaluation_task.py
│   ├── tools/                  # Implémentations des outils utilisés par les agents
│   │   ├── file_processor.py
│   │   └── long_term_memory.py
│   ├── models/                 # Modèles de données Pydantic
│   │   └── models.py
│   ├── config/                 # Fichiers de configuration et données statiques
│   │   └── learning_objectives.py
        └── objectif.json
│   ├── student_manager.py      # Logique de gestion des profils étudiants
│   └── main_system.py          # Point d'entrée principal et orchestration des Crews
└── README.md                   # Ce fichier de documentation
```

## Installation
1.  **Cloner le dépôt (si applicable) ou décompresser l'archive.**
2.  **Créer un environnement virtuel (recommandé):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Sur Windows: .\venv\Scripts\activate
    ```
3.  **Installer les dépendances:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Un fichier `requirements.txt` devrait être créé avec les dépendances nécessaires comme `crewai`, `langchain-groq`, `pydantic`, `chromadb`, `pandas`, `rich`, `python-dotenv`.)*
4.  **Configuration de l'API Groq:**
    Créez un fichier `.env` à la racine du projet et ajoutez votre clé API Groq:
    ```
    GROQ_API_KEY="votre_clé_api_groq_ici"
    ```

## Utilisation
Pour exécuter le système de tutorat, vous pouvez lancer le fichier `main_system.py`:

```bash
python3 src/main_system.py
```

Le script `main_system.py` contient un exemple d'utilisation qui crée un étudiant, génère un exercice, évalue une réponse simulée et affiche des messages du coach personnel.

## Composants Clés

### Modèles Pydantic
Situés dans `src/models/models.py`, ces modèles définissent la structure des données utilisées à travers l'application, assurant la validation et la clarté des données.
-   `StudentProfile`: Profil d'un étudiant.
-   `Exercise`: Structure d'un exercice de mathématiques.
-   `EvaluationResult`: Résultat de l'évaluation d'une réponse d'étudiant.
-   `CoachPersonal`: Messages personnalisés du coach.

### Outils (Tools)
Situés dans `src/tools/`, ces modules fournissent des fonctionnalités spécifiques qui peuvent être utilisées par les agents.
-   `file_processor.py`: Contient la classe `FileProcessor` pour l'extraction de texte à partir de différents types de fichiers (PDF, images).
-   `long_term_memory.py`: Contient la classe `LongTermMemory` qui interagit avec ChromaDB pour la persistance des données et la gestion de la mémoire à long terme.

### Agents
Définis dans `src/agents/`, chaque agent a un rôle, un objectif et un historique (`backstory`) spécifiques, le rendant expert dans son domaine.
-   `ExerciseCreatorAgent`: Agent spécialisé dans la création d'exercices de mathématiques adaptés au niveau de l'étudiant.
-   `EvaluatorAgent`: Agent expert en évaluation des réponses mathématiques, capable d'identifier les erreurs et de fournir des retours pédagogiques.
-   `PersonalCoachAgent`: Agent dédié à l'accompagnement personnalisé, à la motivation et aux stratégies d'apprentissage.

### Tâches (Tasks)
Définies dans `src/tasks/`, les tâches représentent les actions spécifiques que les agents doivent accomplir. Elles encapsulent la logique métier et définissent les entrées/sorties attendues.
-   `ExerciseCreationTask`: Tâche pour la génération d'un exercice de mathématiques.
-   `EvaluationTask`: Tâche pour l'évaluation de la réponse d'un étudiant.

### Gestionnaire d'Étudiants (StudentManager)
Situé dans `src/student_manager.py`, cette classe gère la création, le chargement et la sauvegarde des profils étudiants. Elle intègre également la synchronisation avec la mémoire à long terme (ChromaDB).

### Système Principal (MathTutoringSystem)
Le fichier `src/main_system.py` contient la classe `MathTutoringSystem`, qui est le cœur de l'application. Elle est responsable de:
-   L'initialisation du modèle de langage (LLM).
-   L'initialisation des agents et des tâches.
-   La gestion du cycle de vie des étudiants.
-   L'orchestration des flux de travail en utilisant les Crews de CrewAI pour générer des exercices et évaluer les réponses.
-   L'intégration des objectifs d'apprentissage et de la mémoire à long terme.

## Flux de Travail (Crew)
Le système utilise CrewAI pour orchestrer les interactions entre les agents et les tâches. Par exemple, pour générer un exercice, une `Crew` est formée avec l'agent `ExerciseCreator` et la tâche `ExerciseCreationTask`. De même, pour l'évaluation, l'agent `Evaluator` est associé à la tâche `EvaluationTask`.

Cette approche permet de définir des flux de travail complexes de manière claire et maintenable, en tirant parti des capacités collaboratives des agents. Chaque agent se concentre sur son domaine d'expertise, contribuant à un système global cohérent et performant.


