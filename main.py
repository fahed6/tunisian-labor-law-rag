"""
main.py — Point d'entrée unique du projet.

    python main.py --help         liste toutes les commandes (une par étape CRISP-DM)

Le code est dans src/agent_rag/ ; ce fichier ne fait que le lancer.
"""

import sys
from pathlib import Path

# Permet de lancer le projet même sans « pip install -e . » :
# on ajoute src/ à la liste des dossiers où Python cherche les modules.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agent_rag.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
