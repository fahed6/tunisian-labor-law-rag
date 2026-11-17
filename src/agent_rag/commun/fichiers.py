"""
fichiers.py — Lecture et écriture des fichiers JSON du projet.

Chaque étape CRISP-DM lit le fichier produit par l'étape précédente et
écrit le sien. Ces deux fonctions sont le seul endroit qui touche au disque
pour les JSON : le format (UTF-8, accents conservés, indentation) est
donc identique partout.
"""

import json
from pathlib import Path
from typing import Any


def lire_json(chemin: Path, defaut: Any = None) -> Any:
    """Lit un fichier JSON. Renvoie `defaut` s'il n'existe pas encore."""
    if not chemin.exists():
        return defaut
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def ecrire_json(donnees: Any, chemin: Path) -> None:
    """Écrit des données en JSON (crée le dossier si besoin)."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(donnees, f, ensure_ascii=False, indent=2)


def exiger(chemin: Path, commande: str) -> None:
    """Arrête le programme avec un message clair si un fichier attendu manque."""
    if not chemin.exists():
        raise SystemExit(
            f"Fichier manquant : {chemin}\n"
            f"Lancez d'abord l'étape précédente :  python main.py {commande}"
        )
