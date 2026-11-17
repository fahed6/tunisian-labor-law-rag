"""
baseline.py — Un modèle de RÉFÉRENCE volontairement naïf.

Règle d'or : un modèle complexe n'a d'intérêt que s'il fait MIEUX qu'une
méthode simple. Ici : classer les chunks selon le nombre de mots partagés
avec la question (aucune notion de sens, pas de synonymes).
"""

import re


def mots(texte: str) -> set[str]:
    """Mots en minuscules de plus de 3 lettres (on ignore « le », « de », « et »...)."""
    return {m for m in re.findall(r"\w+", texte.lower()) if len(m) > 3}


def recherche_par_mots_cles(question: str, chunks: list[dict], k: int) -> list[dict]:
    mots_question = mots(question)
    scores = [len(mots_question & mots(c["texte"])) for c in chunks]
    ordre = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ordre[:k]]
