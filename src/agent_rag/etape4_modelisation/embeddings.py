"""
embeddings.py — Transformer un texte en vecteur.

Un modèle d'embedding transforme un texte en une liste de nombres (384 ici).
Deux textes au SENS proche donnent deux vecteurs proches, même sans mot commun :

    "hausse des prix"   -> [0.12, -0.40, 0.33, ...]  \\
                                                       } proches
    "l'inflation monte" -> [0.10, -0.38, 0.35, ...]  /

C'est la RECHERCHE SÉMANTIQUE (par le sens), opposée à la recherche par mots-clés.
"""

import numpy as np

from agent_rag import config

_modele = None   # chargé une seule fois (quelques secondes), puis réutilisé


def charger_modele():
    """Charge (et télécharge au premier lancement, ~470 Mo) le modèle d'embedding."""
    global _modele
    if _modele is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Chargement du modèle d'embedding : {config.MODELE_EMBEDDING}")
        _modele = SentenceTransformer(config.MODELE_EMBEDDING)
    return _modele


def vectoriser(textes: list[str]) -> np.ndarray:
    """
    Liste de textes -> matrice de forme (nombre de textes, 384).

    normalize_embeddings=True : chaque vecteur a une longueur 1. Alors la
    similarité cosinus  cos(A, B) = A·B / (||A||·||B||)  devient simplement A·B.
    """
    return charger_modele().encode(
        textes,
        normalize_embeddings=True,
        show_progress_bar=len(textes) > 50,
    )
