"""
eval_recherche.py — A. Évaluer la RECHERCHE (le « R » de RAG).

C'est le maillon le plus important : si la recherche ne trouve pas le bon
article, le LLM ne peut pas bien répondre, quelle que soit sa qualité.
"""

from collections.abc import Callable

from agent_rag.etape5_evaluation import metriques

FonctionRecherche = Callable[[str, int], list[dict]]


def evaluer_recherche(rechercher: FonctionRecherche, jeu_de_test: list[dict], k: int) -> dict:
    """Calcule Hit@1, Hit@k et MRR pour une fonction de recherche."""
    rangs = [
        metriques.rang(rechercher(ex["question"], k), ex["id_article_attendu"])
        for ex in jeu_de_test
    ]
    return {
        "Hit@1": round(metriques.hit_at(rangs, 1), 3),
        f"Hit@{k}": round(metriques.hit_at(rangs, k), 3),
        "MRR": round(metriques.mrr(rangs), 3),
    }
