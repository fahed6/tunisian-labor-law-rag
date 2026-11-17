"""
metriques.py — Les métriques, en fonctions pures (faciles à tester).

    Hit@1 : % de questions dont le bon article est en 1re position
    Hit@k : % de questions dont le bon article est dans les k premiers
    MRR   : Mean Reciprocal Rank = moyenne de 1/rang
            rang 1 -> 1 ; rang 2 -> 0,5 ; rang 3 -> 0,33 ; absent -> 0
"""


def rang(resultats: list[dict], id_attendu: str) -> int | None:
    """Position (1, 2, 3...) de l'article attendu dans les résultats, ou None."""
    for position, resultat in enumerate(resultats, start=1):
        if resultat["id_article"] == id_attendu:
            return position
    return None


def hit_at(rangs: list[int | None], k: int) -> float:
    return sum(1 for r in rangs if r is not None and r <= k) / len(rangs) if rangs else 0.0


def mrr(rangs: list[int | None]) -> float:
    return sum(1 / r for r in rangs if r) / len(rangs) if rangs else 0.0


def taux(valeurs: list[bool]) -> float:
    """Proportion de True (ex. : part des réponses qui citent une source)."""
    return sum(valeurs) / len(valeurs) if valeurs else 0.0
