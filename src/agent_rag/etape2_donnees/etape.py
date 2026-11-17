"""
Étape 2 — orchestration.

    Entrée  : data/brut/code_du_travail_2010.pdf
    Sortie  : data/brut/articles_bruts.json   (données BRUTES, jamais modifiées)
    Lancer  : python main.py donnees
"""

from agent_rag import config
from agent_rag.commun.fichiers import ecrire_json
from agent_rag.etape2_donnees.collecte import collecter
from agent_rag.etape2_donnees.exploration import afficher_rapport, explorer


def executer() -> list[dict]:
    print("ÉTAPE 2 — Compréhension des données\n2.1 Collecte du Code du travail (PDF) :")
    articles, origine = collecter()

    # On sauvegarde les données BRUTES telles quelles : si l'on veut refaire
    # la préparation autrement, on repart d'ici sans tout re-télécharger.
    ecrire_json(articles, config.FICHIER_ARTICLES_BRUTS)
    print(f"\n{len(articles)} articles ({origine}) -> {config.FICHIER_ARTICLES_BRUTS}")

    print("\n2.2 Exploration :")
    afficher_rapport(explorer(articles), exemple=articles[0] if articles else None)
    return articles
