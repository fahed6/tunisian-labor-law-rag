"""
Étape 4 — orchestration.

    Entrée  : data/prepare/chunks.json
    Sortie  : data/index/vecteurs.npy + data/index/metadonnees.json
    Lancer  : python main.py modelisation
"""

from agent_rag import config
from agent_rag.commun.fichiers import exiger, lire_json
from agent_rag.etape4_modelisation.index_vectoriel import IndexVectoriel

# Requêtes de vérification : 5 questions de droit du travail, formulées SANS recopier
# les mots des articles, + 1 question HORS SUJET (on observe alors des scores bas :
# utile pour décider quand l'agent doit répondre « je ne sais pas »).
REQUETES_TEST = [
    "combien de temps dure la période d'essai",
    "mon patron m'a renvoyé sans raison, quelle indemnité puis-je demander",
    "où déposer une plainte contre mon employeur",
    "jours de vacances payées par an",
    "salaire en dessous du minimum légal",
    "recette du couscous tunisien",          # hors sujet : le Code du travail n'en parle pas
]


def executer() -> IndexVectoriel:
    print("ÉTAPE 4 — Modélisation : construction de l'index vectoriel")
    exiger(config.FICHIER_CHUNKS, "preparation")

    chunks = lire_json(config.FICHIER_CHUNKS)
    print(f"  {len(chunks)} chunks à vectoriser")
    index = IndexVectoriel.construire(chunks)
    n, d = index.dimensions
    print(f"  Matrice d'embeddings : ({n}, {d}) = {n} chunks × {d} dimensions")
    index.sauvegarder()
    print(f"  -> {config.DOSSIER_INDEX}")

    # Vérification rapide : les résultats ont-ils du SENS ?
    print("\nTest de la recherche sémantique :")
    for requete in REQUETES_TEST:
        print(f"\n  « {requete} »")
        for r in index.rechercher(requete, k=3):
            lieu = f"partie {r['partie']}/{r['nb_parties']}, p.{r.get('page_pdf', '?')}"
            print(f"    [{r['score']:.3f}] {r['titre']} ({lieu}) | {r['texte'][:70]}...")
    return index
