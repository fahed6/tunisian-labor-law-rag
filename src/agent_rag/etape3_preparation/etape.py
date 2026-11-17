"""
Étape 3 — orchestration.

    Entrée  : data/brut/articles_bruts.json
    Sortie  : data/prepare/chunks.json  +  data/prepare/stats_preparation.json
    Lancer  : python main.py preparation
"""

import json
import statistics

from agent_rag import config
from agent_rag.commun.fichiers import ecrire_json, exiger, lire_json
from agent_rag.etape3_preparation.decoupage import creer_chunks
from agent_rag.etape3_preparation.nettoyage import preparer_articles


def preparer(articles_bruts: list[dict]) -> tuple[list[dict], dict]:
    """Brut -> chunks (sans lire ni écrire de fichier : réutilisable partout)."""
    articles, stats = preparer_articles(articles_bruts)
    chunks = creer_chunks(articles)

    mots = [len(c["texte"].split()) for c in chunks] or [0]
    stats["chunks"] = len(chunks)
    stats["articles_decoupes"] = len({c["id_article"] for c in chunks if c["nb_parties"] > 1})
    stats["mots_par_chunk"] = {"min": min(mots), "mediane": statistics.median(mots), "max": max(mots)}
    stats["taille_chunk"] = config.TAILLE_CHUNK
    stats["chevauchement"] = config.CHEVAUCHEMENT
    return chunks, stats


def afficher_statistiques(stats: dict) -> None:
    """Le tableau « avant / après » : combien supprimé, et POURQUOI."""
    ecartes = stats["abroges"] + stats["ecartes_manuels"] + stats["doublons"] + stats["trop_courts"]
    print("\n  AVANT / APRÈS")
    print(f"  Articles bruts lus        : {stats['lus']:>5}   ({stats['mots_avant']} mots)")
    print(f"  - abrogés (sans texte)    : {stats['abroges']:>5}   [3.1 sélection]")
    print(f"  - écartés à la main       : {stats['ecartes_manuels']:>5}   [3.1 sélection, voir config.TITRES_ECARTES]")
    print(f"  - doublons                : {stats['doublons']:>5}   [3.2 nettoyage]")
    print(f"  - trop courts (< {config.LONGUEUR_MIN_ARTICLE} mots)   : {stats['trop_courts']:>5}   [3.1 sélection]")
    print(f"  Articles conservés        : {stats['conserves']:>5}   ({stats['mots_apres']} mots, "
          f"{ecartes} écartés)")
    lg = stats["mots_par_chunk"]
    print(f"  Chunks créés              : {stats['chunks']:>5}   (max {stats['taille_chunk']} mots, "
          f"chevauchement {stats['chevauchement']})")
    print(f"  dont articles découpés    : {stats['articles_decoupes']:>5}   "
          f"(les autres restent entiers : 1 article = 1 chunk)")
    print(f"  Mots par chunk (en-tête compris) : min {lg['min']}, médiane {lg['mediane']}, max {lg['max']}")


def executer() -> list[dict]:
    print("ÉTAPE 3 — Préparation des données")
    exiger(config.FICHIER_ARTICLES_BRUTS, "donnees")

    chunks, stats = preparer(lire_json(config.FICHIER_ARTICLES_BRUTS))
    afficher_statistiques(stats)

    ecrire_json(chunks, config.FICHIER_CHUNKS)
    ecrire_json(stats, config.FICHIER_STATS_PREPARATION)
    print(f"\n-> {config.FICHIER_CHUNKS}\n-> {config.FICHIER_STATS_PREPARATION}")
    if chunks:
        # on montre un chunk d'article LONG s'il y en a un : les métadonnées y sont plus parlantes
        exemple = next((c for c in chunks if c["nb_parties"] > 1), chunks[0])
        print("\nExemple de chunk prêt pour l'étape 4 :")
        print(json.dumps(exemple, ensure_ascii=False, indent=2))
    return chunks
