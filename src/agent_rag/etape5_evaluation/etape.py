"""
Étape 5 — orchestration, verdict et rapport.

    Entrée  : l'index de l'étape 4 (+ Ollama pour générer le jeu de test / la partie B)
    Sortie  : data/prepare/jeu_de_test.json
              rapports/evaluation.json          (dernier résultat, détaillé)
              rapports/historique_experiences.csv (une ligne par exécution)
    Lancer  : python main.py evaluation                 # A. recherche seulement
              python main.py evaluation --generation    # A + B. réponses (lent)
              python main.py evaluation --nouveau-jeu   # régénère les questions

L'historique est la mémoire du CYCLE CRISP-DM : après chaque changement
(taille de chunk, modèle d'embedding, prompt...), on relance l'évaluation
et on compare les lignes du fichier CSV.
"""

import csv
from datetime import datetime

from agent_rag import config
from agent_rag.commun.fichiers import ecrire_json, lire_json
from agent_rag.etape4_modelisation.index_vectoriel import IndexVectoriel
from agent_rag.etape5_evaluation.baseline import recherche_par_mots_cles
from agent_rag.etape5_evaluation.eval_recherche import evaluer_recherche
from agent_rag.etape5_evaluation.jeu_de_test import construire_jeu_de_test

FICHIER_HISTORIQUE = config.DOSSIER_RAPPORTS / "historique_experiences.csv"


def obtenir_jeu_de_test(index: IndexVectoriel, nouveau: bool = False) -> list[dict]:
    """Réutilise le jeu sauvegardé (reproductibilité), sauf demande contraire."""
    jeu = None if nouveau else lire_json(config.FICHIER_JEU_TEST)
    ids_connus = {c["id_article"] for c in index.chunks}
    if jeu and all(ex["id_article_attendu"] in ids_connus for ex in jeu):
        print(f"  Jeu de test existant réutilisé ({len(jeu)} questions)")
        return jeu
    print("  Génération d'un nouveau jeu de test :")
    jeu = construire_jeu_de_test(index.chunks)
    ecrire_json(jeu, config.FICHIER_JEU_TEST)
    return jeu


def verifier_criteres(recherche: dict, generation: dict | None, k: int) -> dict[str, bool]:
    """Compare les résultats aux critères de succès de l'étape 1."""
    semantique, baseline = recherche["semantique"], recherche["baseline"]
    criteres = {
        f"Hit@{k} >= {config.SEUIL_HIT_K}": semantique[f"Hit@{k}"] >= config.SEUIL_HIT_K,
        f"MRR >= {config.SEUIL_MRR}": semantique["MRR"] >= config.SEUIL_MRR,
        "Recherche sémantique >= baseline (MRR)": semantique["MRR"] >= baseline["MRR"],
    }
    if generation:
        agent = generation["scores"]["agent"]
        criteres[f"Agent : citation >= {config.SEUIL_CITATION}"] = agent["citation"] >= config.SEUIL_CITATION
        criteres[f"Agent : fidélité >= {config.SEUIL_FIDELITE}"] = agent["fidelite"] >= config.SEUIL_FIDELITE
    return criteres


def ajouter_a_l_historique(rapport: dict, k: int) -> None:
    ligne = {
        "date": rapport["date"],
        "taille_chunk": config.TAILLE_CHUNK,
        "chevauchement": config.CHEVAUCHEMENT,
        "embedding": config.MODELE_EMBEDDING.split("/")[-1],
        "llm": config.MODELE_LLM,
        "k": k,
        "nb_questions": rapport["nb_questions"],
        **{f"semantique_{m}": v for m, v in rapport["recherche"]["semantique"].items()},
        **{f"baseline_{m}": v for m, v in rapport["recherche"]["baseline"].items()},
        "valide": rapport["valide"],
    }
    if rapport.get("generation"):
        for nom, scores in rapport["generation"]["scores"].items():
            ligne.update({f"{nom}_{m}": v for m, v in scores.items()})
    nouveau = not FICHIER_HISTORIQUE.exists()
    FICHIER_HISTORIQUE.parent.mkdir(parents=True, exist_ok=True)
    with open(FICHIER_HISTORIQUE, "a", newline="", encoding="utf-8") as f:
        ecrivain = csv.DictWriter(f, fieldnames=list(ligne), extrasaction="ignore")
        if nouveau:
            ecrivain.writeheader()
        ecrivain.writerow(ligne)


def executer(generation: bool = False, nouveau_jeu: bool = False) -> dict:
    print("ÉTAPE 5 — Évaluation\n")
    k = config.NB_RESULTATS
    index = IndexVectoriel.charger()

    print("5.1 Jeu de test")
    jeu = obtenir_jeu_de_test(index, nouveau_jeu)

    print("\n5.2 A. Évaluation de la RECHERCHE")
    recherche = {
        "baseline": evaluer_recherche(
            lambda q, k: recherche_par_mots_cles(q, index.chunks, k), jeu, k),
        "semantique": evaluer_recherche(index.rechercher, jeu, k),
    }
    print("=" * 58)
    print(f"{'Métrique':<12}{'Baseline mots-clés':>22}{'RAG sémantique':>22}")
    print("-" * 58)
    for m in recherche["semantique"]:
        print(f"{m:<12}{recherche['baseline'][m]:>22.2f}{recherche['semantique'][m]:>22.2f}")
    print("=" * 58)

    resultats_generation = None
    if generation:
        from agent_rag.commun import llm
        from agent_rag.etape5_evaluation.eval_generation import evaluer_generation
        disponible, message = llm.verifier_disponibilite()
        if disponible:   # le juge est un autre modèle : il doit aussi être installé
            disponible, message = llm.verifier_disponibilite(config.MODELE_JUGE)
        if not disponible:
            print(f"\n/!\\ Partie B ignorée : {message}")
        else:
            print(f"\n5.3 B. Évaluation des RÉPONSES : RAG classique vs Agent "
                  f"({config.TAILLE_EVAL_GENERATION} questions, LLM local : patience)")
            resultats_generation = evaluer_generation(index, jeu[:config.TAILLE_EVAL_GENERATION])
            print("\n" + "=" * 58)
            print(f"{'Métrique':<16}{'RAG classique':>20}{'Agent':>20}")
            print("-" * 58)
            s = resultats_generation["scores"]
            for m in s["agent"]:
                print(f"{m:<16}{str(s['rag_classique'].get(m, '—')):>20}{str(s['agent'][m]):>20}")
            print("=" * 58)

    criteres = verifier_criteres(recherche, resultats_generation, k)
    print("\n5.4 Critères de succès (fixés à l'étape 1) :")
    for nom, ok in criteres.items():
        print(f"  {'OK         ' if ok else 'NON ATTEINT'}  {nom}")

    valide = all(criteres.values())
    rapport = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "nb_questions": len(jeu),
        "recherche": recherche,
        "generation": resultats_generation,
        "criteres": criteres,
        "valide": valide,
    }
    ecrire_json(rapport, config.FICHIER_RAPPORT_EVALUATION)
    ajouter_a_l_historique(rapport, k)
    print(f"\nRapport : {config.FICHIER_RAPPORT_EVALUATION}\nHistorique : {FICHIER_HISTORIQUE}")

    if valide:
        print("\n=> Modèle VALIDÉ : on peut passer au DÉPLOIEMENT (étape 6).")
    else:
        print("\n=> Retour aux étapes 3 ou 4 (CRISP-DM est un CYCLE) : autre taille de")
        print("   chunk, autre modèle d'embedding, plus de données, meilleur prompt...")
    return rapport
