"""
jeu_de_test.py — Construire un jeu de test (question, article attendu).

Sans annotateurs humains, on utilise une technique courante : la
GÉNÉRATION SYNTHÉTIQUE de questions. Pour quelques articles tirés au hasard,
le LLM écrit une question à laquelle l'article répond, SANS recopier le titre.

Plan B sans Ollama : les titres servent de questions (scores trop optimistes,
car la question et l'article partagent alors les mêmes mots).

Le jeu est SAUVEGARDÉ : on rejoue exactement la même évaluation après chaque
modification du modèle (reproductibilité = comparer ce qui est comparable).
"""

import random

from agent_rag import config
from agent_rag.commun import llm

CONSIGNE_QUESTION = """Voici un extrait d'article d'actualité :

{texte}

Écris UNE question courte, en français, qu'un lecteur pourrait poser et à laquelle
cet extrait répond. Reformule avec tes propres mots : ne recopie pas le titre.
Réponds uniquement par la question."""


def tirer_echantillon(chunks: list[dict], taille: int) -> list[dict]:
    """Un seul chunk par article (le premier) pour ne pas tester deux fois le même."""
    premiers = [c for c in chunks if c["id_chunk"].endswith("_0")]
    random.seed(config.GRAINE_ALEATOIRE)
    return random.sample(premiers, min(taille, len(premiers)))


def construire_jeu_de_test(chunks: list[dict], taille: int = config.TAILLE_JEU_TEST,
                           utiliser_llm: bool | None = None) -> list[dict]:
    if utiliser_llm is None:
        utiliser_llm, message = llm.verifier_disponibilite()
        if not utiliser_llm:
            print(f"  /!\\ {message}\n  -> Les TITRES servent de questions (scores optimistes).")

    jeu = []
    for i, chunk in enumerate(tirer_echantillon(chunks, taille), start=1):
        question = (llm.demander(CONSIGNE_QUESTION.format(texte=chunk["texte"]))
                    if utiliser_llm else chunk["titre"])
        print(f"  Q{i:02d}: {question}")
        jeu.append({"question": question, "id_article_attendu": chunk["id_article"]})
    return jeu
