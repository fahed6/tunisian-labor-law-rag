"""
eval_generation.py — B. Évaluer les RÉPONSES : RAG classique vs Agent.

Pour chaque question, on mesure :
  - bon_document : le bon article fait-il partie des documents consultés ?
  - cite_source  : la réponse cite-t-elle un document consulté (lien ou titre exact) ?
                   (critère MÉTIER de l'étape 1 : « l'utilisateur peut vérifier »)
  - fidele       : un LLM-JUGE estime-t-il que la réponse est appuyée par
                   les extraits ? (technique « LLM-as-a-judge »)
                   Le juge est un modèle PLUS FORT que celui évalué (config.MODELE_JUGE).
  - temps        : secondes pour répondre (critère de DÉPLOIEMENT)
  - outil_utilise (agent) : le LLM a-t-il DÉCIDÉ lui-même d'appeler la recherche ?
  - secours (agent)       : le garde-fou a-t-il dû chercher à sa place ?

Leçon tirée des tests : avec qwen2.5:3b comme juge, une réponse recopiée mot pour
mot des extraits était jugée « NON » 3 fois sur 3. Un juge faible rend la métrique
inutilisable : on vérifie TOUJOURS le juge sur quelques cas évidents (vrais ET faux).
"""

import time

from agent_rag import config
from agent_rag.commun import llm
from agent_rag.etape4_modelisation.index_vectoriel import IndexVectoriel
from agent_rag.etape4_modelisation.rag import ChaineRAG
from agent_rag.etape5_evaluation.metriques import taux
from agent_rag.etape6_deploiement.agent import creer_agent

CONSIGNE_JUGE = """Tu évalues la FIDÉLITÉ d'une réponse par rapport à des extraits d'articles.

EXTRAITS :
{extraits}

RÉPONSE À ÉVALUER :
{reponse}

Question : les informations factuelles de la RÉPONSE sont-elles appuyées par les EXTRAITS ?
- Ignore la liste des sources, les titres et les liens : juge uniquement les faits affirmés.
- Une réponse qui dit honnêtement ne pas trouver l'information est FIDÈLE.
- Une réponse qui contient un fait absent des extraits ou contradictoire n'est PAS fidèle.
Réponds par un seul mot : OUI ou NON."""


def juger_fidelite(reponse: str, documents: list[dict]) -> bool:
    """LLM-juge : un modèle PLUS FORT (config.MODELE_JUGE), à température 0 (stable)."""
    if not documents:   # aucun extrait : fidèle seulement si la réponse n'affirme rien
        documents = [{"titre": "(aucun extrait)", "lien": "", "texte": ""}]
    extraits = "\n".join(
        f"- {d.get('titre', '')} ({d.get('lien', '')}) : {d.get('texte') or d.get('extrait', '')}"
        for d in documents)
    verdict = llm.demander(CONSIGNE_JUGE.format(extraits=extraits, reponse=reponse),
                           modele=config.MODELE_JUGE, temperature=0)
    return verdict.strip().strip("*").upper().startswith("OUI")


def cite_une_source(reponse: str, documents: list[dict]) -> bool:
    """La réponse cite un document consulté : son lien OU son titre exact."""
    texte = reponse.lower()
    return any((d.get("lien") and d["lien"].lower() in texte)
               or (d.get("titre") and d["titre"].lower() in texte)
               for d in documents)


def evaluer_generation(index: IndexVectoriel, jeu_de_test: list[dict]) -> dict:
    """Compare RAG classique et Agent sur les mêmes questions."""
    chaine_rag = ChaineRAG(index.rechercher)
    agent = creer_agent(index)
    lien_vers_article = {c["lien"]: c["id_article"] for c in index.chunks}
    details = {"rag_classique": [], "agent": []}

    for i, exemple in enumerate(jeu_de_test, start=1):
        question, attendu = exemple["question"], exemple["id_article_attendu"]
        print(f"\n  Q{i}: {question}")

        # --- RAG classique ---------------------------------------------------
        debut = time.perf_counter()
        r = chaine_rag.repondre(question)
        details["rag_classique"].append({
            "question": question,
            "temps": round(time.perf_counter() - debut, 1),
            "bon_document": any(p["id_article"] == attendu for p in r.passages),
            "cite_source": cite_une_source(r.texte, r.passages),
            "fidele": juger_fidelite(r.texte, r.passages),
            "reponse": r.texte,
        })

        # --- Agent (nouvelle conversation pour chaque question) ----------------
        agent.reinitialiser()
        debut = time.perf_counter()
        a = agent.repondre(question)
        documents = [doc for e in a.etapes if isinstance(e.resultat, list) for doc in e.resultat]
        details["agent"].append({
            "question": question,
            "temps": round(time.perf_counter() - debut, 1),
            # le LLM a-t-il DÉCIDÉ lui-même de chercher ? (hors recherche de secours)
            "outil_utilise": any(e.nom == "rechercher_actualites" and not e.automatique
                                 for e in a.etapes),
            "secours": any(e.automatique for e in a.etapes),
            "bon_document": any(lien_vers_article.get(s["lien"]) == attendu for s in a.sources),
            "cite_source": cite_une_source(a.texte, a.sources),
            "fidele": juger_fidelite(a.texte, documents),
            "reponse": a.texte,
        })

        for nom, liste in details.items():
            d = liste[-1]
            print(f"     {nom:<14} {d['temps']:>6}s | bon doc {d['bon_document']!s:<5} "
                  f"| cite {d['cite_source']!s:<5} | fidèle {d['fidele']}")

    return {"scores": {nom: resumer(liste) for nom, liste in details.items()},
            "details": details}


def resumer(liste: list[dict]) -> dict:
    scores = {
        "bon_document": round(taux([d["bon_document"] for d in liste]), 3),
        "citation": round(taux([d["cite_source"] for d in liste]), 3),
        "fidelite": round(taux([d["fidele"] for d in liste]), 3),
        "temps_moyen_s": round(sum(d["temps"] for d in liste) / len(liste), 1) if liste else 0,
    }
    if liste and "outil_utilise" in liste[0]:
        scores["outil_utilise"] = round(taux([d["outil_utilise"] for d in liste]), 3)
        scores["secours"] = round(taux([d["secours"] for d in liste]), 3)
    return scores
