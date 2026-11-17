"""
Étape 6 — interface en ligne de commande pour discuter avec l'agent.

    Entrée  : l'index de l'étape 4 + Ollama démarré
    Lancer  : python main.py agent
"""

import json

from agent_rag import config
from agent_rag.commun import llm
from agent_rag.etape4_modelisation.index_vectoriel import IndexVectoriel
from agent_rag.etape6_deploiement.agent import Agent
from agent_rag.etape6_deploiement.outils import creer_outils


def executer() -> None:
    disponible, message = llm.verifier_disponibilite()
    if not disponible:
        raise SystemExit(message)

    index = IndexVectoriel.charger()
    agent = Agent(creer_outils(index))
    print("=" * 70)
    print(f" Agent d'actualité (RAG) — LLM local : {config.MODELE_LLM}")
    print(f" Base : {len(index.chunks)} passages d'articles indexés")
    print(f" Outils : {', '.join(agent.outils.outils)}")
    print(" Tapez votre question ; 'nouveau' pour oublier le contexte ; 'quitter' pour sortir.")
    print("=" * 70)

    while True:
        question = input("\nVous > ").strip()
        if question.lower() in {"quitter", "exit", "q"}:
            print("Au revoir !")
            return
        if question.lower() == "nouveau":
            agent.reinitialiser()
            print("(Nouvelle conversation)")
            continue
        if not question:
            continue

        try:
            reponse = agent.repondre(question)
        except Exception as erreur:
            print(f"\n[Erreur] {erreur}")
            continue

        # La boucle agentique rendue visible
        for etape in reponse.etapes:
            origine = " (recherche de secours)" if etape.automatique else ""
            print(f"   [outil{origine}] {etape.nom}({json.dumps(etape.arguments, ensure_ascii=False)})")
        print(f"\nAgent > {reponse.texte}")
