"""
ÉTAPE 6 (CRISP-DM) — DÉPLOIEMENT (Deployment)

On met le modèle RAG entre les mains de l'utilisateur, sous forme d'AGENT :

    outils.py    les outils (fonctions Python) + leur description pour le LLM
    prompts.py   les consignes de l'agent (system prompt)
    agent.py     la boucle agentique : LLM <-> outils, avec mémoire
    etape.py     interface en ligne de commande (python main.py agent)

Deux autres interfaces réutilisent EXACTEMENT le même agent :
    app/interface.py   page web Streamlit (démonstration)
    app/api.py         API REST FastAPI (intégration dans d'autres logiciels)
"""
