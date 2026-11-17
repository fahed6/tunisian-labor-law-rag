"""
agent_rag — Agent d'actualité RAG, construit selon la méthode CRISP-DM.

Organisation du package (un sous-dossier = une étape CRISP-DM) :

    commun/               briques partagées (fichiers JSON, accès au LLM)
    etape2_donnees/       collecte RSS + exploration
    etape3_preparation/   nettoyage + découpage en chunks
    etape4_modelisation/  embeddings, index vectoriel, chaîne RAG classique
    etape5_evaluation/    jeu de test, métriques, baseline, LLM-juge
    etape6_deploiement/   outils + agent (boucle agentique)

Dans chaque dossier d'étape, le fichier `etape.py` contient la fonction
`executer()` qui orchestre l'étape ; les autres fichiers sont les briques.
L'étape 1 (compréhension du métier) est un document : docs/1_comprehension_metier.md
"""

__version__ = "1.0.0"
