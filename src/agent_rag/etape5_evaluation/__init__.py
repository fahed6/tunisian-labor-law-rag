"""
ÉTAPE 5 (CRISP-DM) — ÉVALUATION (Evaluation)

On vérifie AVEC DES CHIFFRES les critères de succès fixés à l'étape 1,
à deux niveaux :

  A. La RECHERCHE (le « R ») — rapide, sans LLM une fois le jeu de test créé
       jeu_de_test.py      paires (question, article attendu) générées par le LLM
       baseline.py         modèle de référence naïf : mots en commun
       metriques.py        Hit@1, Hit@k, MRR
       eval_recherche.py   compare baseline vs recherche sémantique

  B. Les RÉPONSES (le « G ») — plus lent : RAG classique vs Agent
       eval_generation.py  citation des sources, fidélité (LLM-juge), temps

  etape.py   orchestration + verdict + rapport (rapports/evaluation.json)
"""
