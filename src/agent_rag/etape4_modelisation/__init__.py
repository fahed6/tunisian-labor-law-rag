"""
ÉTAPE 4 (CRISP-DM) — MODÉLISATION (Modeling)

On construit le modèle RAG = Retrieval-Augmented Generation, en 3 temps :

    R  Retrieve  : retrouver les passages proches de la question
                   embeddings.py       texte -> vecteur
                   index_vectoriel.py  stockage des vecteurs + recherche cosinus
    A  Augment   : insérer ces passages dans la consigne du LLM   -> rag.py
    G  Generate  : le LLM rédige la réponse à partir des passages -> rag.py

    etape.py     orchestration : chunks -> vecteurs -> index sur disque -> tests

À l'étape 6, ce même moteur de recherche devient un OUTIL qu'un AGENT
décide (ou non) d'utiliser.
"""
