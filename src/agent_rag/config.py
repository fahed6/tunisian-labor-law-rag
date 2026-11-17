"""
config.py — TOUS les réglages du projet, au même endroit.
=========================================================

Règle d'architecture : aucune « valeur magique » dans le reste du code.
Pour changer de modèle, de flux RSS, de taille de découpage ou de seuil
de réussite, c'est ICI et seulement ici.

Le fichier suit l'ordre des étapes CRISP-DM.
"""

from pathlib import Path

# ==========================================================================
# Chemins
# ==========================================================================
# Ce fichier est dans  <racine>/src/agent_rag/config.py
#   parents[0] = agent_rag/   parents[1] = src/   parents[2] = <racine>
RACINE = Path(__file__).resolve().parents[2]

DOSSIER_DONNEES = RACINE / "data"
DOSSIER_BRUT = DOSSIER_DONNEES / "brut"          # étape 2 : données telles que collectées
DOSSIER_PREPARE = DOSSIER_DONNEES / "prepare"    # étape 3 : données nettoyées + découpées
DOSSIER_INDEX = DOSSIER_DONNEES / "index"        # étape 4 : vecteurs + métadonnées
DOSSIER_EXEMPLE = DOSSIER_DONNEES / "exemple"    # jeu fictif pour travailler hors-ligne
DOSSIER_RAPPORTS = RACINE / "rapports"           # étape 5 : résultats d'évaluation
DOSSIER_DOCS = RACINE / "docs"

FICHIER_PDF_SOURCE = DOSSIER_BRUT / "code_du_travail_2010.pdf"      # PDF d'origine (jamais modifié)
FICHIER_TEXTE_BRUT = DOSSIER_BRUT / "code_du_travail_2010.txt"      # texte extrait du PDF, tel quel
FICHIER_ARTICLES_BRUTS = DOSSIER_BRUT / "articles_bruts.json"
FICHIER_EXEMPLE = DOSSIER_EXEMPLE / "articles_exemple.json"
FICHIER_CHUNKS = DOSSIER_PREPARE / "chunks.json"
FICHIER_STATS_PREPARATION = DOSSIER_PREPARE / "stats_preparation.json"   # chiffres avant / après (étape 3)
FICHIER_VECTEURS = DOSSIER_INDEX / "vecteurs.npy"
FICHIER_METADONNEES = DOSSIER_INDEX / "metadonnees.json"
FICHIER_JEU_TEST = DOSSIER_PREPARE / "jeu_de_test.json"
FICHIER_RAPPORT_EVALUATION = DOSSIER_RAPPORTS / "evaluation.json"
FICHIER_DOC_METIER = DOSSIER_DOCS / "1_comprehension_metier.md"

# ==========================================================================
# Étape 1 — Critères de succès (fixés AVANT de coder, vérifiés à l'étape 5)
# ==========================================================================
SEUIL_HIT_K = 0.70        # le bon article est dans les k premiers résultats
SEUIL_MRR = 0.50          # rang moyen du bon article (1 = toujours premier)
SEUIL_CITATION = 0.80     # part des réponses qui citent au moins une source
SEUIL_FIDELITE = 0.70     # part des réponses jugées fidèles aux extraits

# ==========================================================================
# Étape 2 — Source de données : Code du travail tunisien (PDF)
# ==========================================================================
# Texte de loi : source publique. Édition 2010 des Publications officielles.
# /!\ Plusieurs lois l'ont modifié depuis : limite à signaler (fraîcheur).
NOM_SOURCE = "Code du travail tunisien (édition 2010)"
URL_SOURCE = "https://a-mla.org/tunisia/code-du-travail-1966"
DATE_LOI_ORIGINE = (30, 4, 1966)    # loi n° 66-27 du 30 avril 1966 (jour, mois, année)
STATUT_EN_VIGUEUR = "en vigueur"
STATUT_ABROGE = "abrogé"            # article supprimé par une loi : il n'a plus de texte

# ==========================================================================
# Étape 3 — Préparation : un article = un chunk, sauf s'il est long
# ==========================================================================
# Justification (exploration, 484 articles en vigueur) : médiane 62 mots, 72 % des
# articles ont <= 100 mots, 28 % davantage (max 790). Un article est l'unité
# naturelle d'un texte de loi : on le garde ENTIER quand il tient dans TAILLE_CHUNK
# mots, et on ne découpe que les plus longs. 100 mots reste proche de la fenêtre
# de lecture du modèle d'embedding (~128 tokens) : au-delà, la fin du texte
# n'est pas « vue » par la recherche. Valeur de départ, À TESTER à l'étape 5
# (expériences : 60 / 100 / 200 mots, un seul réglage change à la fois).
TAILLE_CHUNK = 100          # nombre de mots MAXIMUM par morceau
CHEVAUCHEMENT = 20          # mots repris du morceau précédent (articles longs seulement)
LONGUEUR_MIN_ARTICLE = 8    # en dessous (en mots), l'article est écarté

# Sélection manuelle, DOCUMENTÉE : articles sans règle applicable, qui polluent la recherche.
# Constaté en testant la baseline : « Loi 66-27, article 4 » (790 mots = liste de décrets
# abrogés en 1966) remontait en réponse à des questions sans rapport.
TITRES_ECARTES = {
    "Loi 66-27, article 4": "liste de décrets abrogés en 1966 : aucune règle applicable",
}

# Métadonnées recopiées sur CHAQUE chunk : ce sont elles qui permettent de citer
# précisément (« Article 14, Chapitre III — Fin du contrat de travail, p. 19 »).
CHAMPS_METADONNEES = ["numero", "texte_loi", "livre", "division", "chapitre",
                      "section", "modification", "statut", "page_pdf"]

# ==========================================================================
# Étape 4 — Modélisation : embedding (le « R » de RAG) + LLM (le « G »)
# ==========================================================================
# Choix 1 — modèle d'embedding : multilingue (le texte est en français, des questions
# en arabe seraient possibles plus tard), léger (~470 Mo, vecteurs de 384 nombres),
# exécuté en LOCAL sur le processeur, gratuit, aucune donnée envoyée à l'extérieur.
# Limite connue : il ne « lit » qu'une fenêtre limitée de texte (voir max_seq_length) ;
# d'où des chunks courts (étape 3). À comparer à l'étape 5 avec un 2e modèle.
MODELE_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Choix 2 — k = 3 : assez pour qu'une bonne réponse figure parmi les résultats
# (un même sujet est souvent traité dans 2-3 articles), assez peu pour que le prompt
# reste court : un LLM de 3 milliards de paramètres se perd avec trop de texte.
# Hypothèse à valider à l'étape 5 en comparant Hit@1, Hit@3 et Hit@5.
NB_RESULTATS = 3            # le « k » de top-k

# LLM open source exécuté en local par Ollama (https://ollama.com).
# Choix 3 — LLM : il DOIT savoir utiliser des outils (tools) ; qwen2.5 le fait, comprend
# le français, est open source et tourne sur un PC sans carte graphique (lentement).
# Limite connue : un petit modèle « oublie » parfois d'appeler l'outil de recherche
# (d'où le garde-fou de l'étape 6). Alternatives : llama3.1, mistral-nemo...
#   ollama pull qwen2.5:3b        (rapide, ~1,9 Go)
#   ollama pull qwen2.5:7b        (plus précis, plus lent)
MODELE_LLM = "qwen2.5:3b"
TEMPERATURE_LLM = 0.2       # basse = moins d'inventions (essentiel pour un RAG)
MAX_TOKENS_REPONSE = 400    # longueur max d'une réponse

# ==========================================================================
# Étape 5 — Évaluation
# ==========================================================================
TAILLE_JEU_TEST = 15        # questions pour évaluer la recherche
TAILLE_EVAL_GENERATION = 4  # questions pour évaluer les réponses (lent sur CPU)
# Le JUGE doit être plus fort que le modèle évalué : avec qwen2.5:3b comme juge,
# des réponses manifestement fidèles étaient rejetées (constaté lors des tests).
#   ollama pull qwen2.5:7b
MODELE_JUGE = "qwen2.5:7b"
GRAINE_ALEATOIRE = 42       # même tirage à chaque exécution (reproductibilité)

# ==========================================================================
# Étape 6 — Déploiement : l'agent
# ==========================================================================
MAX_TOURS_AGENT = 4         # garde-fou : allers-retours max avec les outils
RECHERCHE_DE_SECOURS = True # garde-fou : si le LLM conclut sans AUCUN outil, notre code
                            # lance la recherche à sa place (voir agent.py).
                            # Mettre False pour observer le comportement brut du modèle.
