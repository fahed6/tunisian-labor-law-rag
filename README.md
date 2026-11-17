# Agent d'actualité RAG — un projet de référence CRISP-DM

Projet pédagogique complet : construire **de A à Z** un système **RAG**
(*Retrieval-Augmented Generation*) puis un **agent** qui l'utilise comme **outil**,
en suivant les **6 étapes de CRISP-DM**.

L'agent répond à des questions sur l'actualité du jour (flux RSS de France 24,
Le Monde, franceinfo) **en citant ses sources**.
**Tout est gratuit et local** : embeddings et LLM open source (Qwen 2.5 via Ollama), aucune clé API.

```
   1. Métier ──► 2. Données ──► 3. Préparation ──► 4. Modélisation ──► 5. Évaluation ──► 6. Déploiement
       ▲                              ▲                  (RAG)              │              (Agent)
       │                              └─────── critères non atteints ───────┤
       └────────────────────────────── le besoin a changé ──────────────────┘
```

## Documentation

| Document | Contenu |
|---|---|
| [`docs/1_comprehension_metier.md`](docs/1_comprehension_metier.md) | **Étape 1** : problème, solution, critères de succès chiffrés |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Organisation du code, flux de données, RAG vs Agent, principes |
| [`docs/GUIDE_DEMO.md`](docs/GUIDE_DEMO.md) | Comment exécuter et **présenter** chaque étape |
| [`docs/GUIDE_ETUDIANTS.md`](docs/GUIDE_ETUDIANTS.md) ([PDF](docs/GUIDE_ETUDIANTS.pdf)) | **Cahier des charges** pour réaliser votre propre projet (modalités, livrables, grille) |

## Architecture en un coup d'œil

```
main.py                          point d'entrée unique
src/agent_rag/
├── config.py                    tous les réglages
├── commun/                      fichiers JSON + accès au LLM
├── etape2_donnees/              collecte RSS + exploration
├── etape3_preparation/          nettoyage + chunks
├── etape4_modelisation/         embeddings, index vectoriel, chaîne RAG
├── etape5_evaluation/           jeu de test, baseline, métriques, LLM-juge
└── etape6_deploiement/          outils + agent (boucle agentique)
app/                             API FastAPI + interface Streamlit
notebooks/                       démonstration étape par étape
tests/                           tests automatisés (pytest)
data/  rapports/                 sorties de chaque étape
```

Un dossier = une étape CRISP-DM. Détails dans [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Installation

Pré-requis : Python ≥ 3.10 et [Ollama](https://ollama.com/download).

```bash
ollama pull qwen2.5:3b          # le LLM de l'agent (~1,9 Go, rapide)
ollama pull qwen2.5:7b          # le LLM-juge de l'étape 5 (~4,7 Go, plus fiable)

uv sync                         # option A : avec uv (recommandé)
# ou
python -m venv .venv            # option B : avec pip
.venv\Scripts\activate          #   (Linux/Mac : source .venv/bin/activate)
pip install -r requirements.txt
```

Le modèle d'embedding (~470 Mo) se télécharge seul au premier lancement de l'étape 4.

## Utilisation

Une seule commande, une sous-commande par étape (avec uv : `uv run python main.py ...`) :

| Étape CRISP-DM | Commande |
|---|---|
| 2. Données | `python main.py donnees` |
| 3. Préparation | `python main.py preparation` |
| 4. Modélisation | `python main.py modelisation` |
| 4. Tester le RAG classique | `python main.py rag "Quelles sont les nouvelles du rugby ?"` |
| 5. Évaluation (recherche) | `python main.py evaluation` |
| 5. Évaluation (+ réponses RAG vs agent) | `python main.py evaluation --generation` |
| 6. Agent dans le terminal | `python main.py agent` |
| 6. API REST | `python main.py api` → http://localhost:8000/docs |
| 6. Interface web | `python main.py interface` → http://localhost:8501 (ou double-clic sur `lancer_interface.bat`) |
| 2 → 5 d'un coup | `python main.py pipeline` |

```bash
uv run pytest                   # 29 tests, < 1 s, sans Internet ni LLM
```

Sans Internet, l'étape 2 bascule sur un petit jeu d'articles **fictifs** (`data/exemple/`).

### Exemple de session avec l'agent

```
Vous > Quel est le score du match France-Turquie ?
   [outil] rechercher_actualites({"requete": "score match France-Turquie"})

Agent > D'après les articles, les Bleus ont pris l'avantage sur un but contre son camp…
Sources :
- France-Turquie : sur corner, les Bleus prennent l'avantage… (franceinfo) : https://...
```

La ligne `[outil]` montre le moment où **l'agent décide seul** d'appeler l'outil RAG.

## Pistes d'exercices

1. **Étape 3** — Faire varier `TAILLE_CHUNK` (50, 120, 300) et comparer dans `rapports/historique_experiences.csv`.
2. **Étape 4** — Tester un autre modèle d'embedding (`intfloat/multilingual-e5-small`).
3. **Étape 4** — Remplacer l'index NumPy par ChromaDB ou FAISS (seul `index_vectoriel.py` change).
4. **Étape 5** — Écrire 10 questions de test **à la main** et comparer aux questions générées.
5. **Étape 5** — Modifier `prompts.py` et mesurer l'effet sur la citation et la fidélité.
6. **Étape 6** — Ajouter un outil `filtrer_par_source(source)` ou un filtre par date.
7. **Étape 6** — Brancher un autre LLM (OpenAI, Claude, Gemini) en ne modifiant que `commun/llm.py`.

## Limites connues (à discuter en cours)

- Les flux RSS ne donnent que **titre + résumé** : l'agent ne connaît pas l'article complet.
- L'index doit être **reconstruit** pour intégrer l'actualité du jour (`python main.py pipeline`).
- Le jeu de test est **synthétique** (généré par le LLM) et le **juge** est un LLM (qwen2.5:7b) :
  utile, mais moins fiable qu'une annotation humaine.
- Le petit LLM local (qwen2.5:3b) décide mal quand appeler un outil : un **garde-fou** dans le code
  (« recherche de secours ») garantit la recherche. Voir la section 7 de `docs/ARCHITECTURE.md`.
- Sur un PC sans carte graphique, une réponse de l'agent prend de 30 s à 2 min.














articles -> 519 -> 484
675 chunks 

(UIT-Project-main) PS C:\Users\manno\Downloads\UIT-Project-main> uv run python main.py evaluation
ÉTAPE 5 — Évaluation

5.1 Jeu de test
  Génération d'un nouveau jeu de test :
  Q01: Quelles sont les principales obligations de l'employeur et du travailleur selon cet extrait ?
  Q02: Quelles sont les responsabilités des groupements de travailleurs et d'employeurs liés par une convention collective de travail ?
  Q03: Quels sont les cas dans lesquels un contrat de travail à durée déterminée peut être conclu selon cet extrait ?
  Q04: Quelles sont les conditions minimales d'âge pour l'apprentissage des mineurs selon cet arrêté ?
  Q05: Dans les entreprises minières ou sur les chantiers éloignés, comment sont organisés les jours de repos hebdomadaire ?
  Q06: Quelles sont les limitations légales concernant le nombre d'heures supplémentaires pour un travailleur à temps partiel ?
  Q07: Quels sont les arrêtés qui déterminent la répartition journalière de la durée légale du travail dans les entreprises agricoles ?
  Q08: Quelle est la procédure à suivre pour dénoncer une convention collective d'établissement conclue pour une durée indéterminée ?
  Q09: Quels sont les documents obligatoires pour le contrat d'apprentissage et comment sont-ils conservés ?
  Q10: Quelles sont les sanctions auxquelles sont soumis les employeurs en cas de non-respect des prescriptions légales concernant les conditions de travail ?
  Q11: Qui sont tenus de permettre aux agents spécialisés d'effectuer des constatations dans les établissements dangereux, insalubres ou incommodes ?
  Q12: Quelles sont les conditions prévues aux articles suivants pour l'établissement d'une carte professionnelle d'identité pour les voyageurs et représentants de commerce en Tunisie ?
  Q13: Quels sont les délais et les procédures pour répondre à une récusation lors d'un procès prud'hommal ?
  Q14: Quelles sont les sanctions pour la rupture abusive du contrat de travail selon cet extrait d'article ?
  Q15: Quelles sont les conditions pour que des étrangers puissent être désignés ou élus à un poste d'administration ou de direction d'un syndicat professionnel en Tunisie ?

5.2 A. Évaluation de la RECHERCHE
  Chargement du modèle d'embedding : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 898.07it/s]
==========================================================
Métrique        Baseline mots-clés        RAG sémantique
----------------------------------------------------------
Hit@1                         0.67                  0.67
Hit@3                         0.80                  0.80
MRR                           0.71                  0.73
==========================================================

5.4 Critères de succès (fixés à l'étape 1) :
  OK           Hit@3 >= 0.7
  OK           MRR >= 0.5
  OK           Recherche sémantique >= baseline (MRR)

Rapport : C:\Users\manno\Downloads\UIT-Project-main\rapports\evaluation.json
Historique : C:\Users\manno\Downloads\UIT-Project-main\rapports\historique_experiences.csv

=> Modèle VALIDÉ : on peut passer au DÉPLOIEMENT (étape 6).
(UIT-Project-main) PS C:\Users\manno\Downloads\UIT-Project-main> 