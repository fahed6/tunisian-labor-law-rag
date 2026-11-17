# Étape 1 — Compréhension du métier

**Projet :** assistant d'information sur le Code du travail tunisien (RAG + agent)



---

## 1. Le problème

### Qui est l'utilisateur ?

Un **salarié** ou un **employeur** en Tunisie, sans formation juridique, qui rencontre
un problème de travail et veut savoir **ce que dit la loi** avant de consulter un
avocat, l'inspection du travail ou un syndicat.

### Quelles questions pose-t-il ? (exemples réels visés)

| Côté | Question type |
|---|---|
| Salarié | Combien de temps peut durer ma période d'essai ? |
| Salarié | Mon employeur peut-il me licencier sans préavis ? |
| Salarié | Que puis-je faire si mon licenciement est abusif ? |
| Salarié | À combien de jours de congé annuel payé ai-je droit ? |
| Salarié | Comment saisir le conseil de prud'hommes ? |
| Salarié | Ai-je droit à un certificat de travail quand je pars ? |
| Employeur | Dans quels cas puis-je conclure un contrat à durée déterminée ? |
| Employeur | Que dois-je faire avant un licenciement pour raison économique ? |
| Employeur | Que risque-t-on en payant un salaire inférieur au minimum ? |
| Les deux | Comment se règle un conflit individuel de travail ? |

### Ce qui est hors périmètre (l'agent doit répondre « je ne sais pas »)

Droit d'un autre pays, droit de la famille, fiscalité, sécurité sociale, montant actuel
du salaire minimum, harcèlement (le texte utilisé n'en parle pas), et tout **conseil
personnalisé** sur un dossier particulier.

---

## 2. Pourquoi un LLM seul ne suffit pas

| Limite | Conséquence pour le droit du travail |
|---|---|
| **Hallucination** | Le modèle peut inventer un numéro d'article ou un délai de préavis qui « sonne juste ». |
| **Mélange de pays** | Il connaît surtout le droit français : il risque de répondre avec des règles qui ne sont pas tunisiennes. |
| **Pas de source** | L'utilisateur ne peut pas vérifier : or, en droit, une réponse sans article est inutilisable. |
| **Connaissance figée** | Le texte a été modifié à plusieurs reprises ; le modèle ne sait pas quelle version il a vue. |

### Exemple de mauvaise réponse *(à compléter avec un vrai test)*

Lancer, **sans RAG** : `ollama run qwen2.5:3b "Quel est le préavis en cas de licenciement en Tunisie ?"`

> *[Coller ici la réponse réelle du modèle.]*
>
> **Problèmes à relever :** aucun article cité ? chiffre non vérifiable ? règle française ?
> *(Garder cette réponse : elle sera comparée à celle de l'agent à l'étape 6.)*

---

## 3. La solution : un agent RAG avec des outils

```
                         ┌─────────────────────────────┐
  Question  ───────────► │           AGENT (LLM)       │ ◄── system prompt : répondre
  de l'utilisateur       │  décide quel outil appeler  │     d'après les sources, citer
                         └──────────┬──────────────────┘     l'article, ne pas inventer
                                    │
              ┌─────────────────────┴───────────────────────┐
              ▼                                             ▼
   ┌────────────────────────┐                 ┌─────────────────────────────┐
   │ outil 1 : rechercher_  │                 │ outil 2 : rechercher_       │
   │ code_du_travail(question)                │ article(numero)             │
   │  = RAG : embeddings +  │                 │  = lecture directe d'un     │
   │  index → top-k articles│                 │  article (dit s'il est abrogé)
   └───────────┬────────────┘                 └──────────────┬──────────────┘
               └──────────────────┬──────────────────────────┘
                                  ▼
                 Base : 484 articles en vigueur (Code du travail, éd. 2010)
                 chaque extrait porte : n° d'article, livre, chapitre, page
                                  │
                                  ▼
           Réponse = texte d'après les extraits + « Source : Article N » + avertissement
```

- **Retrieve** : trouver les articles proches de la question (recherche sémantique).
- **Augment** : mettre ces articles dans le prompt.
- **Generate** : le LLM rédige uniquement à partir d'eux, en citant les numéros.
- **Agent** : le LLM *choisit* d'appeler l'outil 1 (question de fond), l'outil 2 (« que dit l'article 23 ? »)
  ou aucun outil (« bonjour »).

---

## 4. Critères de succès chiffrés

| # | Type | Critère | Mesure | Seuil |
|---|---|---|---|---|
| C1 | Data mining | Le bon article est retrouvé | **Hit@3** sur le jeu de test (≥ 15 questions) | **≥ 0,70** |
| C2 | Data mining | Le bon article est bien classé | **MRR** | **≥ 0,50** |
| C3 | Data mining | Mieux qu'une méthode naïve | Hit@3 du RAG − Hit@3 de la baseline (mots communs) | **≥ +0,10** |
| C4 | Métier | Les réponses citent leur source | Part des réponses citant au moins un n° d'article | **≥ 0,90** |
| C5 | Métier | Les réponses sont fidèles aux extraits | Part jugée « fidèle » (LLM-juge) | **≥ 0,70** |
| C6 | Métier | L'agent avoue ne pas savoir | Part des questions **hors base** (≥ 5) où il refuse de répondre | **≥ 0,80** |
| C7 | Métier | Pas de conseil juridique déguisé | Part des réponses de fond contenant l'avertissement « information, pas un avis juridique » | **≥ 0,95** |


---

## 5. Contraintes et risques

| Domaine | Risque | Mesure prévue |
|---|---|---|
| **Droits** | Source = texte de loi publié (Publications officielles, éd. 2010). Vérifier les conditions de réutilisation du site d'hébergement. | Citer la source et l'édition ; ne redistribuer que pour usage pédagogique. |
| **Fraîcheur** | L'édition date de 2010 ; la dernière modification qu'elle contient date de **2007** ; des textes plus récents existent (ex. une loi sur les contrats de travail et la sous-traitance publiée au Journal officiel en septembre 2025). | Afficher l'édition dans chaque réponse ; l'indiquer comme limite principale ; l'agent doit renvoyer vers les textes à jour. |
| **Erreur à enjeu** | Une mauvaise réponse peut léser quelqu'un (délai de recours, préavis). | Réponse = extrait + article cité ; refus si rien de pertinent ; avertissement systématique. |
| **Conseil juridique** | L'outil ne doit pas se substituer à un avocat. | System prompt + critère C7. |
| **Biais / couverture** | Seul le Code est couvert : pas de jurisprudence, ni de conventions collectives, ni de textes d'application. | Documenter le périmètre (section 1). |
| **Confidentialité** | L'utilisateur peut raconter son cas personnel. | Aucune donnée personnelle stockée ni journalisée ; avertir de ne pas en saisir. |
| **Coût** | LLM et embeddings **locaux** (Ollama, sentence-transformers) : gratuit, mais lent sans GPU (~2 min/réponse). | Modèle 3B par défaut ; juger sur peu de questions. |
| **Qualité des données** | PDF : coupures de mots, articles abrogés vides (35), titres de chapitre parfois tronqués. | Nettoyage et sélection à l'étape 3 ; chiffres avant/après. |
| **Langue** | Texte en français ; questions possibles en arabe. | Périmètre : **français uniquement** (limite assumée). |

---

## 6. Résumé

Un salarié ou un employeur tunisien pose une question de droit du travail en français.
L'agent retrouve les articles pertinents du Code du travail, répond **uniquement** à
partir d'eux, **cite le numéro d'article**, rappelle que ce n'est pas un avis juridique
et **dit qu'il ne sait pas** quand le Code ne répond pas. Le succès est mesuré par
7 critères chiffrés (section 4), vérifiés automatiquement à l'étape 5.
