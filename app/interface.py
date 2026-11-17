"""
interface.py — Interface web de démonstration (Streamlit), étape 6.
==================================================================

Un ONGLET PAR ÉTAPE CRISP-DM + le chat avec l'agent. Idéal pour présenter.
Cette page ne contient AUCUNE logique métier : elle appelle les fonctions du
package agent_rag (les mêmes que le terminal et l'API).

Rappel Streamlit : à CHAQUE clic, tout le script est ré-exécuté.
  - st.session_state  garde des valeurs entre deux exécutions (la conversation)
  - st.cache_resource ne charge qu'une fois les objets lourds (index, modèle)

Lancer :  python main.py interface       (ou : streamlit run app/interface.py)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # sans installation

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from agent_rag import config  # noqa: E402
from agent_rag.commun import llm  # noqa: E402
from agent_rag.commun.fichiers import ecrire_json, lire_json  # noqa: E402
from agent_rag.etape2_donnees.exploration import explorer, vers_dataframe  # noqa: E402
from agent_rag.etape3_preparation.decoupage import decouper_en_chunks  # noqa: E402
from agent_rag.etape3_preparation.nettoyage import nettoyer_texte  # noqa: E402

st.set_page_config(page_title="Agent d'actualité RAG", page_icon="📰", layout="wide")


@st.cache_resource(show_spinner="Chargement de l'index et du modèle d'embedding...")
def charger_index():
    from agent_rag.etape4_modelisation.index_vectoriel import IndexVectoriel
    return IndexVectoriel.charger()


def index_present() -> bool:
    if not config.FICHIER_VECTEURS.exists():
        st.info("Index absent : cliquez sur « Lancer le pipeline » dans la barre latérale.")
        return False
    return True


# ==========================================================================
# Barre latérale : configuration + mise à jour des données
# ==========================================================================
st.title("📰 Agent d'actualité RAG — un projet CRISP-DM")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"**LLM :** `{config.MODELE_LLM}` (Ollama, local)")
    st.write(f"**Embedding :** `{config.MODELE_EMBEDDING.split('/')[-1]}`")
    st.write(f"**Chunks :** {config.TAILLE_CHUNK} mots, chevauchement {config.CHEVAUCHEMENT}")
    disponible, message = llm.verifier_disponibilite()
    (st.success if disponible else st.error)(message)

    st.divider()
    st.header("🔄 Mettre à jour l'actualité")
    st.caption("Relance les étapes 2 → 4 avec les articles du jour.")
    if st.button("Lancer le pipeline", width="stretch"):
        from agent_rag.etape2_donnees.collecte import collecter
        from agent_rag.etape3_preparation.etape import preparer
        from agent_rag.etape4_modelisation.index_vectoriel import IndexVectoriel

        with st.status("Pipeline en cours...", expanded=True) as statut:
            st.write("Étape 2 : collecte des flux RSS...")
            articles, origine = collecter(verbeux=False)
            ecrire_json(articles, config.FICHIER_ARTICLES_BRUTS)
            st.write("Étape 3 : nettoyage et découpage...")
            chunks, _ = preparer(articles)
            ecrire_json(chunks, config.FICHIER_CHUNKS)
            st.write("Étape 4 : calcul des embeddings...")
            IndexVectoriel.construire(chunks).sauvegarder()
            charger_index.clear()
            st.session_state.pop("agent", None)
            statut.update(label=f"Terminé : {len(articles)} articles ({origine})", state="complete")

onglets = st.tabs(["1️⃣ Métier", "2️⃣ Données", "3️⃣ Préparation",
                   "4️⃣ Modélisation (RAG)", "5️⃣ Évaluation", "6️⃣ Agent"])

# ==========================================================================
# 1 — Compréhension du métier
# ==========================================================================
with onglets[0]:
    st.markdown(config.FICHIER_DOC_METIER.read_text(encoding="utf-8"))

# ==========================================================================
# 2 — Compréhension des données
# ==========================================================================
with onglets[1]:
    st.header("Étape 2 — Compréhension des données")
    articles_bruts = lire_json(config.FICHIER_ARTICLES_BRUTS)
    if not articles_bruts:
        st.info("Aucune donnée : cliquez sur « Lancer le pipeline ».")
    else:
        stats = explorer(articles_bruts)
        df = vers_dataframe(articles_bruts)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Articles", stats["nb_articles"])
        c2.metric("Sources", len(stats["par_source"]))
        c3.metric("Résumés vides", stats["champs_vides"]["resume"])
        c4.metric("Doublons (titre)", stats["doublons_titre"])

        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Articles par source")
            st.bar_chart(pd.Series(stats["par_source"]))
            st.caption("Une source dominante = un risque de biais dans les réponses.")
        with g2:
            st.subheader("Longueur des résumés (mots)")
            effectifs, bornes = np.histogram(df["nb_mots_resume"].dropna(), bins=15)
            st.bar_chart(pd.Series(effectifs, index=[f"{int(b)}" for b in bornes[:-1]]))
            st.caption(f"À comparer avec la taille d'un chunk : {config.TAILLE_CHUNK} mots.")
        st.write(f"Résumés contenant du HTML : **{stats['resumes_avec_html']}** — "
                 f"période : {stats['date_min']} → {stats['date_max']}")
        st.dataframe(df[["source", "titre", "resume", "date", "lien"]], width="stretch")

# ==========================================================================
# 3 — Préparation des données
# ==========================================================================
with onglets[2]:
    st.header("Étape 3 — Préparation des données")
    st.subheader("Nettoyage : essayez vous-même")
    texte_sale = st.text_input("Texte brut (avec HTML)",
                               "<p>Le prix de l&#39;essence   augmente <b>encore</b>.</p>")
    a, b = st.columns(2)
    a.code(texte_sale, language="html")
    b.code(nettoyer_texte(texte_sale), language="text")

    st.subheader("Découpage en chunks avec chevauchement")
    c1, c2 = st.columns(2)
    taille = c1.slider("Taille d'un chunk (mots)", 3, 12, 6)
    chevauchement = c2.slider("Chevauchement (mots)", 0, taille - 1, 2)
    for i, morceau in enumerate(decouper_en_chunks(" ".join(f"mot{i}" for i in range(1, 21)),
                                                   taille, chevauchement), 1):
        st.text(f"chunk {i} : {morceau}")

    chunks = lire_json(config.FICHIER_CHUNKS)
    if chunks:
        st.subheader(f"Sur les vraies données : {len(chunks)} chunks")
        st.dataframe(pd.DataFrame(chunks)[["id_chunk", "source", "titre", "texte"]], width="stretch")

# ==========================================================================
# 4 — Modélisation : R, puis R + A + G
# ==========================================================================
with onglets[3]:
    st.header("Étape 4 — Modélisation : le RAG")
    if index_present():
        index = charger_index()
        n, d = index.dimensions
        st.write(f"Index : **{n} chunks × {d} dimensions**")

        st.subheader("① Embeddings : le sens compte, pas les mots")
        from agent_rag.etape4_modelisation.embeddings import vectoriser
        phrases = [p for p in st.text_area(
            "Une phrase par ligne (la 1re sert de référence)",
            "La période d'essai est fixée par la loi.\n"
            "Le salarié bénéficie de congés annuels payés.\n"
            "Le licenciement abusif donne droit à réparation.\n"
            "Une recette de couscous tunisien.").split("\n") if p.strip()]
        if phrases:
            vecteurs = vectoriser(phrases)
            st.dataframe(pd.DataFrame({"phrase": phrases,
                                       "similarité cosinus": (vecteurs @ vecteurs[0]).round(3)}),
                         width="stretch")

        st.subheader("② Retrieve : le moteur de recherche")
        requete = st.text_input("Votre recherche", "licenciement abusif")
        k = st.slider("Nombre de résultats (k)", 1, 10, config.NB_RESULTATS)
        if requete:
            st.dataframe(pd.DataFrame(index.rechercher(requete, k=k))[["score", "source", "titre", "texte"]],
                         width="stretch")

        st.subheader("③ Retrieve + Augment + Generate : la chaîne RAG classique")
        question_rag = st.text_input("Question", "Combien de temps dure la période d'essai ?", key="rag")
        if st.button("Générer la réponse (LLM local, ~1 min)", disabled=not disponible):
            from agent_rag.etape4_modelisation.rag import ChaineRAG
            with st.spinner("Recherche puis génération..."):
                r = ChaineRAG(index.rechercher).repondre(question_rag)
            st.markdown(r.texte)
            with st.expander("Passages fournis au LLM"):
                st.dataframe(pd.DataFrame(r.passages)[["score", "titre", "texte"]], width="stretch")

# ==========================================================================
# 5 — Évaluation
# ==========================================================================
with onglets[4]:
    st.header("Étape 5 — Évaluation")
    st.markdown(
        "- **Hit@1 / Hit@k** : % de questions dont le bon article est 1er / dans les k premiers\n"
        "- **MRR** : moyenne de 1/rang (1 = toujours premier)\n"
        "- **Citation / Fidélité** : la réponse cite-t-elle une source ? est-elle appuyée par les extraits ?")
    st.caption("Lancer une évaluation complète : `python main.py evaluation --generation`")

    rapport = lire_json(config.FICHIER_RAPPORT_EVALUATION)
    if not rapport:
        st.info("Aucun rapport : lancez `python main.py evaluation`.")
    else:
        st.write(f"Dernière évaluation : **{rapport['date']}** — {rapport['nb_questions']} questions")
        scores = pd.DataFrame({"Baseline mots-clés": rapport["recherche"]["baseline"],
                               "RAG sémantique": rapport["recherche"]["semantique"]})
        c1, c2 = st.columns(2)
        c1.dataframe(scores, width="stretch")
        c2.bar_chart(scores)

        if rapport.get("generation"):
            st.subheader("Réponses : RAG classique vs Agent")
            st.dataframe(pd.DataFrame(rapport["generation"]["scores"]), width="stretch")

        st.subheader("Critères de succès (étape 1)")
        for nom, ok in rapport["criteres"].items():
            st.write(("✅ " if ok else "❌ ") + nom)

    historique = config.DOSSIER_RAPPORTS / "historique_experiences.csv"
    if historique.exists():
        st.subheader("Historique des expériences (le cycle CRISP-DM)")
        st.dataframe(pd.read_csv(historique), width="stretch")

# ==========================================================================
# 6 — Déploiement : l'agent
# ==========================================================================
with onglets[5]:
    st.header("Étape 6 — Discuter avec l'agent")
    if index_present():
        from agent_rag.etape6_deploiement.agent import creer_agent

        if "agent" not in st.session_state:
            st.session_state.agent = creer_agent(charger_index())
            st.session_state.affichage = []
        if st.button("🗑️ Nouvelle conversation"):
            st.session_state.agent.reinitialiser()
            st.session_state.affichage = []

        def afficher_etapes(etapes):
            for e in etapes:
                with st.expander(f"🔧 Outil : {e['nom']}({e['arguments']})"):
                    st.code(e["resultat"], language="json")

        for bulle in st.session_state.affichage:
            with st.chat_message(bulle["role"]):
                afficher_etapes(bulle.get("etapes", []))
                st.markdown(bulle["texte"])

        question = st.chat_input("Posez une question sur le Code du travail tunisien...", disabled=not disponible)
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.affichage.append({"role": "user", "texte": question})

            with st.chat_message("assistant"):
                with st.spinner("L'agent réfléchit (le LLM local peut être lent)..."):
                    try:
                        reponse = st.session_state.agent.repondre(question)
                        texte = reponse.texte
                        etapes = [{"nom": e.nom + (" — recherche de secours" if e.automatique else ""),
                                   "arguments": json.dumps(e.arguments, ensure_ascii=False),
                                   "resultat": json.dumps(e.resultat, ensure_ascii=False, indent=2)}
                                  for e in reponse.etapes]
                    except Exception as erreur:
                        texte, etapes = f"⚠️ Erreur : {erreur}", []
                afficher_etapes(etapes)
                st.markdown(texte)
            st.session_state.affichage.append({"role": "assistant", "texte": texte, "etapes": etapes})
