from agent_rag.commun.llm import ReponseLLM
from agent_rag.etape4_modelisation.rag import ChaineRAG


def test_recherche_trouve_le_document_le_plus_proche(index):
    resultats = index.rechercher("Qui a gagné le match de rugby ?", k=2)
    assert resultats[0]["id_article"] == "a1"
    assert resultats[0]["score"] >= resultats[1]["score"]


def test_recherche_respecte_k(index):
    assert len(index.rechercher("prix", k=3)) == 3


def test_chaine_rag_injecte_les_passages_dans_la_consigne(index):
    consignes_recues = []

    def faux_llm(messages, outils=None):
        consignes_recues.append(messages[0]["content"])
        return ReponseLLM(texte="Réponse")

    reponse = ChaineRAG(index.rechercher, appeler_llm=faux_llm, k=1).repondre("inflation et prix")
    assert reponse.passages[0]["id_article"] == "a2"
    assert "L'inflation fait grimper les prix" in consignes_recues[0]    # Augment
    assert "QUESTION : inflation et prix" in consignes_recues[0]
