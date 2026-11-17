"""
On teste la BOUCLE AGENTIQUE avec un faux LLM qui joue un scénario écrit
à l'avance : c'est notre code (pas le LLM) qu'on vérifie.
"""

import pytest

from agent_rag.commun.llm import AppelOutil, ReponseLLM
from agent_rag.etape6_deploiement.agent import Agent, est_conversationnel
from agent_rag.etape6_deploiement.outils import creer_outils


class FauxLLM:
    """Renvoie les réponses prévues, une par appel, et mémorise ce qu'il reçoit."""

    def __init__(self, scenario):
        self.scenario = list(scenario)
        self.appels = []

    def __call__(self, messages, outils=None):
        self.appels.append([dict(m) for m in messages])
        return self.scenario.pop(0)


def test_agent_repond_sans_outil_quand_inutile(index):
    llm = FauxLLM([ReponseLLM(texte="Bonjour !")])
    agent = Agent(creer_outils(index), appeler_llm=llm, recherche_de_secours=False)
    reponse = agent.repondre("Salut")
    assert reponse.texte == "Bonjour !"
    assert reponse.etapes == []


def test_garde_fou_recherche_de_secours(index):
    llm = FauxLLM([
        ReponseLLM(texte="Je ne trouve rien."),          # conclut sans avoir cherché...
        ReponseLLM(texte="Le XV de France a gagné."),    # ...répond après la recherche de secours
    ])
    agent = Agent(creer_outils(index), appeler_llm=llm)
    reponse = agent.repondre("Quoi de neuf en rugby ?")

    assert reponse.texte == "Le XV de France a gagné."
    etape = reponse.etapes[0]
    assert (etape.nom, etape.arguments, etape.automatique) ==         ("rechercher_actualites", {"requete": "Quoi de neuf en rugby ?"}, True)
    assert llm.appels[1][-1]["role"] == "tool"                  # le LLM a reçu les résultats
    assert "Je ne trouve rien." not in [m["content"] for m in agent.historique]


def test_pas_de_secours_pour_une_salutation(index):
    llm = FauxLLM([ReponseLLM(texte="Bonjour ! Je réponds sur l'actualité.")])
    reponse = Agent(creer_outils(index), appeler_llm=llm).repondre("Bonjour, que sais-tu faire ?")
    assert reponse.etapes == []


@pytest.mark.parametrize("message, attendu", [
    ("Bonjour !", True), ("Merci beaucoup", True), ("Qui es-tu ?", True),
    ("Bonjour, quel est le score du match France-Turquie ?", False),
    ("Quoi de neuf sur l'inflation ?", False),
])
def test_est_conversationnel(message, attendu):
    assert est_conversationnel(message) is attendu


def test_pas_de_secours_si_le_llm_a_deja_cherche(index):
    llm = FauxLLM([
        ReponseLLM(texte="", appels_outils=[AppelOutil("obtenir_date_du_jour", {})]),
        ReponseLLM(texte="Nous sommes lundi."),
    ])
    reponse = Agent(creer_outils(index), appeler_llm=llm).repondre("Quel jour ?")
    assert [e.automatique for e in reponse.etapes] == [False]


def test_agent_appelle_l_outil_puis_repond(index):
    llm = FauxLLM([
        ReponseLLM(texte="", appels_outils=[AppelOutil("rechercher_actualites",
                                                       {"requete": "rugby", "nombre_resultats": "2"})]),
        ReponseLLM(texte="Le XV de France a gagné. Source : https://exemple.fr/1"),
    ])
    agent = Agent(creer_outils(index), appeler_llm=llm)
    reponse = agent.repondre("Quoi de neuf en rugby ?")

    assert [e.nom for e in reponse.etapes] == ["rechercher_actualites"]
    assert len(reponse.etapes[0].resultat) == 2                     # "2" converti en entier
    assert reponse.sources[0]["lien"] == "https://exemple.fr/1"
    # Au 2e appel, le LLM a bien reçu le résultat de l'outil (rôle « tool »)
    assert llm.appels[1][-1]["role"] == "tool"
    assert "XV de France" in llm.appels[1][-1]["content"]


def test_agent_garde_la_memoire_entre_les_questions(index):
    llm = FauxLLM([ReponseLLM(texte="R1"), ReponseLLM(texte="R2")])
    agent = Agent(creer_outils(index), appeler_llm=llm, recherche_de_secours=False)
    agent.repondre("Q1")
    agent.repondre("Q2")
    contenus = [m["content"] for m in llm.appels[1]]
    assert contenus[1:] == ["Q1", "R1", "Q2"]                      # après le system prompt


def test_outil_inconnu_renvoie_une_erreur_au_llm(index):
    resultat = creer_outils(index).executer("outil_imaginaire", {})
    assert "erreur" in resultat


def test_garde_fou_nombre_de_tours(index):
    boucle_infinie = [ReponseLLM(texte="", appels_outils=[AppelOutil("obtenir_date_du_jour", {})])] * 10
    agent = Agent(creer_outils(index), appeler_llm=FauxLLM(boucle_infinie), max_tours=3)
    reponse = agent.repondre("?")
    assert len(reponse.etapes) == 3
    assert "maximal" in reponse.texte


def test_historique_nettoye_si_le_llm_plante(index):
    def llm_en_panne(messages, outils=None):
        raise ConnectionError("Ollama arrêté")

    agent = Agent(creer_outils(index), appeler_llm=llm_en_panne)
    with pytest.raises(ConnectionError):
        agent.repondre("Q")
    assert len(agent.historique) == 1                              # seul le system prompt reste
