"""
agent.py — L'AGENT et sa boucle agentique.
==========================================

Un LLM seul produit du texte. Un AGENT est un LLM à qui l'on donne des OUTILS.

  1. On décrit les outils au LLM (outils.py).
  2. Le LLM lit la question et DÉCIDE s'il a besoin d'un outil.
  3. Si oui, il ne répond pas : il renvoie une demande d'appel (nom + arguments).
  4. NOTRE code exécute la fonction Python et renvoie le résultat (rôle « tool »).
  5. Le LLM lit le résultat, puis rappelle un outil OU rédige la réponse finale.

     Utilisateur ──► LLM ──appel d'outil──► notre code exécute l'outil
                      ▲                              │
                      └────── résultat (« tool ») ◄──┘
                      │
                      ▼ (plus d'appel d'outil)
                 Réponse finale + sources

GARDE-FOU « recherche de secours » (config.RECHERCHE_DE_SECOURS)
  Constat de l'étape 5 : qwen2.5:3b n'appelait l'outil que pour 1 question factuelle
  sur 3 et répondait « je ne trouve rien »... sans avoir cherché. Ni un prompt plus
  strict, ni un message de rappel, ni un modèle 7b n'ont rendu ce choix fiable.
  Règle d'ingénierie : on ne fait pas une confiance aveugle au LLM. Si le LLM veut
  conclure SANS avoir utilisé aucun outil, NOTRE CODE lance la recherche avec la
  question, puis redonne la main au LLM avec les résultats. L'étape est marquée
  `automatique=True` pour rester visible (interfaces, évaluation).

Différence avec la chaîne RAG classique (étape 4, rag.py) :
  RAG classique : recherche TOUJOURS, UNE fois, avec la question brute.
  Agent RAG     : décide SI il cherche, QUOI chercher (reformulation),
                  peut chercher PLUSIEURS fois et utiliser d'autres outils.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_rag import config
from agent_rag.commun import llm
from agent_rag.commun.llm import AppelOutil, ReponseLLM
from agent_rag.etape6_deploiement.outils import BoiteAOutils
from agent_rag.etape6_deploiement.prompts import CONSIGNES_AGENT

OUTIL_DE_SECOURS = "rechercher_actualites"

# Messages purement conversationnels : pas de recherche de secours pour eux.
# Une règle simple et lisible (plutôt qu'un classifieur LLM : qwen2.5:3b s'est
# trompé sur 3 messages sur 10 lors des tests).
FORMULES_CONVERSATION = ("bonjour", "bonsoir", "salut", "coucou", "hello", "merci",
                         "au revoir", "qui es-tu", "qui es tu", "que sais-tu faire",
                         "que sais tu faire", "comment ça va", "ça va")


def est_conversationnel(message: str) -> bool:
    """Vrai pour « Bonjour ! », « Merci beaucoup », « Qui es-tu ? »...
    Faux dès que le message est long (« Bonjour, quel est le score du match ? »)."""
    texte = message.lower().strip(" !?.,")
    return len(texte.split()) <= 5 and any(f in texte for f in FORMULES_CONVERSATION)


@dataclass
class EtapeOutil:
    """Une étape visible : quel outil, quels arguments, quel résultat, décidée par qui."""
    nom: str
    arguments: dict
    resultat: Any
    automatique: bool = False   # True = lancée par le garde-fou, pas par le LLM


@dataclass
class ReponseAgent:
    texte: str
    etapes: list[EtapeOutil] = field(default_factory=list)

    @property
    def sources(self) -> list[dict]:
        """Les documents consultés (sans doublon), d'après les résultats d'outils."""
        vues, sources = set(), []
        for etape in self.etapes:
            if isinstance(etape.resultat, list):
                for doc in etape.resultat:
                    if isinstance(doc, dict) and doc.get("lien") and doc["lien"] not in vues:
                        vues.add(doc["lien"])
                        sources.append({k: doc.get(k) for k in ("titre", "source", "lien", "date")})
        return sources


class Agent:
    """
    Utilisation :
        agent = Agent(creer_outils(IndexVectoriel.charger()))
        reponse = agent.repondre("Quoi de neuf sur le rugby ?")
        reponse.texte, reponse.etapes, reponse.sources
        agent.repondre("Et sur l'Ukraine ?")     # l'agent se souvient du contexte
    """

    def __init__(self, outils: BoiteAOutils,
                 appeler_llm: Callable = llm.appeler_llm,   # injectable (tests, autre LLM)
                 consignes: str = CONSIGNES_AGENT,
                 max_tours: int = config.MAX_TOURS_AGENT,
                 recherche_de_secours: bool = config.RECHERCHE_DE_SECOURS):
        self.outils = outils
        self.appeler_llm = appeler_llm
        self.consignes = consignes
        self.max_tours = max_tours
        self.recherche_de_secours = recherche_de_secours and OUTIL_DE_SECOURS in outils.outils
        self.reinitialiser()

    def reinitialiser(self) -> None:
        """Nouvelle conversation : on repart des seules consignes (rôle « system »)."""
        self.historique: list[dict] = [{"role": "system", "content": self.consignes}]

    def repondre(self, question: str) -> ReponseAgent:
        """Fait tourner la boucle agentique jusqu'à la réponse finale."""
        taille_avant = len(self.historique)
        self.historique.append({"role": "user", "content": question})
        try:
            return self._boucle(question)
        except Exception:
            # Tour incomplet (ex. : Ollama arrêté) : on l'efface pour garder
            # un historique valide, puis on laisse remonter l'erreur.
            del self.historique[taille_avant:]
            raise

    def _boucle(self, question: str) -> ReponseAgent:
        etapes: list[EtapeOutil] = []

        for _ in range(self.max_tours):
            reponse = self.appeler_llm(self.historique, outils=self.outils.schemas())

            # GARDE-FOU : le LLM conclut sans avoir utilisé AUCUN outil pendant ce tour.
            # Sa réponse prématurée est écartée ; notre code cherche à sa place.
            if not reponse.appels_outils and not etapes and self.recherche_de_secours \
                    and not est_conversationnel(question):
                reponse = ReponseLLM(texte="", appels_outils=[
                    AppelOutil(OUTIL_DE_SECOURS, {"requete": question})])
                automatique = True
            else:
                automatique = False

            # On garde TOUJOURS la réponse du LLM, y compris ses demandes d'outils
            self.historique.append(reponse.message_assistant())

            # Cas 1 : aucune demande d'outil -> c'est la réponse finale
            if not reponse.appels_outils:
                return ReponseAgent(texte=reponse.texte, etapes=etapes)

            # Cas 2 : un ou plusieurs outils à exécuter
            for appel in reponse.appels_outils:
                resultat = self.outils.executer(appel.nom, appel.arguments)
                etapes.append(EtapeOutil(appel.nom, appel.arguments, resultat, automatique))
                self.historique.append({
                    "role": "tool",
                    "content": BoiteAOutils.en_texte(resultat),
                    "tool_name": appel.nom,
                })
            # ... puis on repart pour un tour : le LLM va lire les résultats

        return ReponseAgent(
            texte="(Nombre maximal d'étapes atteint : je n'ai pas pu conclure.)",
            etapes=etapes,
        )


def creer_agent(index=None) -> Agent:
    """Fabrique l'agent complet (par défaut à partir de l'index de l'étape 4)."""
    from agent_rag.etape4_modelisation.index_vectoriel import IndexVectoriel
    from agent_rag.etape6_deploiement.outils import creer_outils
    return Agent(creer_outils(index or IndexVectoriel.charger()))
