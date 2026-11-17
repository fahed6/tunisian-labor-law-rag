"""
rag.py — La chaîne RAG « classique » : Retrieve -> Augment -> Generate.

    question ──► [R] index.rechercher() ──► k passages
                                              │
             [A] consigne = règles + passages + question
                                              │
             [G] LLM ──► réponse + sources

Ici la recherche est TOUJOURS faite, une seule fois, avec la question telle
quelle. C'est simple et prévisible. Ses limites (« bonjour » déclenche une
recherche inutile, pas de reformulation, pas de 2e recherche) motivent
l'AGENT de l'étape 6, qui DÉCIDE quand et quoi chercher.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from agent_rag import config
from agent_rag.commun import llm

CONSIGNE_RAG = """Tu es un assistant d'information sur l'actualité. Réponds en français.
Réponds UNIQUEMENT à partir des extraits numérotés ci-dessous.
Si les extraits ne permettent pas de répondre, dis-le honnêtement : n'invente rien.
Sois concis (quelques phrases), puis termine par la liste des sources utilisées
sous la forme :  - Titre (Source) : URL complète du lien

EXTRAITS :
{passages}

QUESTION : {question}"""


@dataclass
class ReponseRAG:
    texte: str
    passages: list[dict] = field(default_factory=list)   # ce que la recherche a renvoyé


def formater_passages(passages: list[dict]) -> str:
    """[A] Met les passages trouvés dans un format lisible par le LLM."""
    return "\n\n".join(
        f"[{i}] {p['titre']} ({p['source']}, {p['date']})\n"
        f"Lien : {p['lien']}\nExtrait : {p['texte']}"
        for i, p in enumerate(passages, start=1)
    )


class ChaineRAG:
    def __init__(self, rechercher: Callable[[str, int], list[dict]],
                 appeler_llm: Callable = llm.appeler_llm,
                 k: int = config.NB_RESULTATS):
        self.rechercher = rechercher     # ex. : index.rechercher
        self.appeler_llm = appeler_llm   # injectable : un faux LLM pour les tests
        self.k = k

    def repondre(self, question: str) -> ReponseRAG:
        passages = self.rechercher(question, self.k)                                 # R
        consigne = CONSIGNE_RAG.format(passages=formater_passages(passages),
                                       question=question)                             # A
        reponse = self.appeler_llm([{"role": "user", "content": consigne}])         # G
        return ReponseRAG(texte=reponse.texte, passages=passages)
