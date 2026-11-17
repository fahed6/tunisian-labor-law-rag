"""
outils.py — Les OUTILS de l'agent.

Un outil = une fonction Python + une DESCRIPTION.
Le LLM ne lit jamais le code : il ne voit QUE le nom, la description et les
paramètres (au format JSON Schema). Une description claire = le LLM sait
QUAND et COMMENT utiliser l'outil.

Pour ajouter un outil : écrire la fonction, puis l'ajouter dans `creer_outils()`.
Rien d'autre à modifier (l'agent découvre les outils automatiquement).
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agent_rag import config
from agent_rag.etape4_modelisation.index_vectoriel import IndexVectoriel

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


@dataclass
class Outil:
    nom: str
    description: str
    parametres: dict                 # JSON Schema des paramètres
    fonction: Callable[..., Any]     # la fonction Python réellement exécutée

    def schema(self) -> dict:
        """Ce que le LLM « voit » de l'outil."""
        return {
            "type": "function",
            "function": {"name": self.nom, "description": self.description,
                         "parameters": self.parametres},
        }


class BoiteAOutils:
    """Regroupe les outils et les exécute à la demande du LLM."""

    def __init__(self, outils: list[Outil]):
        self.outils = {o.nom: o for o in outils}

    def schemas(self) -> list[dict]:
        return [o.schema() for o in self.outils.values()]

    def executer(self, nom: str, arguments: dict) -> Any:
        """Exécute un outil. En cas d'erreur, on RENVOIE l'erreur au LLM
        (au lieu de planter) : il pourra corriger et réessayer."""
        outil = self.outils.get(nom)
        if outil is None:
            return {"erreur": f"L'outil « {nom} » n'existe pas. "
                              f"Outils disponibles : {list(self.outils)}"}
        try:
            return outil.fonction(**arguments)   # {"a": 1} -> fonction(a=1)
        except Exception as erreur:
            return {"erreur": f"{type(erreur).__name__} : {erreur}"}

    @staticmethod
    def en_texte(resultat: Any) -> str:
        """Le LLM reçoit du texte : on sérialise en JSON (structuré, lisible)."""
        return resultat if isinstance(resultat, str) else json.dumps(resultat, ensure_ascii=False)


# ==========================================================================
# Les outils de notre agent d'actualité
# ==========================================================================
def creer_outils(index: IndexVectoriel) -> BoiteAOutils:

    def rechercher_actualites(requete: str, nombre_resultats: int = config.NB_RESULTATS) -> list[dict]:
        """OUTIL PRINCIPAL = le « R » du RAG, mis à disposition de l'agent."""
        # Les petits modèles envoient parfois "5" (texte) : on convertit et on borne
        k = max(1, min(int(nombre_resultats), 10))
        return [
            {"titre": r["titre"], "source": r["source"], "date": r["date"],
             "lien": r["lien"], "extrait": r["texte"], "score": r["score"]}
            for r in index.rechercher(str(requete), k=k)
        ]

    def obtenir_date_du_jour() -> str:
        """Le LLM ne connaît pas la date du jour : on la lui donne."""
        maintenant = datetime.now()
        return f"{JOURS[maintenant.weekday()]} {maintenant:%d/%m/%Y, %H:%M}"

    return BoiteAOutils([
        Outil(
            nom="rechercher_actualites",
            description=(
                "Recherche dans une base d'articles d'actualité récents (médias francophones) "
                "les passages les plus pertinents. Renvoie titre, source, date, lien et extrait. "
                "À utiliser pour TOUTE question sur l'actualité ou des événements récents."
            ),
            parametres={
                "type": "object",
                "properties": {
                    "requete": {"type": "string",
                                "description": "Mots-clés à chercher, ex. : 'inflation zone euro'"},
                    "nombre_resultats": {"type": "integer",
                                         "description": f"Entre 1 et 10 ({config.NB_RESULTATS} conseillé)"},
                },
                "required": ["requete"],
            },
            fonction=rechercher_actualites,
        ),
        Outil(
            nom="obtenir_date_du_jour",
            description=("Renvoie la date et l'heure actuelles. À utiliser quand la question "
                         "dépend de la date (« aujourd'hui », « cette semaine »)."),
            parametres={"type": "object", "properties": {}},
            fonction=obtenir_date_du_jour,
        ),
    ])
