"""
api.py — L'agent exposé en API REST avec FastAPI (étape 6 : déploiement).
=========================================================================

Une API permet à N'IMPORTE QUEL logiciel (site web, application mobile,
chatbot Teams...) d'utiliser l'agent en envoyant des requêtes HTTP.

    GET    /sante                 l'API, l'index et le LLM sont-ils prêts ?
    POST   /rechercher            le « R » seul : passages les plus proches
    POST   /rag                   chaîne RAG classique (recherche systématique)
    POST   /agent                 l'agent (avec mémoire via session_id)
    DELETE /agent/{session_id}    oublier une conversation

Lancer :  python main.py api      puis ouvrir http://localhost:8000/docs
          (FastAPI génère une documentation interactive pour tester l'API)
"""

import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # sans installation

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agent_rag import config  # noqa: E402
from agent_rag.commun import llm  # noqa: E402
from agent_rag.etape4_modelisation.index_vectoriel import IndexVectoriel  # noqa: E402
from agent_rag.etape4_modelisation.rag import ChaineRAG  # noqa: E402
from agent_rag.etape6_deploiement.agent import Agent  # noqa: E402
from agent_rag.etape6_deploiement.outils import creer_outils  # noqa: E402

etat: dict = {}                  # objets chargés une fois au démarrage
sessions: dict[str, Agent] = {}  # une conversation (= un agent avec sa mémoire) par session


@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    """Au démarrage : charger l'index et le modèle d'embedding UNE seule fois."""
    index = IndexVectoriel.charger()
    index.rechercher("préchauffage", k=1)   # charge le modèle d'embedding maintenant
    etat["index"] = index
    etat["outils"] = creer_outils(index)
    yield
    sessions.clear()


app = FastAPI(title="Agent d'actualité RAG", version="1.0.0", lifespan=cycle_de_vie,
              description="Projet pédagogique CRISP-DM : RAG + agent avec outils.")


# ------------------------------------------------------------------ schémas
# Pydantic valide automatiquement les données reçues et renvoyées.
class RequeteRecherche(BaseModel):
    requete: str = Field(..., min_length=1, examples=["match de rugby"])
    k: int = Field(config.NB_RESULTATS, ge=1, le=10)


class Question(BaseModel):
    question: str = Field(..., min_length=1, examples=["Quelles sont les nouvelles économiques ?"])
    session_id: str | None = Field(None, description="Pour une question de suivi (mémoire)")


class Source(BaseModel):
    titre: str | None
    source: str | None
    lien: str | None
    date: str | None = None


class EtapeOutilAPI(BaseModel):
    outil: str
    arguments: dict
    automatique: bool = Field(False, description="True = recherche de secours (garde-fou)")


class ReponseAgentAPI(BaseModel):
    session_id: str
    reponse: str
    etapes: list[EtapeOutilAPI]
    sources: list[Source]


# ------------------------------------------------------------------ routes
@app.get("/sante")
def sante():
    disponible, message = llm.verifier_disponibilite()
    return {"api": "ok", "passages_indexes": len(etat["index"].chunks),
            "llm_disponible": disponible, "llm": message}


@app.post("/rechercher")
def rechercher(corps: RequeteRecherche):
    return etat["index"].rechercher(corps.requete, k=corps.k)


@app.post("/rag")
def rag(corps: Question):
    reponse = ChaineRAG(etat["index"].rechercher).repondre(corps.question)
    return {"reponse": reponse.texte,
            "sources": [{k: p[k] for k in ("titre", "source", "lien", "date")} for p in reponse.passages]}


@app.post("/agent", response_model=ReponseAgentAPI)
def agent(corps: Question):
    session_id = corps.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = Agent(etat["outils"])
    try:
        reponse = sessions[session_id].repondre(corps.question)
    except Exception as erreur:
        raise HTTPException(status_code=503, detail=f"LLM indisponible : {erreur}") from erreur
    return ReponseAgentAPI(
        session_id=session_id,
        reponse=reponse.texte,
        etapes=[EtapeOutilAPI(outil=e.nom, arguments=e.arguments, automatique=e.automatique)
                for e in reponse.etapes],
        sources=[Source(**s) for s in reponse.sources],
    )


@app.delete("/agent/{session_id}")
def oublier(session_id: str):
    sessions.pop(session_id, None)
    return {"session_id": session_id, "supprimee": True}
