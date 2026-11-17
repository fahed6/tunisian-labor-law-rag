"""
conftest.py — Outils partagés par les tests.

Les tests ne doivent dépendre ni d'Internet, ni du modèle d'embedding
(470 Mo), ni d'Ollama : ils doivent être RAPIDES et DÉTERMINISTES.
On remplace donc ces dépendances par des « doublures » (fakes).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_rag.etape4_modelisation.index_vectoriel import IndexVectoriel  # noqa: E402

VOCABULAIRE = ["rugby", "match", "inflation", "prix", "ukraine", "guerre", "climat", "pluie"]


def faux_embedding(textes: list[str]) -> np.ndarray:
    """Un « modèle » jouet : un vecteur = présence des mots du vocabulaire."""
    vecteurs = np.array([[float(m in t.lower()) for m in VOCABULAIRE] for t in textes]) + 1e-6
    return vecteurs / np.linalg.norm(vecteurs, axis=1, keepdims=True)


@pytest.fixture
def chunks():
    def chunk(i, texte):
        return {"id_chunk": f"a{i}_0", "id_article": f"a{i}", "source": "Test",
                "titre": texte[:30], "lien": f"https://exemple.fr/{i}", "date": "", "texte": texte}
    return [
        chunk(1, "Le XV de France gagne le match de rugby"),
        chunk(2, "L'inflation fait grimper les prix en Europe"),
        chunk(3, "La guerre en Ukraine entre dans son hiver"),
        chunk(4, "Le climat change : pluie record en novembre"),
    ]


@pytest.fixture
def index(chunks):
    return IndexVectoriel.construire(chunks, fonction_embedding=faux_embedding)
