"""
index_vectoriel.py — La « base de données vectorielle » (le « R » de RAG).

Volontairement minimaliste pour être compris en entier :
    - une matrice NumPy, une ligne par chunk     -> data/index/vecteurs.npy
    - les chunks (texte + métadonnées) en JSON   -> data/index/metadonnees.json
    La ligne i de la matrice correspond au chunk i.

En production on utiliserait Chroma, FAISS, Qdrant ou pgvector : ils font
la même chose, à grande échelle. Comme le reste du projet n'utilise que
`IndexVectoriel.rechercher()`, on peut changer de technologie sans rien casser.
"""

from collections.abc import Callable

import numpy as np

from agent_rag import config
from agent_rag.commun.fichiers import ecrire_json, exiger, lire_json
from agent_rag.etape4_modelisation.embeddings import vectoriser

FonctionEmbedding = Callable[[list[str]], np.ndarray]


class IndexVectoriel:
    """
    Utilisation :
        index = IndexVectoriel.construire(chunks)    # étape 4, une fois
        index.sauvegarder()
        index = IndexVectoriel.charger()              # étapes 5 et 6
        index.rechercher("inflation en Europe", k=3)
    """

    def __init__(self, vecteurs: np.ndarray, chunks: list[dict],
                 fonction_embedding: FonctionEmbedding = vectoriser):
        if len(vecteurs) != len(chunks):
            raise ValueError("Il faut autant de vecteurs que de chunks")
        self.vecteurs = vecteurs
        self.chunks = chunks
        # Injection de dépendance : les tests peuvent fournir un faux modèle
        self.fonction_embedding = fonction_embedding

    # ---------------------------------------------------------------- création
    @classmethod
    def construire(cls, chunks: list[dict],
                   fonction_embedding: FonctionEmbedding = vectoriser) -> "IndexVectoriel":
        """Vectorise tous les chunks."""
        vecteurs = fonction_embedding([c["texte"] for c in chunks])
        return cls(vecteurs, chunks, fonction_embedding)

    def sauvegarder(self) -> None:
        config.DOSSIER_INDEX.mkdir(parents=True, exist_ok=True)
        np.save(config.FICHIER_VECTEURS, self.vecteurs)
        ecrire_json(self.chunks, config.FICHIER_METADONNEES)

    @classmethod
    def charger(cls) -> "IndexVectoriel":
        exiger(config.FICHIER_VECTEURS, "modelisation")
        return cls(np.load(config.FICHIER_VECTEURS), lire_json(config.FICHIER_METADONNEES))

    # ---------------------------------------------------------------- recherche
    @property
    def dimensions(self) -> tuple[int, int]:
        return self.vecteurs.shape

    def rechercher(self, question: str, k: int = config.NB_RESULTATS) -> list[dict]:
        """Renvoie les k chunks les plus proches de la question, avec leur score."""
        if not self.chunks:
            return []
        # 1) La question passe par LE MÊME modèle que les documents
        vecteur_question = self.fonction_embedding([question])[0]       # (384,)
        # 2) Similarité cosinus avec TOUS les chunks d'un coup : (N,384)@(384,) -> (N,)
        scores = self.vecteurs @ vecteur_question
        # 3) Tri décroissant, on garde les k meilleurs
        meilleurs = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": round(float(scores[i]), 3)} for i in meilleurs]
