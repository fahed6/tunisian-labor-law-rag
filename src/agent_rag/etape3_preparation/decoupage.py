"""
decoupage.py — 3.5 Formatage : découpage en « chunks ».

Pourquoi découper ? Un modèle d'embedding résume UN texte en UN vecteur :
un texte trop long mélange plusieurs sujets et donne un vecteur « flou ».
Des morceaux de taille raisonnable donnent une recherche plus précise.
"""

from agent_rag import config


def decouper_en_chunks(texte: str, taille: int, chevauchement: int) -> list[str]:
    """
    Découpe un texte en morceaux de `taille` mots qui se chevauchent.

    Exemple avec taille=6 et chevauchement=2 :
      texte  : m1 m2 m3 m4 m5 m6 m7 m8 m9 m10
      chunk 1: m1 m2 m3 m4 m5 m6
      chunk 2:             m5 m6 m7 m8 m9 m10
                           ^^^^^ chevauchement

    Le chevauchement évite qu'une idée coupée en deux soit perdue.
    """
    if not 0 <= chevauchement < taille:
        raise ValueError("Il faut 0 <= chevauchement < taille")

    mots = texte.split()
    if len(mots) <= taille:
        return [texte]

    pas = taille - chevauchement
    chunks = []
    for debut in range(0, len(mots), pas):
        chunks.append(" ".join(mots[debut: debut + taille]))
        if debut + taille >= len(mots):
            break
    return chunks


def creer_chunks(articles: list[dict],
                 taille: int = config.TAILLE_CHUNK,
                 chevauchement: int = config.CHEVAUCHEMENT) -> list[dict]:
    """Chaque article -> un ou plusieurs chunks, chacun avec ses MÉTADONNÉES.

    * Un article qui tient dans `taille` mots reste ENTIER (un seul chunk) :
      c'est l'unité naturelle d'un texte de loi.
    * Un article plus long est découpé ; chaque morceau reprend l'EN-TÊTE
      (« Article 14 — Fin du contrat de travail ») pour ne jamais être anonyme.
    * Les métadonnées (titre, source, lien, date + champs du droit) voyagent avec
      chaque chunk : ce sont elles qui permettent à l'agent de CITER SES SOURCES.
    """
    chunks = []
    for article in articles:
        morceaux = decouper_en_chunks(article["texte"], taille, chevauchement)
        en_tete = article.get("en_tete")
        for numero, morceau in enumerate(morceaux):
            chunk = {
                "id_chunk": f"{article['id_article']}_{numero}",
                "id_article": article["id_article"],
                "source": article["source"],
                "titre": article["titre"],
                "lien": article["lien"],
                "date": article["date"],
                "partie": numero + 1,              # « partie 2 sur 3 » d'un article long
                "nb_parties": len(morceaux),
                "texte": f"{en_tete}. {morceau}" if en_tete else morceau,
            }
            chunk.update({c: article[c] for c in config.CHAMPS_METADONNEES if c in article})
            chunks.append(chunk)
    return chunks
