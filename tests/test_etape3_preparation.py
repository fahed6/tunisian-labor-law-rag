import pytest

from agent_rag import config
from agent_rag.etape3_preparation.decoupage import creer_chunks, decouper_en_chunks
from agent_rag.etape3_preparation.etape import preparer
from agent_rag.etape3_preparation.nettoyage import (
    construire_en_tete,
    intitule,
    nettoyer_texte,
    preparer_articles,
)


# --------------------------------------------------------------------------
# 3.2 Nettoyage
# --------------------------------------------------------------------------
def test_nettoyage_espaces_et_apostrophes():
    brut = "Le contrat\n  de l\u2019employeur\xa0prend  fin."
    assert nettoyer_texte(brut) == "Le contrat de l'employeur prend fin."


def test_nettoyage_ne_touche_pas_au_contenu_juridique():
    texte = "Les peines prévues à l'article 234 du Code du Travail (1996) s'appliquent."
    assert nettoyer_texte(texte) == texte


def test_nettoyage_texte_vide():
    assert nettoyer_texte(None) == ""


# --------------------------------------------------------------------------
# 3.3 Construction : l'en-tête
# --------------------------------------------------------------------------
def test_intitule_retire_le_prefixe_et_les_majuscules():
    assert intitule("Chapitre III — Fin du contrat de travail") == "Fin du contrat de travail"
    assert intitule("TITRE PREMIER — LE CONTRAT DE TRAVAIL") == "Le contrat de travail"
    assert intitule("") == ""


def test_en_tete_prefere_le_chapitre_puis_la_division():
    article = {"chapitre": "Chapitre III — Fin du contrat de travail",
               "division": "TITRE PREMIER — LE CONTRAT DE TRAVAIL"}
    assert construire_en_tete(article, "Article 14") == "Article 14 — Fin du contrat de travail"
    assert construire_en_tete({"division": "TITRE II — LE SALAIRE"}, "Article 9") == \
        "Article 9 — Le salaire"
    assert construire_en_tete({}, "Article 1") == "Article 1"


# --------------------------------------------------------------------------
# 3.1 / 3.2 / 3.4 Sélection, dédoublonnage, intégration
# --------------------------------------------------------------------------
def brut(titre, resume, lien, **extra):
    return {"source": "S", "titre": titre, "resume": resume, "lien": lien, **extra}


def test_preparation_dedoublonne_et_filtre():
    long_resume = "un texte suffisamment long pour être conservé dans la base"
    bruts = [
        brut("Article 1", long_resume, "l1"),
        brut("article 1", long_resume, "l2"),                       # même titre
        brut("Article 2", long_resume, "l1"),                       # même lien
        brut("Article 3", "trop court", "l3"),                      # trop court
        brut("Article 4", "", "l4", statut=config.STATUT_ABROGE),   # abrogé
    ]
    articles, stats = preparer_articles(bruts)
    assert [a["titre"] for a in articles] == ["Article 1"]
    assert (stats["lus"], stats["doublons"], stats["trop_courts"], stats["abroges"],
            stats["conserves"]) == (5, 2, 1, 1, 1)


def test_titres_ecartes_a_la_main_sont_comptes_a_part():
    resume = "un texte suffisamment long pour être conservé dans la base"
    titre_ecarte = next(iter(config.TITRES_ECARTES))
    articles, stats = preparer_articles([brut(titre_ecarte, resume, "l1"), brut("Article 1", resume, "l2")])
    assert [a["titre"] for a in articles] == ["Article 1"]
    assert stats["ecartes_manuels"] == 1


def test_preparation_construit_en_tete_et_recopie_les_metadonnees():
    resume = "un texte suffisamment long pour être conservé dans la base"
    bruts = [brut("Article 14", resume, "l14", numero="14", page_pdf=19,
                  chapitre="Chapitre III — Fin du contrat de travail",
                  statut=config.STATUT_EN_VIGUEUR)]
    (article,), _ = preparer_articles(bruts)
    assert article["en_tete"] == "Article 14 — Fin du contrat de travail"
    assert article["texte"] == resume                 # le corps reste le texte de loi, sans en-tête
    assert (article["numero"], article["page_pdf"]) == ("14", 19)


def test_statistiques_avant_apres_coherentes():
    resume = "un texte suffisamment long pour être conservé dans la base"
    bruts = [brut("Article 1", resume, "l1"), brut("Article 2", "", "l2", statut=config.STATUT_ABROGE)]
    articles, stats = preparer_articles(bruts)
    assert stats["lus"] == (stats["conserves"] + stats["abroges"] + stats["ecartes_manuels"]
                            + stats["doublons"] + stats["trop_courts"])
    assert stats["mots_apres"] == sum(len(a["texte"].split()) for a in articles)


# --------------------------------------------------------------------------
# 3.5 Découpage
# --------------------------------------------------------------------------
def test_decoupage_avec_chevauchement():
    texte = " ".join(f"m{i}" for i in range(1, 11))
    assert decouper_en_chunks(texte, taille=6, chevauchement=2) == [
        "m1 m2 m3 m4 m5 m6", "m5 m6 m7 m8 m9 m10"]


def test_decoupage_texte_court_un_seul_chunk():
    assert decouper_en_chunks("un deux trois", taille=6, chevauchement=2) == ["un deux trois"]


def test_decoupage_parametres_invalides():
    with pytest.raises(ValueError):
        decouper_en_chunks("a b c", taille=3, chevauchement=3)


def test_chunks_portent_les_metadonnees():
    article = {"id_article": "x", "source": "S", "titre": "T", "lien": "L", "date": "D",
               "texte": " ".join(["mot"] * 10)}
    chunks = creer_chunks([article], taille=6, chevauchement=2)
    assert [c["id_chunk"] for c in chunks] == ["x_0", "x_1"]
    assert all(c["lien"] == "L" and c["source"] == "S" for c in chunks)


def test_article_court_reste_entier_et_article_long_garde_son_en_tete():
    court = {"id_article": "a", "source": "S", "titre": "Article 1", "lien": "L1", "date": "",
             "en_tete": "Article 1 — Sujet", "texte": "un deux trois", "numero": "1"}
    long = {"id_article": "b", "source": "S", "titre": "Article 2", "lien": "L2", "date": "",
            "en_tete": "Article 2 — Sujet", "texte": " ".join(f"m{i}" for i in range(30)),
            "numero": "2", "page_pdf": 7}
    chunks = creer_chunks([court, long], taille=12, chevauchement=2)
    par_article = {}
    for c in chunks:
        par_article.setdefault(c["id_article"], []).append(c)

    assert len(par_article["a"]) == 1 and par_article["a"][0]["nb_parties"] == 1
    assert par_article["a"][0]["texte"] == "Article 1 — Sujet. un deux trois"

    parties = par_article["b"]
    assert len(parties) > 1
    assert [c["partie"] for c in parties] == list(range(1, len(parties) + 1))
    assert all(c["texte"].startswith("Article 2 — Sujet. ") for c in parties)   # chaque morceau garde le contexte
    assert all(c["numero"] == "2" and c["page_pdf"] == 7 for c in parties)      # et ses métadonnées


def test_preparer_de_bout_en_bout():
    resume = " ".join(f"mot{i}" for i in range(250))
    chunks, stats = preparer([brut("Article 1", resume, "l1", numero="1")])
    assert stats["chunks"] == len(chunks) > 1
    assert stats["articles_decoupes"] == 1
    assert all(c["id_article"] == chunks[0]["id_article"] for c in chunks)
