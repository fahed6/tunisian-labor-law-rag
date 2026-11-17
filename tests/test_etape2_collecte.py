"""
Tests de l'étape 2 (collecte du Code du travail).

Ils utilisent de FAUSSES pages (du texte écrit ici) : pas besoin du vrai PDF,
donc rapides et déterministes.
"""

import pytest

from agent_rag.etape2_donnees.collecte import (
    controle_numerotation,
    date_depuis_note,
    decouper_en_articles,
    normaliser_numero,
)

# Une mini « édition » : loi de promulgation, début du code, un livre, un chapitre,
# un article modifié (note sur 2 lignes), un groupe d'articles abrogés, puis l'index final.
PAGES = [
    # page 1 : cover (aucun article)
    "REPUBLIQUE TUNISIENNE\nCODE DU TRAVAIL\n2010\n",
    # page 2 : loi de promulgation
    "3\nLoi n° 66-27 du 30 avril 1966, portant promulgation du code.\n"
    "Article premier\nLes textes ci-après constituent le code du travail.\n"
    "Article 2\nLes conventions collectives restent en vigueur.\n",
    # page 3 : le code commence
    "9\nCODE DU TRAVAIL\nDISPOSITIONS GENERALES\n"
    "Article premier\nLe présent code s'applique aux établissements industriels.\n"
    "Article 5. bis ( Ajouté par la loi n° 93-66 du 5 juillet 1993)\n"
    "Aucune discrimination entre l'homme et la femme.\n",
    # page 4 : un livre, un titre, un chapitre
    "LIVRE PREMIER\nFORMATION DES RAPPORTS DE TRAVAIL\nTITRE PREMIER\nLE CONTRAT DE TRAVAIL\n"
    "Chapitre Premier\nFormation du contrat\n"
    "Article 6 (Modifié par la loi n° 96-62 du 15 juillet\n1996).\n"
    "Le contrat de travail est une convention. Voir l'article 7.\n"
    "Article 15 du présent code fixe le préavis (cette ligne est du texte, pas un article).\n"
    "Article 7\nLa relation de travail est prouvée par tous moyens.\n"
    "Articles 8 et 9 (Abrogés par la loi n° 96-62 du 15 juillet 1996).\n"
    "Article 10. - (Abrogé par la loi n° 96-62 du 15 juillet 1996).\n"
    "Article 11\nTexte de l'article onze.\n12\n",     # « 12 » = numéro de page en fin de page
    # page 5 : index final (ne doit PAS être lu)
    "TABLE DES MATIERES\nArticle 999\nIndex sans valeur.\n",
]


@pytest.fixture
def articles():
    return decouper_en_articles(PAGES)


def par_numero(articles, numero, texte_loi="code"):
    return next(a for a in articles if a["numero"] == numero and a["texte_loi"] == texte_loi)


# --------------------------------------------------------------------------
@pytest.mark.parametrize("brut, attendu", [
    ("premier", "1"), ("6-2", "6-2"), ("6 - 2", "6-2"), ("5. bis", "5 bis"),
    ("12 - bis", "12 bis"), ("9. -", "9"), ("14 quinter", "14 quinter"),
])
def test_normaliser_numero(brut, attendu):
    assert normaliser_numero(brut) == attendu


def test_date_depuis_note_prend_la_derniere_loi_citee():
    note = "Modifié par la loi n° 94-29 du 21 février 1994 puis la loi n° 96-62 du 1er août 1996"
    assert date_depuis_note(note) == "Thu, 01 Aug 1996 00:00:00 GMT"
    assert date_depuis_note("") is None


def test_separation_promulgation_et_code(articles):
    assert [a["numero"] for a in articles if a["texte_loi"] == "promulgation"] == ["1", "2"]
    assert par_numero(articles, "1")["livre"] == "DISPOSITIONS GENERALES"


def test_hierarchie_livre_titre_chapitre(articles):
    a = par_numero(articles, "6")
    assert a["livre"].startswith("LIVRE PREMIER — FORMATION DES RAPPORTS")
    assert a["division"].startswith("TITRE PREMIER — LE CONTRAT")
    assert a["chapitre"] == "Chapitre Premier — Formation du contrat"


def test_note_de_modification_sur_deux_lignes(articles):
    a = par_numero(articles, "6")
    assert a["modification"] == "Modifié par la loi n° 96-62 du 15 juillet 1996"
    assert a["date"] == "Mon, 15 Jul 1996 00:00:00 GMT"
    assert "convention" in a["resume"]


def test_une_phrase_qui_cite_un_article_nest_pas_un_article(articles):
    # « Article 15 du présent code... » reste dans le texte de l'article 6
    assert "Article 15 du présent code" in par_numero(articles, "6")["resume"]
    assert not any(a["numero"] == "15" for a in articles)


def test_groupe_dabroges_donne_un_enregistrement_par_numero(articles):
    for numero in ("8", "9", "10"):
        a = par_numero(articles, numero)
        assert a["statut"] == "abrogé"
        assert a["resume"] == ""


def test_le_texte_ne_deborde_pas_sur_larticle_suivant(articles):
    assert par_numero(articles, "7")["resume"] == "La relation de travail est prouvée par tous moyens."


def test_numero_de_page_retire_et_index_final_ignore(articles):
    assert "12" not in par_numero(articles, "11")["resume"].split()
    assert not any(a["numero"] == "999" for a in articles)


def test_titre_et_lien_uniques_pour_le_dedoublonnage(articles):
    assert len({a["titre"] for a in articles}) == len(articles)
    assert len({a["lien"] for a in articles}) == len(articles)
    assert " " not in par_numero(articles, "5 bis")["lien"]


def test_champs_historiques_toujours_presents(articles):
    for a in articles:
        assert {"source", "titre", "resume", "lien", "date"} <= a.keys()


def test_controle_numerotation_detecte_un_trou():
    faux = [{"numero": n, "texte_loi": "code"} for n in ("1", "2", "4", "4")]
    rapport = controle_numerotation(faux)
    assert rapport["manquants"] == [3]
    assert rapport["doublons"] == ["4"]
