import pytest

from agent_rag.etape5_evaluation.baseline import recherche_par_mots_cles
from agent_rag.etape5_evaluation.eval_recherche import evaluer_recherche
from agent_rag.etape5_evaluation.metriques import hit_at, mrr, rang, taux


def test_rang():
    resultats = [{"id_article": "a"}, {"id_article": "b"}]
    assert rang(resultats, "b") == 2
    assert rang(resultats, "z") is None


def test_hit_et_mrr():
    rangs = [1, 2, None, 4]
    assert hit_at(rangs, 1) == 0.25
    assert hit_at(rangs, 3) == 0.5
    assert mrr(rangs) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def test_listes_vides():
    assert hit_at([], 3) == 0.0 and mrr([]) == 0.0 and taux([]) == 0.0


def test_baseline_mots_communs(chunks):
    assert recherche_par_mots_cles("hausse des prix", chunks, k=1)[0]["id_article"] == "a2"


def test_evaluation_recherche_parfaite(index):
    jeu = [{"question": "match de rugby", "id_article_attendu": "a1"},
           {"question": "guerre en Ukraine", "id_article_attendu": "a3"}]
    assert evaluer_recherche(index.rechercher, jeu, k=3) == {"Hit@1": 1.0, "Hit@3": 1.0, "MRR": 1.0}
