"""
exploration.py — 2.2 Description et exploration des données.

Questions auxquelles cette étape répond AVANT de transformer quoi que ce soit :
  - Combien d'articles ? De quelles sources ?          -> risque de biais
  - Quels champs sont vides ? Y a-t-il des doublons ?  -> qualité
  - Les textes sont-ils longs ou courts ?              -> taille des chunks (étape 3)
  - Y a-t-il du HTML ? De quand datent les articles ?  -> nettoyage, fraîcheur

Séparation calcul / affichage :
  `explorer()` CALCULE et renvoie un dictionnaire de statistiques ;
  `afficher_rapport()` l'AFFICHE. Le même calcul sert ainsi au terminal,
  au notebook et à l'interface web.
"""

from email.utils import parsedate_to_datetime

import pandas as pd

CHAMPS = ["titre", "resume", "lien", "date"]


def lire_date(texte: str):
    """Les flux RSS utilisent le format des e-mails (RFC 822) :
    ex. "Fri, 13 Nov 2026 20:26:36 GMT"."""
    try:
        return parsedate_to_datetime(texte)
    except (TypeError, ValueError):
        return None


def vers_dataframe(articles: list[dict]) -> pd.DataFrame:
    """Liste d'articles -> tableau pandas (lignes = articles, colonnes = champs)."""
    df = pd.DataFrame(articles)
    df["nb_mots_resume"] = df["resume"].str.split().str.len()
    return df


def explorer(articles: list[dict]) -> dict:
    """Calcule toutes les statistiques d'exploration."""
    df = vers_dataframe(articles)
    dates = pd.to_datetime(df["date"].map(lire_date), errors="coerce", utc=True)

    return {
        "nb_articles": len(df),
        "colonnes": [c for c in df.columns if c != "nb_mots_resume"],
        "par_source": df["source"].value_counts().to_dict(),
        # un champ vide ("") n'est pas « NaN » pour pandas : on compte à la main
        "champs_vides": {c: int((df[c].str.strip() == "").sum()) for c in CHAMPS},
        "doublons_titre": int(df.duplicated(subset="titre").sum()),
        "doublons_lien": int(df.duplicated(subset="lien").sum()),
        "longueur_resume": df["nb_mots_resume"].describe().round(1).to_dict(),
        "resumes_avec_html": int(df["resume"].str.contains(r"<[^>]+>", regex=True).sum()),
        "date_min": f"{dates.min():%d/%m/%Y %H:%M}" if dates.notna().any() else None,
        "date_max": f"{dates.max():%d/%m/%Y %H:%M}" if dates.notna().any() else None,
        "dates_illisibles": int(dates.isna().sum()),
    }


def afficher_rapport(stats: dict, exemple: dict | None = None) -> None:
    """Affiche le rapport d'exploration dans le terminal."""
    print("\n" + "=" * 70 + "\nRAPPORT D'EXPLORATION DES DONNÉES\n" + "=" * 70)
    print(f"[a] Volume      : {stats['nb_articles']} articles — colonnes {stats['colonnes']}")
    print("[b] Par source  : " + ", ".join(f"{s} = {n}" for s, n in stats["par_source"].items()))
    print("[c] Champs vides: " + ", ".join(f"{c} = {n}" for c, n in stats["champs_vides"].items()))
    print(f"[d] Doublons    : {stats['doublons_titre']} par titre, {stats['doublons_lien']} par lien")
    lg = stats["longueur_resume"]
    print(f"[e] Longueur des résumés (mots) : moyenne {lg['mean']}, "
          f"min {lg['min']}, médiane {lg['50%']}, max {lg['max']}")
    print(f"[f] Résumés contenant du HTML : {stats['resumes_avec_html']}")
    if stats["date_min"]:
        print(f"[g] Période     : du {stats['date_min']} au {stats['date_max']} (UTC), "
              f"{stats['dates_illisibles']} date(s) illisible(s)")
    if exemple:
        print(f"[h] Exemple     : {exemple['titre']}\n                  {exemple['resume'][:150]}...")

    print("-" * 70)
    print("CONCLUSIONS POUR L'ÉTAPE 3 (préparation) :")
    print("  * supprimer les balises HTML ; * supprimer les doublons ;")
    print("  * écarter les articles sans texte exploitable ;")
    print("  * fusionner titre + résumé ; * découper les textes longs en chunks.")
    print("-" * 70)
