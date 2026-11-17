"""
nettoyage.py — 3.1 Sélection, 3.2 Nettoyage, 3.3 Construction, 3.4 Intégration.

Adapté au Code du travail (texte de loi issu d'un PDF). Chaque sous-étape
répond à un constat de l'exploration (étape 2) :

  constat de l'étape 2                         ->  décision
  ------------------------------------------------------------------------------
  35 articles abrogés, sans aucun texte        ->  3.1 les écarter (ils sont retrouvables
                                                   par l'outil « rechercher_article »)
  1 article = liste de décrets abrogés (790 m.) ->  3.1 l'écarter (config.TITRES_ECARTES) :
                                                   il remontait en tête de recherche à tort
  0 balise HTML, 0 doublon                     ->  pas de nettoyage HTML ; le dédoublonnage
                                                   reste comme garde-fou
  retours à la ligne et apostrophes ’ du PDF   ->  3.2 espaces normalisés, apostrophe droite '
  « Article 14 » seul ne dit rien du sujet     ->  3.3 en-tête « Article 14 — Fin du contrat
                                                   de travail » (numéro + chapitre)
  champs propres au droit (chapitre, page...)  ->  3.4 recopiés sur chaque chunk (citations)
"""

import hashlib
import re

from agent_rag import config


# --------------------------------------------------------------------------
# 3.2 Nettoyage d'un texte
# --------------------------------------------------------------------------
def nettoyer_texte(texte: str) -> str:
    """
    Normalise un texte de loi extrait d'un PDF.

        entrée : "Le contrat\\n  de l’employeur\\xa0prend fin."
        sortie : "Le contrat de l'employeur prend fin."

    Le contenu juridique n'est JAMAIS modifié : seuls les espaces, les coupures de
    ligne et le type d'apostrophe changent (l'utilisateur tape ', le PDF contient ’).
    """
    texte = (texte or "").replace("\xad", "")                 # trait d'union conditionnel invisible
    texte = texte.replace("\xa0", " ").replace("’", "'")      # espace insécable, apostrophe typographique
    texte = re.sub(r"\s+", " ", texte)                        # retours à la ligne et espaces multiples
    return texte.strip()


def identifiant_stable(lien: str) -> str:
    """3.3 Le même lien donne TOUJOURS le même identifiant (hachage MD5)."""
    return hashlib.md5(lien.encode("utf-8")).hexdigest()[:10]


# --------------------------------------------------------------------------
# 3.3 Construction : l'en-tête qui donne du contexte à chaque morceau
# --------------------------------------------------------------------------
def intitule(rubrique: str) -> str:
    """
    "Chapitre III — Fin du contrat de travail"  ->  "Fin du contrat de travail"
    "TITRE PREMIER — LE CONTRAT DE TRAVAIL"     ->  "Le contrat de travail"
    """
    texte = nettoyer_texte(rubrique).split(" — ", 1)[-1].strip(" *")
    return texte.capitalize() if texte.isupper() else texte


def construire_en_tete(article: dict, titre: str) -> str:
    """
    "Article 14 — Fin du contrat de travail".

    Le numéro seul ne dit rien du SUJET ; ajouter le chapitre donne des mots-clés
    à la recherche (« fin du contrat », « préavis »...). L'en-tête est répété sur
    chaque morceau d'un article long : ainsi aucun morceau n'est « anonyme ».
    """
    rubrique = (intitule(article.get("chapitre", ""))
                or intitule(article.get("division", ""))
                or intitule(article.get("livre", "")))
    return f"{titre} — {rubrique}" if rubrique else titre


# --------------------------------------------------------------------------
# 3.1 -> 3.4 sur l'ensemble des articles
# --------------------------------------------------------------------------
def preparer_articles(articles_bruts: list[dict]) -> tuple[list[dict], dict]:
    """
    Sélectionne, nettoie, dédoublonne, construit et met au format unique.
    Renvoie (articles_propres, statistiques « avant / après »).
    """
    propres = []
    liens_vus, titres_vus = set(), set()
    stats = {"lus": len(articles_bruts), "abroges": 0, "ecartes_manuels": 0,
             "doublons": 0, "trop_courts": 0,
             "mots_avant": sum(len((a.get("resume") or "").split()) for a in articles_bruts)}

    for article in articles_bruts:
        # 3.1 Sélection : un article abrogé n'a plus de texte en vigueur
        if article.get("statut") == config.STATUT_ABROGE:
            stats["abroges"] += 1
            continue

        titre = nettoyer_texte(article.get("titre", ""))
        corps = nettoyer_texte(article.get("resume", ""))
        lien = (article.get("lien") or "").strip()

        # 3.1 Sélection : écarts décidés à la main, avec leur raison (config.TITRES_ECARTES)
        if titre in config.TITRES_ECARTES:
            stats["ecartes_manuels"] += 1
            continue

        # 3.2 Dédoublonnage : même lien OU même titre = même article
        if lien in liens_vus or titre.lower() in titres_vus:
            stats["doublons"] += 1
            continue
        liens_vus.add(lien)
        titres_vus.add(titre.lower())

        # 3.1 Sélection : un texte trop pauvre n'apporte rien à la recherche
        if len(corps.split()) < config.LONGUEUR_MIN_ARTICLE:
            stats["trop_courts"] += 1
            continue

        # 3.3 Construction : en-tête contextuel (titre + chapitre)
        en_tete = construire_en_tete(article, titre)

        # 3.4 Intégration : un format unique ; les métadonnées du droit sont recopiées
        propre = {
            "id_article": identifiant_stable(lien or titre),
            "source": article.get("source", "inconnue"),
            "titre": titre,
            "en_tete": en_tete,
            "texte": corps,
            "lien": lien,
            "date": article.get("date", ""),
        }
        propre.update({c: article[c] for c in config.CHAMPS_METADONNEES if c in article})
        propres.append(propre)

    stats["conserves"] = len(propres)
    stats["mots_apres"] = sum(len(p["texte"].split()) for p in propres)
    return propres, stats
