"""
collecte.py — 2.1 Collecte initiale des données : le Code du travail tunisien.

La source est un PDF (texte de loi). On le transforme en UNE ligne PAR ARTICLE.
Chaque article collecté a TOUJOURS les clés historiques du projet de référence
    {"source", "titre", "resume", "lien", "date"}
(ainsi les étapes suivantes fonctionnent sans changement) PLUS des clés propres
au droit, qui serviront à citer précisément :
    "numero"       "6-2", "5 bis", "234"        (numéro d'article normalisé)
    "texte_loi"    "code" ou "promulgation"     (le code lui-même / la loi qui le promulgue)
    "livre", "division", "chapitre", "section"  (où se trouve l'article)
    "modification" note d'origine : "Modifié par la loi n° 96-62 du 15 juillet 1996"
    "statut"       "en vigueur" ou "abrogé"
    "page_pdf"     page du PDF où commence l'article

Contraintes héritées de l'étape 3 du projet de référence :
  * "lien" doit être UNIQUE par article (le dédoublonnage s'appuie dessus) ;
  * "titre" doit être UNIQUE aussi ;
  * "date" est au format e-mail (RFC 822), lu par exploration.py.

Principe « données brutes » : on ne corrige RIEN dans les textes (même les
coupures de mots dues au PDF). Seule modification : retirer les numéros de page,
qui ne font pas partie de la loi. Le texte complet extrait est aussi sauvegardé
tel quel (config.FICHIER_TEXTE_BRUT).

Séparation lecture / analyse : `lire_pages_pdf()` touche au disque,
`decouper_en_articles()` est une fonction pure (testable sans PDF).
"""

import re
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path

from agent_rag import config

# --------------------------------------------------------------------------
# Expressions régulières (reconnaissance de la structure du code)
# --------------------------------------------------------------------------
SUFFIXES = "bis|ter|quater|quinquies|quinter|sexies|septies|octies|nonies|decies"

# "Article 6", "Article 6-2", "Article 5. bis", "Article 12 - bis", "Article premier"
RE_NUMERO = re.compile(rf"^(premier|\d+)(?:\s*[.\-–]?\s*(?:\d+|{SUFFIXES}))*$", re.IGNORECASE)
# début d'article : "Article <numéro> [(note de modification]"
RE_ARTICLE = re.compile(r"^Article\s+([^(]+?)\s*(\(.*)?$")
# lignes groupées : "Articles 49 et 50 (Abrogés ...", "Article 270 à 277 (Abrogés ...", "Art. 336 et 337 (..."
RE_GROUPE = re.compile(r"^(?:Articles?|Art\.)\s+(\d+)(?:\s*(et|à|-|–)\s*(\d+))?\s*(\(.*)$")
RE_DEBUT_CODE = re.compile(r"^CODE DU TRAVAIL$")   # titre qui ouvre le code, après la loi de promulgation
RE_FIN_TABLE = re.compile(r"^\s*TABLE DES MATI[EÈ]RES\s*$")   # le code se termine ici (index ensuite)

RE_LIVRE = re.compile(r"^LIVRE\s+\S+")
RE_TITRE = re.compile(r"^TITRE\s+(PREMIER|[IVXLC]+|\d+)\b")
RE_CHAPITRE = re.compile(r"^Chapitre\s+(Premier|[IVXLC]+|\d+)\s*(?:[-–.:]\s*(.*))?$")
RE_SECTION = re.compile(r"^Section\s+(unique|[IVXLC]+|\d+)\s*\.?\s*[-–]\s*(.+)$")

RE_DATE_LOI = re.compile(r"du\s+(1er|\d{1,2})\s+([a-zéèêûîôàç]+)\s+(\d{4})", re.IGNORECASE)
MOIS = {"janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
        "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10,
        "novembre": 11, "décembre": 12, "decembre": 12}


# --------------------------------------------------------------------------
# Petites fonctions utilitaires
# --------------------------------------------------------------------------
def _epurer_numero(brut: str) -> str:
    """Retire la ponctuation de fin : "9. -" -> "9" ; "5." -> "5"."""
    return brut.strip().rstrip(" .-–:").strip()


def normaliser_numero(brut: str) -> str:
    """"5. bis" -> "5 bis" ;  "6 - 2" -> "6-2" ;  "premier" -> "1"."""
    texte = _epurer_numero(brut)
    if texte.lower() == "premier":
        return "1"
    texte = re.sub(rf"\s*[.\-–]?\s*({SUFFIXES})\b", r" \1", texte, flags=re.IGNORECASE)  # suffixe latin
    texte = re.sub(r"\s*[-–]\s*", "-", texte)       # "6 - 2" -> "6-2"
    return re.sub(r"\s+", " ", texte).lower().strip()


def date_rfc822(jour: int, mois: int, annee: int) -> str:
    """Date au format e-mail, comme dans les flux RSS : "Sat, 30 Apr 1966 00:00:00 GMT"."""
    return format_datetime(datetime(annee, mois, jour, tzinfo=timezone.utc), usegmt=True)


def date_depuis_note(note: str) -> str | None:
    """Dernière date de loi citée dans une note de modification (ou None)."""
    trouvees = RE_DATE_LOI.findall(note or "")
    for jour, mois, annee in reversed(trouvees):
        if mois.lower() in MOIS:
            return date_rfc822(1 if jour == "1er" else int(jour), MOIS[mois.lower()], int(annee))
    return None


def _est_numero_de_page(ligne: str) -> bool:
    return ligne.strip().isdigit() and len(ligne.strip()) <= 3


def _debut_article(ligne: str):
    """Renvoie le « match » si la ligne ouvre un article, sinon None.

    Une phrase comme « Article 15 du présent code ... » n'est PAS un début
    d'article : après « Article », seul un numéro est accepté.
    """
    m = RE_ARTICLE.match(ligne)
    if m and RE_NUMERO.match(_epurer_numero(m.group(1))):
        return m
    return None


# --------------------------------------------------------------------------
# Lecture du PDF
# --------------------------------------------------------------------------
def lire_pages_pdf(chemin: Path) -> list[str]:
    """Texte de chaque page du PDF (liste de str)."""
    import pymupdf   # PyMuPDF : extraction plus fiable que pypdf sur les titres en capitales

    with pymupdf.open(chemin) as document:
        return [page.get_text() for page in document]


def _lignes_avec_page(pages: list[str]):
    """Itère (numéro_de_page_pdf, ligne) en retirant les numéros de page imprimés
    (1re ou dernière ligne non vide de la page) et en s'arrêtant à l'index final."""
    for num_page, texte in enumerate(pages, start=1):
        lignes = [l.replace("\f", "").rstrip() for l in texte.splitlines()]
        non_vides = [i for i, l in enumerate(lignes) if l.strip()]
        if non_vides:
            for i in (non_vides[0], non_vides[-1]):
                if _est_numero_de_page(lignes[i]):
                    lignes[i] = ""
        for ligne in lignes:
            if RE_FIN_TABLE.match(ligne):
                return
            yield num_page, ligne


# --------------------------------------------------------------------------
# Analyse : texte -> articles
# --------------------------------------------------------------------------
def decouper_en_articles(pages: list[str]) -> list[dict]:
    """Fonction pure : liste de pages (texte) -> liste d'articles (dict)."""
    lignes = [(p, l.strip()) for p, l in _lignes_avec_page(pages)]
    lignes = [(p, l) for p, l in lignes if l]          # on ignore les lignes vides

    contexte = {"livre": "", "division": "", "chapitre": "", "section": ""}
    dans_le_code = False       # False tant qu'on lit la loi de promulgation (avant « LIVRE PREMIER »)
    articles: list[dict] = []
    courant: dict | None = None
    i = 0

    def lire_note(note: str, i: int) -> tuple[str, str, int]:
        """Une note « (Modifié par la loi ... » peut se prolonger sur 1-3 lignes.
        Renvoie (note_nettoyée, texte_restant_après_la_parenthèse, nouvel_indice)."""
        essais = 0
        while ")" not in note and i < len(lignes) and essais < 3:
            note += " " + lignes[i][1]
            i += 1
            essais += 1
        note, _, reste = note.partition(")")
        return re.sub(r"\s+", " ", note.lstrip("( ")).strip(), reste.lstrip(" .").strip(), i

    def contexte_courant(page: int, numero: str, note: str, lignes_texte: list[str]) -> dict:
        return {
            "numero": numero,
            "texte_loi": "code" if dans_le_code else "promulgation",
            "livre": contexte["livre"] if dans_le_code
            else "Loi de promulgation du code du travail",
            "division": contexte["division"] if dans_le_code else "",
            "chapitre": contexte["chapitre"] if dans_le_code else "",
            "section": contexte["section"] if dans_le_code else "",
            "modification": note,
            "page_pdf": page,
            "_lignes": lignes_texte,
        }

    def est_structure(ligne: str) -> bool:
        return bool(RE_LIVRE.match(ligne) or RE_TITRE.match(ligne)
                    or RE_CHAPITRE.match(ligne) or RE_SECTION.match(ligne))

    def intitule_suivant(debut: int, majuscules: bool) -> tuple[str, int]:
        """Intitulé d'un livre/titre/chapitre = ligne(s) qui suivent l'en-tête."""
        j, morceaux = debut, []
        while j < len(lignes):
            texte = lignes[j][1]
            if _debut_article(texte) or est_structure(texte):
                break
            if morceaux and (not majuscules or not texte.isupper()):
                break
            morceaux.append(texte)
            j += 1
            if not majuscules:
                break          # un chapitre : on ne prend qu'une ligne
        return " ".join(morceaux), j

    while i < len(lignes):
        page, ligne = lignes[i]

        # --- le code commence après la loi de promulgation ------------------------
        if RE_DEBUT_CODE.match(ligne) and articles:
            dans_le_code = True
            i += 1
            intitule, i = intitule_suivant(i, majuscules=True)   # ex. « DISPOSITIONS GENERALES »
            contexte.update(livre=intitule, division="", chapitre="", section="")
            courant = None
            continue

        # --- en-têtes de structure -------------------------------------------------
        if RE_LIVRE.match(ligne):
            intitule, i = intitule_suivant(i + 1, majuscules=True)
            contexte.update(livre=f"{ligne} — {intitule}".strip(" —"), division="", chapitre="", section="")
            dans_le_code = True
            courant = None
            continue
        if RE_TITRE.match(ligne):
            intitule, i = intitule_suivant(i + 1, majuscules=True)
            contexte.update(division=f"{ligne} — {intitule}".strip(" —"), chapitre="", section="")
            courant = None
            continue
        m = RE_CHAPITRE.match(ligne)
        if m:
            intitule = m.group(2) or ""
            i += 1
            if not intitule:
                intitule, i = intitule_suivant(i, majuscules=False)
            en_tete = ligne if not m.group(2) else ligne[:ligne.index(m.group(2))].rstrip(" -–.:")
            contexte.update(chapitre=f"{en_tete} — {intitule}".strip(" —"), section="")
            courant = None
            continue
        if RE_SECTION.match(ligne):
            contexte["section"] = ligne
            i += 1
            courant = None
            continue

        # --- début d'un article ----------------------------------------------------
        m = _debut_article(ligne)
        if m:
            note, reste, i = lire_note(m.group(2) or "", i + 1) if m.group(2) else ("", "", i + 1)
            courant = contexte_courant(page, normaliser_numero(m.group(1)), note,
                                       [reste] if reste else [])
            articles.append(courant)
            continue

        # --- ligne groupée d'articles abrogés : un enregistrement par numéro --------
        m = RE_GROUPE.match(ligne)
        if m:
            debut_n, lien_mots, fin_n, reste_ligne = int(m.group(1)), m.group(2), m.group(3), m.group(4)
            note, _, i = lire_note(reste_ligne, i + 1)
            if fin_n is None:
                numeros = [debut_n]
            elif lien_mots == "et":
                numeros = [debut_n, int(fin_n)]
            else:
                numeros = list(range(debut_n, int(fin_n) + 1))
            for n in numeros:
                articles.append(contexte_courant(page, str(n), note, []))
            courant = None           # le texte qui suit n'appartient à aucun de ces articles
            continue

        # --- ligne de contenu : appartient à l'article en cours --------------------
        if courant is not None:
            courant["_lignes"].append(ligne)
        i += 1

    return [_finaliser(a) for a in articles]


def _finaliser(a: dict) -> dict:
    """Ajoute les clés « historiques » (source, titre, resume, lien, date) et le statut."""
    corps = "\n".join(a.pop("_lignes"))
    numero = a["numero"]
    promulgation = a["texte_loi"] == "promulgation"

    titre = f"Loi 66-27, article {numero}" if promulgation else f"Article {numero}"
    slug = numero.replace(" ", "-")
    ancre = f"promulgation-{slug}" if promulgation else f"article-{slug}"
    date = date_depuis_note(a["modification"]) or date_rfc822(*config.DATE_LOI_ORIGINE)
    abroge = bool(re.search(r"abrog", a["modification"], re.IGNORECASE)) or \
        bool(re.fullmatch(r"\(?\s*abrog\w*[^.]*\.?\)?", corps.strip(), re.IGNORECASE))

    return {
        "source": config.NOM_SOURCE,
        "titre": titre,
        "resume": corps,
        "lien": f"{config.URL_SOURCE}#{ancre}",
        "date": date,
        **a,
        "statut": config.STATUT_ABROGE if abroge else config.STATUT_EN_VIGUEUR,
    }


# --------------------------------------------------------------------------
# Contrôle qualité de la numérotation
# --------------------------------------------------------------------------
def controle_numerotation(articles: list[dict]) -> dict:
    """Détecte doublons et trous dans la suite des numéros du code (articles « code » seulement)."""
    numeros = [a["numero"] for a in articles if a["texte_loi"] == "code"]
    principaux = [int(re.match(r"\d+", n).group()) for n in numeros]
    vus, doublons = set(), []
    for n in numeros:
        (doublons.append(n) if n in vus else vus.add(n))
    manquants = sorted(set(range(1, max(principaux) + 1)) - set(principaux)) if principaux else []
    return {"nb": len(numeros), "doublons": doublons, "manquants": manquants,
            "numero_max": max(principaux) if principaux else 0}


# --------------------------------------------------------------------------
# Point d'entrée de l'étape 2.1
# --------------------------------------------------------------------------
def collecter(verbeux: bool = True) -> tuple[list[dict], str]:
    """Lit le PDF du code du travail et renvoie (articles, origine) avec origine = "pdf"."""
    if not config.FICHIER_PDF_SOURCE.exists():
        raise SystemExit(
            f"PDF introuvable : {config.FICHIER_PDF_SOURCE}\n"
            "Téléchargez le Code du travail (PDF) et placez-le à cet emplacement."
        )
    pages = lire_pages_pdf(config.FICHIER_PDF_SOURCE)

    # Texte brut complet, conservé tel quel (preuve de ce qu'on a réellement lu).
    config.FICHIER_TEXTE_BRUT.parent.mkdir(parents=True, exist_ok=True)
    config.FICHIER_TEXTE_BRUT.write_text("\f".join(pages), encoding="utf-8")

    articles = decouper_en_articles(pages)
    if verbeux:
        controle = controle_numerotation(articles)
        print(f"  -> {len(pages)} pages lues, {len(articles)} articles extraits "
              f"(dernier numéro du code : {controle['numero_max']})")
        if controle["doublons"] or controle["manquants"]:
            print(f"  /!\\ numéros en double : {controle['doublons'][:10]} ; "
                  f"numéros manquants : {controle['manquants'][:10]}")
    return articles, "pdf"
