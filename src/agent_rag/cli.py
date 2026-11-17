"""
cli.py — Une seule commande pour piloter tout le projet.

    python main.py donnees                 Étape 2 : collecte + exploration
    python main.py preparation             Étape 3 : nettoyage + chunks
    python main.py modelisation            Étape 4 : embeddings + index
    python main.py rag "ma question"       Étape 4 : tester la chaîne RAG classique
    python main.py evaluation [--generation] [--nouveau-jeu]
                                           Étape 5 : métriques + critères de succès
    python main.py agent                   Étape 6 : discuter avec l'agent (terminal)
    python main.py api                     Étape 6 : lancer l'API REST (FastAPI)
    python main.py interface               Étape 6 : lancer l'interface web (Streamlit)
    python main.py pipeline                Étapes 2 -> 5 d'un coup (mise à jour de l'actualité)
"""

import argparse
import subprocess
import sys

from agent_rag import config


def lancer_rag(question: str) -> None:
    from agent_rag.etape4_modelisation.index_vectoriel import IndexVectoriel
    from agent_rag.etape4_modelisation.rag import ChaineRAG

    reponse = ChaineRAG(IndexVectoriel.charger().rechercher).repondre(question)
    print("Passages retrouvés (R) :")
    for p in reponse.passages:
        print(f"  [{p['score']:.3f}] ({p['source']}) {p['titre'][:80]}")
    print(f"\nRéponse générée (G) :\n{reponse.texte}")


def lancer_pipeline() -> None:
    from agent_rag.etape2_donnees import etape as e2
    from agent_rag.etape3_preparation import etape as e3
    from agent_rag.etape4_modelisation import etape as e4
    from agent_rag.etape5_evaluation import etape as e5

    for numero, module in [(2, e2), (3, e3), (4, e4)]:
        print("\n" + "#" * 70 + f"\n#  ÉTAPE {numero}\n" + "#" * 70)
        module.executer()
    print("\n" + "#" * 70 + "\n#  ÉTAPE 5\n" + "#" * 70)
    e5.executer(nouveau_jeu=True)   # nouvelles données => nouvelles questions
    print("\nPipeline terminé. Discutez avec l'agent :  python main.py agent")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Agent d'actualité RAG — un projet CRISP-DM de bout en bout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sous = parser.add_subparsers(dest="commande", required=True, metavar="commande")

    sous.add_parser("donnees", help="Étape 2 : collecte RSS + exploration")
    sous.add_parser("preparation", help="Étape 3 : nettoyage + découpage en chunks")
    sous.add_parser("modelisation", help="Étape 4 : embeddings + index vectoriel")
    p_rag = sous.add_parser("rag", help="Étape 4 : poser une question à la chaîne RAG classique")
    p_rag.add_argument("question")
    p_eval = sous.add_parser("evaluation", help="Étape 5 : métriques + critères de succès")
    p_eval.add_argument("--generation", action="store_true",
                        help="évaluer aussi les réponses (RAG classique vs agent) — lent")
    p_eval.add_argument("--nouveau-jeu", action="store_true",
                        help="régénérer le jeu de test avec le LLM")
    sous.add_parser("agent", help="Étape 6 : discuter avec l'agent dans le terminal")
    p_api = sous.add_parser("api", help="Étape 6 : lancer l'API REST (FastAPI)")
    p_api.add_argument("--port", type=int, default=8000)
    sous.add_parser("interface", help="Étape 6 : lancer l'interface web (Streamlit)")
    sous.add_parser("pipeline", help="Étapes 2 -> 5 d'un coup")

    args = parser.parse_args(argv)

    if args.commande == "donnees":
        from agent_rag.etape2_donnees.etape import executer
        executer()
    elif args.commande == "preparation":
        from agent_rag.etape3_preparation.etape import executer
        executer()
    elif args.commande == "modelisation":
        from agent_rag.etape4_modelisation.etape import executer
        executer()
    elif args.commande == "rag":
        lancer_rag(args.question)
    elif args.commande == "evaluation":
        from agent_rag.etape5_evaluation.etape import executer
        executer(generation=args.generation, nouveau_jeu=args.nouveau_jeu)
    elif args.commande == "agent":
        from agent_rag.etape6_deploiement.etape import executer
        executer()
    elif args.commande == "api":
        import uvicorn
        print(f"API : http://localhost:{args.port}/docs  (documentation interactive)")
        uvicorn.run("app.api:app", host="127.0.0.1", port=args.port, app_dir=str(config.RACINE))
    elif args.commande == "interface":
        subprocess.run([sys.executable, "-m", "streamlit", "run",
                        str(config.RACINE / "app" / "interface.py")], check=False)
    elif args.commande == "pipeline":
        lancer_pipeline()


if __name__ == "__main__":
    main()
