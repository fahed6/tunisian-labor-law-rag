"""
llm.py — Le SEUL module qui parle au LLM.
=========================================

Principe d'architecture : le reste du projet ne connaît pas Ollama.
Il manipule des objets simples (`ReponseLLM`, `AppelOutil`) et des messages
au format standard {"role": ..., "content": ...}.

=> Pour passer à un autre fournisseur (OpenAI, Claude, Gemini, Mistral...),
   on ne modifie QUE ce fichier. C'est le principe de l'« adaptateur ».

Pré-requis : Ollama installé et lancé, avec le modèle téléchargé :
    ollama pull qwen2.5:3b
"""

from dataclasses import dataclass, field

from agent_rag import config


@dataclass
class AppelOutil:
    """Le LLM demande à exécuter l'outil `nom` avec ces `arguments`."""
    nom: str
    arguments: dict


@dataclass
class ReponseLLM:
    """Réponse du LLM : du texte, et/ou des demandes d'appel d'outils."""
    texte: str
    appels_outils: list[AppelOutil] = field(default_factory=list)

    def message_assistant(self) -> dict:
        """
        Le message à remettre dans l'historique de la conversation.
        Il doit contenir les appels d'outils : au tour suivant, le LLM doit
        se souvenir de ce qu'il a demandé.
        """
        message = {"role": "assistant", "content": self.texte}
        if self.appels_outils:
            message["tool_calls"] = [
                {"function": {"name": a.nom, "arguments": a.arguments}}
                for a in self.appels_outils
            ]
        return message


def verifier_disponibilite(modele: str = config.MODELE_LLM) -> tuple[bool, str]:
    """Vérifie qu'Ollama répond ET que le modèle demandé est installé."""
    try:
        import ollama
        installes = [m.model for m in ollama.list().models]
    except Exception:
        return False, ("Ollama ne répond pas. Lancez l'application Ollama "
                       "(ou « ollama serve ») puis réessayez.")
    if modele not in installes:
        return False, (f"Le modèle « {modele} » n'est pas installé. "
                       f"Tapez : ollama pull {modele}  (installés : {installes})")
    return True, f"Ollama prêt ({modele})"


def appeler_llm(messages: list[dict], outils: list[dict] | None = None,
                modele: str = config.MODELE_LLM,
                temperature: float = config.TEMPERATURE_LLM) -> ReponseLLM:
    """
    Envoie une conversation au LLM et renvoie sa réponse.

    messages : [{"role": "system",    "content": "consignes"},
                {"role": "user",      "content": "question"},
                {"role": "assistant", "content": "...", "tool_calls": [...]},
                {"role": "tool",      "content": "résultat", "tool_name": "..."}]
    outils   : description (JSON Schema) des outils que le LLM peut appeler
    """
    import ollama   # import local : le reste du projet fonctionne sans Ollama

    reponse = ollama.chat(
        model=modele,
        messages=messages,
        tools=outils,
        options={"temperature": temperature, "num_predict": config.MAX_TOKENS_REPONSE},
    )
    message = reponse.message
    return ReponseLLM(
        texte=(message.content or "").strip(),
        appels_outils=[
            AppelOutil(nom=a.function.name, arguments=dict(a.function.arguments or {}))
            for a in (message.tool_calls or [])
        ],
    )


def demander(consigne: str, modele: str = config.MODELE_LLM,
             temperature: float = config.TEMPERATURE_LLM) -> str:
    """Raccourci : une question simple, une réponse texte."""
    return appeler_llm([{"role": "user", "content": consigne}],
                       modele=modele, temperature=temperature).texte
