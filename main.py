"""
API de l'assistant de brief fragrances Givaudan (hackathon HETIC).
Lancer :  uvicorn main:app --host 0.0.0.0 --port 8000
Swagger : http://localhost:8000/docs
"""
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import rag

etat = {"rag": None, "erreur": None}

# Sessions en mémoire (démo : pas de base de données, tout est perdu au redémarrage).
SESSIONS = {}
DUREE_SESSION = timedelta(hours=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        etat["rag"] = rag.CatalogueRAG()
    except Exception as e:  # Ollama éteint, modèle absent, CSV introuvable...
        etat["erreur"] = str(e)
    yield


app = FastAPI(
    title="Givaudan – Assistant de brief Fragrances (RAG)",
    description=(
        "Assistant de brief pour des parfums sur mesure. Le client professionnel (marque de lessive, "
        "de shampoing, de savon...) décrit son projet en langage courant. L'API garde la conversation "
        "(sessions), remplit une fiche brief au fil des messages et choisit l'étape :\n"
        "- **decouverte** : questions pour comprendre le besoin, aucun ingrédient proposé ;\n"
        "- **pistes** : 2 ou 3 pistes olfactives illustrées par des ingrédients du catalogue ;\n"
        "- **affinage** : la piste est affinée selon les réactions du client.\n\n"
        "Les messages hors sujet sont recentrés et les demandes sans rapport (candidature, presse...) "
        "sont redirigées. Jamais de prix ni de délai.\n\n"
        "**Catalogue réel** de 30 ingrédients de parfumerie issu de givaudan.com (bibliothèque de molécules), "
        "utilisé uniquement pour une démo de hackathon."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# Autorise l'interface du collègue (autre port / autre machine) à appeler l'API.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ---------- Modèles de données (apparaissent dans le Swagger) ----------

Etape = Literal["decouverte", "pistes", "affinage"]
EtapeReponse = Literal["decouverte", "pistes", "affinage", "recentrage", "redirection"]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Brief(BaseModel):
    """Ce que l'on sait du projet du client ("" = pas encore précisé)."""
    produit: str = ""
    ambiance: str = ""
    cible: str = ""
    contraintes: str = ""
    pays: str = ""


class SessionCreee(BaseModel):
    session_id: str
    message_accueil: str


class DemandeQuestion(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{
        "session_id": None,
        "question": "Je cherche un parfum d'été",
        "nb_resultats": 3,
    }]}}

    session_id: Optional[str] = Field(
        None,
        description="Identifiant renvoyé par POST /sessions ou par un /ask précédent. "
                    "Absent ou inconnu : une nouvelle session est créée automatiquement.",
    )
    question: str = Field(..., description="Message du client, en langage courant.")
    nb_resultats: int = Field(3, ge=1, le=5, description="Nombre maximum d'ingrédients renvoyés.")


class Ingredient(BaseModel):
    """Ingrédient du catalogue. Les cases vides du CSV donnent "". Les listes sont séparées par " ; "."""
    id: str
    nom: str
    type: str = ""
    famille_olfactive: str = ""
    notes: str = ""
    description: str = ""
    applications_recommandees: str = ""
    applications_deconseillees: str = ""
    tenue: str = ""
    origine: str = ""
    naturel_ou_synthetique: str = ""
    durabilite: str = ""
    url_source: str = ""


class IngredientTrouve(Ingredient):
    score: float = Field(..., description="Proximité avec la demande (plus c'est haut, plus c'est proche).")


class ReponseQuestion(BaseModel):
    session_id: str = Field(..., description="À renvoyer dans le prochain /ask pour continuer la conversation.")
    reponse: str = Field(..., description="Réponse rédigée par l'assistant de brief.")
    etape: EtapeReponse = Field(
        ..., description="decouverte, pistes ou affinage ; recentrage (message hors sujet) ou "
                         "redirection (candidature, presse, fournisseur, éthique)."
    )
    brief: Brief = Field(..., description="Fiche brief mise à jour après ce message.")
    ingredients_proposes: List[IngredientTrouve] = Field(
        ..., description="Ingrédients cités dans les pistes olfactives, vide si le conseiller pose seulement une question."
    )


class SessionDetail(BaseModel):
    session_id: str
    etape: Etape
    brief: Brief
    ingredients_proposes: List[str] = Field(..., description="Identifiants des ingrédients déjà proposés.")
    messages: List[Message] = Field(..., description="Historique complet de la conversation.")
    cree_le: datetime


# ---------- Sessions ----------

def _purger_sessions():
    limite = datetime.now() - DUREE_SESSION
    for sid in [sid for sid, s in SESSIONS.items() if s["cree_le"] < limite]:
        del SESSIONS[sid]


def _nouvelle_session():
    _purger_sessions()
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {
        "messages": [{"role": "assistant", "content": rag.MESSAGE_ACCUEIL}],
        "brief": {c: "" for c in rag.CHAMPS_BRIEF},
        "etape": "decouverte",
        "ingredients_proposes": [],
        "cree_le": datetime.now(),
    }
    return sid


def _session(session_id):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session introuvable (expirée ou supprimée).")
    return SESSIONS[session_id]


def _rag():
    if etat["rag"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Le moteur RAG n'est pas prêt. Vérifie qu'Ollama est lancé. Détail : {etat['erreur']}",
        )
    return etat["rag"]


# ---------- Routes ----------

@app.get("/health", tags=["Système"], summary="Vérifier que l'API fonctionne")
def health():
    return {
        "statut": "ok" if etat["rag"] else "erreur",
        "modele_chat": rag.CHAT_MODEL,
        "modele_embedding": rag.EMBED_MODEL,
        "nb_ingredients": len(etat["rag"].ingredients) if etat["rag"] else 0,
        "erreur": etat["erreur"],
    }


@app.post("/sessions", response_model=SessionCreee, tags=["Conversation"],
          summary="Démarrer une conversation (message d'accueil)")
def creer_session():
    sid = _nouvelle_session()
    return SessionCreee(session_id=sid, message_accueil=rag.MESSAGE_ACCUEIL)


@app.post("/ask", response_model=ReponseQuestion, tags=["Conversation"],
          summary="Envoyer un message du client à l'assistant de brief")
def ask(demande: DemandeQuestion):
    moteur = _rag()
    sid = demande.session_id if demande.session_id in SESSIONS else _nouvelle_session()
    session = SESSIONS[sid]
    try:
        texte, etape, resultats = moteur.tour(session, demande.question, k=demande.nb_resultats)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erreur Ollama : {e}")
    return ReponseQuestion(
        session_id=sid,
        reponse=texte,
        etape=etape,
        brief=Brief(**session["brief"]),
        ingredients_proposes=[IngredientTrouve(**a, score=round(s, 3)) for a, s in resultats],
    )


@app.get("/sessions/{session_id}", response_model=SessionDetail, tags=["Conversation"],
         summary="Historique complet, brief et étape d'une session")
def detail_session(session_id: str):
    s = _session(session_id)
    return SessionDetail(session_id=session_id, **s)


@app.delete("/sessions/{session_id}", tags=["Conversation"], summary="Supprimer une session (recommencer)")
def supprimer_session(session_id: str):
    _session(session_id)
    del SESSIONS[session_id]
    return {"supprimee": True, "session_id": session_id}


@app.get("/ingredients", response_model=List[Ingredient], tags=["Catalogue"], summary="Lister le catalogue")
def lister_ingredients(famille: Optional[str] = Query(None, examples=["Boisé"])):
    ingredients = _rag().ingredients
    if famille:
        ingredients = [a for a in ingredients if a["famille_olfactive"].lower() == famille.lower()]
    return ingredients


@app.get("/ingredients/{ingredient_id}", response_model=Ingredient, tags=["Catalogue"],
         summary="Détail d'un ingrédient")
def detail_ingredient(ingredient_id: str):
    for a in _rag().ingredients:
        if a["id"].lower() == ingredient_id.lower():
            return a
    raise HTTPException(status_code=404, detail="Ingrédient introuvable")
