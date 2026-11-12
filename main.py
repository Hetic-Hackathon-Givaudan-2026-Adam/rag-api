"""
API de l'assistant de brief fragrances Givaudan (hackathon HETIC).
Lancer :  uvicorn main:app --host 0.0.0.0 --port 8000
Swagger : http://localhost:8000/docs   —   Page de test : http://localhost:8000/chat
"""
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import demandes
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
        "- **decouverte** : questions sur le produit, l'odeur précise et le pays, aucun ingrédient ;\n"
        "- **propositions** : exactement 2 ingrédients du catalogue, puis « échantillon ou expert ? » ;\n"
        "- **recapitulatif** : tableau à confirmer (échantillon ou contact expert) ;\n"
        "- **terminee** : demande enregistrée avec un numéro (ECH-DEMO-001, EXP-DEMO-001...).\n\n"
        "Les messages hors sujet sont recentrés (**recentrage**) et les demandes sans rapport "
        "(candidature, presse...) sont redirigées (**redirection**). Jamais de prix ni de délai.\n\n"
        "**Catalogue réel** de 30 ingrédients de parfumerie issu de givaudan.com (bibliothèque de molécules), "
        "utilisé uniquement pour une démo de hackathon. Compte client fictif."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

# Autorise l'interface du collègue (autre port / autre machine) à appeler l'API.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ---------- Modèles de données (apparaissent dans le Swagger) ----------

Etape = Literal["decouverte", "propositions", "recapitulatif", "terminee"]
EtapeReponse = Literal["decouverte", "propositions", "recapitulatif", "terminee", "recentrage", "redirection"]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Brief(BaseModel):
    """Ce que l'on sait du projet du client ("" = pas encore précisé)."""
    entreprise: str = Field("", description="Entreprise ou marque du client.")
    projet: str = Field("", description="Contexte du projet en quelques mots (occasion, partenaires...).")
    produit: str = ""
    odeur: str = ""
    odeur_precise: bool = Field(False, description="false si l'odeur est vague (« frais »), true si précise (« linge propre »).")
    cible: str = ""
    contraintes: str = ""
    pays: str = ""


class LigneRecap(BaseModel):
    information: str
    detail: str


class Recapitulatif(BaseModel):
    type: Literal["echantillon", "expert"]
    lignes: List[LigneRecap]


class SessionCreee(BaseModel):
    session_id: str
    message_accueil: str


class DemandeQuestion(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{
        "session_id": None,
        "question": "Je suis JB Industry, je veux créer une brume pour les sportifs pour les futurs Jeux olympiques en partenariat avec Nike.",
    }]}}

    session_id: Optional[str] = Field(
        None,
        description="Identifiant renvoyé par POST /sessions ou par un /ask précédent. "
                    "Absent ou inconnu : une nouvelle session est créée automatiquement.",
    )
    question: str = Field(..., description="Message du client, en langage courant.")
    nb_resultats: int = Field(2, ge=1, le=5, description="Conservé pour compatibilité : le conseiller propose toujours 2 ingrédients.")


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
    reponse: str = Field(..., description="Réponse de l'assistant (texte complet, récapitulatif compris).")
    etape: EtapeReponse = Field(..., description="Étape de cette réponse.")
    brief: Brief = Field(..., description="Fiche brief mise à jour après ce message.")
    ingredients_proposes: List[IngredientTrouve] = Field(
        ..., description="Ingrédients cités dans la réponse, vide si le conseiller pose seulement une question."
    )
    recapitulatif: Optional[Recapitulatif] = Field(
        None, description="Tableau à afficher (étape recapitulatif), sinon null."
    )
    suggestions: List[str] = Field(
        ..., description="Réponses rapides à afficher en boutons (clic = envoyer ce texte comme question)."
    )
    numero_demande: Optional[str] = Field(None, description="Numéro de la demande enregistrée (étape terminee).")


class SessionDetail(BaseModel):
    session_id: str
    etape: Etape
    brief: Brief
    ingredients_proposes: List[str] = Field(..., description="Identifiants des ingrédients déjà proposés.")
    recapitulatif_en_attente: Optional[Recapitulatif] = None
    messages: List[Message] = Field(..., description="Historique complet de la conversation.")
    cree_le: datetime


class Compte(BaseModel):
    entreprise: str
    contact: str
    email: str
    adresse: str


# ---------- Sessions ----------

def _purger_sessions():
    limite = datetime.now() - DUREE_SESSION
    for sid in [sid for sid, s in SESSIONS.items() if s["cree_le"] < limite]:
        del SESSIONS[sid]


def _nouvelle_session():
    _purger_sessions()
    sid = str(uuid.uuid4())
    brief = {c: "" for c in rag.CHAMPS_TEXTE}
    brief["odeur_precise"] = False
    SESSIONS[sid] = {
        "messages": [{"role": "assistant", "content": rag.MESSAGE_ACCUEIL}],
        "brief": brief,
        "etape": "decouverte",
        "ingredients_proposes": [],
        "derniers_proposes": [],
        "recap_en_attente": None,
        "attente_choix_echantillon": False,
        "questions_contexte": 0,  # nombre de questions déjà posées sur l'entreprise / le projet
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
        r = moteur.tour(session, demande.question, session_id=sid)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erreur Ollama : {e}")
    return ReponseQuestion(
        session_id=sid,
        reponse=r["reponse"],
        etape=r["etape"],
        brief=Brief(**session["brief"]),
        ingredients_proposes=[IngredientTrouve(**a, score=round(s, 3)) for a, s in r["ingredients"]],
        recapitulatif=r["recapitulatif"],
        suggestions=r["suggestions"],
        numero_demande=r["numero_demande"],
    )


@app.get("/sessions/{session_id}", response_model=SessionDetail, tags=["Conversation"],
         summary="Historique complet, brief et étape d'une session")
def detail_session(session_id: str):
    s = _session(session_id)
    return SessionDetail(
        session_id=session_id, etape=s["etape"], brief=Brief(**s["brief"]),
        ingredients_proposes=s["ingredients_proposes"], recapitulatif_en_attente=s["recap_en_attente"],
        messages=s["messages"], cree_le=s["cree_le"],
    )


@app.delete("/sessions/{session_id}", tags=["Conversation"], summary="Supprimer une session (recommencer)")
def supprimer_session(session_id: str):
    _session(session_id)
    del SESSIONS[session_id]
    return {"supprimee": True, "session_id": session_id}


@app.get("/echantillons", tags=["Demandes"], summary="Mes échantillons (demandes d'échantillon enregistrées)")
def lister_echantillons():
    return demandes.lister_demandes("echantillon")


@app.get("/demandes", tags=["Demandes"], summary="Toutes les demandes enregistrées")
def lister_toutes_demandes(type: Optional[Literal["echantillon", "expert"]] = Query(None)):
    return demandes.lister_demandes(type)


@app.get("/compte", response_model=Compte, tags=["Demandes"], summary="Compte client de démo")
def compte():
    return demandes.charger_compte()


@app.get("/chat", include_in_schema=False)
def page_de_test():
    """Page web simple pour tester une conversation : http://localhost:8000/chat"""
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_test.html"))


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
