"""
API du conseiller arômes Givaudan (hackathon HETIC).
Lancer :  uvicorn main:app --host 0.0.0.0 --port 8000
Swagger : http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import rag

etat = {"rag": None, "erreur": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        etat["rag"] = rag.CatalogueRAG()
    except Exception as e:  # Ollama éteint, modèle absent, CSV introuvable...
        etat["erreur"] = str(e)
    yield


app = FastAPI(
    title="Givaudan – Conseiller Arômes (RAG)",
    description=(
        "API de la plateforme de commande intelligente. Le client décrit son projet en langage courant, "
        "l'API cherche les arômes les plus proches dans le catalogue et un LLM local (Ollama) rédige la réponse.\n\n"
        "**Catalogue fictif** créé pour la démo (inspiré de FlavorDB)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Autorise l'interface du collègue (autre port / autre machine) à appeler l'API.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ---------- Modèles de données (apparaissent dans le Swagger) ----------

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class DemandeQuestion(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{
        "question": "Je cherche un arôme pour des glaces vendues dans un restaurant de plage",
        "historique": [],
        "nb_resultats": 3,
    }]}}

    question: str = Field(..., description="Demande du client, en langage courant.")
    historique: List[Message] = Field(
        default_factory=list,
        description="Messages précédents de la conversation (facultatif), pour continuer un échange.",
    )
    nb_resultats: int = Field(3, ge=1, le=5, description="Nombre d'arômes à chercher dans le catalogue.")


class Arome(BaseModel):
    id: str
    nom: str
    famille: str
    profil_gout: str
    description: str
    applications: str
    ambiances_client: str
    forme: str
    origine: str
    dosage_conseille: str
    quantite_min_kg: str
    delai_echantillon_jours: str
    prix_indicatif_eur_kg: str


class AromeTrouve(Arome):
    score: float = Field(..., description="Proximité avec la demande (plus c'est haut, plus c'est proche).")


class ReponseQuestion(BaseModel):
    reponse: str = Field(..., description="Réponse rédigée par le conseiller virtuel.")
    aromes_proposes: List[AromeTrouve] = Field(
        ..., description="Arômes cités dans la réponse (vide si le conseiller pose une question ou ne trouve rien)."
    )


# ---------- Routes ----------

def _rag():
    if etat["rag"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Le moteur RAG n'est pas prêt. Vérifie qu'Ollama est lancé. Détail : {etat['erreur']}",
        )
    return etat["rag"]


@app.get("/health", tags=["Système"], summary="Vérifier que l'API fonctionne")
def health():
    return {
        "statut": "ok" if etat["rag"] else "erreur",
        "modele_chat": rag.CHAT_MODEL,
        "modele_embedding": rag.EMBED_MODEL,
        "nb_aromes": len(etat["rag"].aromes) if etat["rag"] else 0,
        "erreur": etat["erreur"],
    }


@app.post("/ask", response_model=ReponseQuestion, tags=["Conseiller"],
          summary="Poser une question au conseiller arômes")
def ask(demande: DemandeQuestion):
    moteur = _rag()
    try:
        texte, resultats = moteur.repondre(
            demande.question,
            [m.model_dump() for m in demande.historique],
            k=demande.nb_resultats,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erreur Ollama : {e}")
    return ReponseQuestion(
        reponse=texte,
        aromes_proposes=[AromeTrouve(**a, score=round(s, 3)) for a, s in resultats],
    )


@app.get("/aromes", response_model=List[Arome], tags=["Catalogue"], summary="Lister le catalogue")
def lister_aromes(famille: Optional[str] = Query(None, examples=["Agrumes"])):
    aromes = _rag().aromes
    if famille:
        aromes = [a for a in aromes if a["famille"].lower() == famille.lower()]
    return aromes


@app.get("/aromes/{arome_id}", response_model=Arome, tags=["Catalogue"], summary="Détail d'un arôme")
def detail_arome(arome_id: str):
    for a in _rag().aromes:
        if a["id"].lower() == arome_id.lower():
            return a
    raise HTTPException(status_code=404, detail="Arôme introuvable")
