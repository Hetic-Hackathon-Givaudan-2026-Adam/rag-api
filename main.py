"""
API de l'assistant de brief fragrances Givaudan (hackathon HETIC).
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
    title="Givaudan – Assistant de brief Fragrances (RAG)",
    description=(
        "Assistant de brief pour des parfums sur mesure. Le client professionnel (marque de lessive, "
        "de shampoing, de savon...) décrit son envie en langage courant ; l'API cherche les ingrédients "
        "les plus proches dans le catalogue et un LLM local (Ollama) propose 2 ou 3 pistes olfactives, "
        "puis pose des questions pour préparer le brief envoyé au commercial. Jamais de prix ni de délai.\n\n"
        "**Catalogue réel** de 30 ingrédients de parfumerie issu de givaudan.com (bibliothèque de molécules), "
        "utilisé uniquement pour une démo de hackathon."
    ),
    version="2.0.0",
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
        "question": "Je cherche un parfum frais et estival pour ma lessive",
        "historique": [],
        "nb_resultats": 3,
    }]}}

    question: str = Field(..., description="Demande du client, en langage courant.")
    historique: List[Message] = Field(
        default_factory=list,
        description="Messages précédents de la conversation (facultatif), pour continuer un échange.",
    )
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
    reponse: str = Field(..., description="Réponse rédigée par l'assistant de brief.")
    ingredients_proposes: List[IngredientTrouve] = Field(
        ..., description="Ingrédients cités dans les pistes olfactives, vide si le conseiller pose seulement une question."
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
        "nb_ingredients": len(etat["rag"].ingredients) if etat["rag"] else 0,
        "erreur": etat["erreur"],
    }


@app.post("/ask", response_model=ReponseQuestion, tags=["Conseiller"],
          summary="Décrire son projet de parfum à l'assistant de brief")
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
        ingredients_proposes=[IngredientTrouve(**a, score=round(s, 3)) for a, s in resultats],
    )


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
