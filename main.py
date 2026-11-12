"""
API de l'assistant de brief fragrances Givaudan (hackathon HETIC).
Lancer :  uvicorn main:app --host 0.0.0.0 --port 8000
Swagger : http://localhost:8000/docs   —   Page de test : http://localhost:8000/chat
"""
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import demandes
import base
import rag
import echantillons
import tts

etat = {"rag": None, "erreur": None}

# Les conversations et les demandes sont stockées dans la base SQLite data/givaudan.db (voir base.py) :
# elles survivent aux redémarrages de l'API.


@asynccontextmanager
async def lifespan(app: FastAPI):
    base.initialiser()
    try:
        etat["rag"] = rag.CatalogueRAG()
    except Exception as e:  # Ollama éteint, modèle absent, CSV introuvable...
        etat["erreur"] = str(e)
    # Charge la voix Piper par défaut en arrière-plan : la première lecture est ainsi immédiate.
    if tts.disponible():
        import threading
        threading.Thread(target=lambda: tts.synthetiser("Bonjour."), daemon=True).start()
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
    version="4.1.0",
    lifespan=lifespan,
)

# Autorise l'interface du collègue (autre port / autre machine) à appeler l'API.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ---------- Modèles de données (apparaissent dans le Swagger) ----------

Etape = Literal["decouverte", "propositions", "recapitulatif", "terminee", "selection", "adresse", "confirmation", "enregistree"]
EtapeReponse = Literal["decouverte", "propositions", "recapitulatif", "terminee", "recentrage", "redirection", "selection", "adresse", "confirmation", "enregistree"]
Parcours = Literal["brief", "echantillons_demo"]


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
    parcours: Parcours = "brief"


class DemandeSession(BaseModel):
    model_config = {"extra": "forbid", "json_schema_extra": {"examples": [{"parcours": "brief"}, {"parcours": "echantillons_demo"}]}}
    parcours: Parcours = Field("brief", description="brief : nouveau chat ingrédients et expert ; echantillons_demo : compositions fictives Bois doux et Bois frais.")


class DemandeQuestion(BaseModel):
    model_config = {"str_strip_whitespace": True, "extra": "forbid", "json_schema_extra": {"examples": [{
        "session_id": None,
        "question": "Je suis JB Industry, je veux créer une brume pour les sportifs pour les futurs Jeux olympiques en partenariat avec Nike.",
    }]}}

    session_id: Optional[str] = Field(
        None,
        description="Identifiant renvoyé par POST /sessions ou par un /ask précédent. "
                    "Absent ou inconnu : une nouvelle session est créée automatiquement.",
    )
    question: str = Field(..., min_length=1, description="Message du client, en langage courant.")
    nb_resultats: int = Field(2, ge=1, le=5, description="Conservé pour compatibilité : le conseiller propose toujours 2 ingrédients.")
    adresse_livraison: Optional[echantillons.AdresseLivraison] = Field(None, description="Adresse fictive réservée au parcours echantillons_demo.")


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
        default_factory=list, description="Réponses rapides à afficher en boutons (clic = envoyer ce texte comme question)."
    )
    numero_demande: Optional[str] = Field(None, description="Numéro de la demande enregistrée (étape terminee).")
    compositions_proposees: List[echantillons.CompositionDemo] = Field(default_factory=list)
    confirmation_echantillon: Optional[echantillons.PropositionEchantillon] = None
    demande_echantillon: Optional[echantillons.DemandeEnregistree] = None


class SessionDetail(BaseModel):
    session_id: str
    parcours: Parcours = "brief"
    etape: Etape
    brief: Brief
    ingredients_proposes: List[str] = Field(..., description="Identifiants des ingrédients déjà proposés.")
    recapitulatif_en_attente: Optional[Recapitulatif] = None
    messages: List[Message] = Field(..., description="Historique complet de la conversation.")
    cree_le: datetime
    adresse_livraison: Optional[echantillons.AdresseLivraison] = None
    confirmation_echantillon: Optional[echantillons.PropositionEchantillon] = None
    demande_echantillon: Optional[echantillons.DemandeEnregistree] = None


class Compte(BaseModel):
    entreprise: str
    contact: str
    email: str
    adresse: str


# ---------- Sessions ----------

def _session_vide(parcours="brief"):
    brief = {c: "" for c in rag.CHAMPS_TEXTE}
    brief["odeur_precise"] = False
    return {
        "messages": [{"role": "assistant", "content": rag.MESSAGE_ACCUEIL}],
        "brief": brief,
        "etape": "decouverte",
        "ingredients_proposes": [],
        "derniers_proposes": [],
        "recap_en_attente": None,
        "attente_choix_echantillon": False,
        "questions_contexte": 0,  # nombre de questions déjà posées sur l'entreprise / le projet
        "cree_le": datetime.now(),
        "parcours": parcours,
        "adresse_livraison": None,
        "confirmation_echantillon": None,
        "demande_echantillon": None,
    }


def _completer(session):
    """Ajoute les champs manquants (sessions enregistrées par une version précédente de l'API)."""
    vide = _session_vide()
    for cle, valeur in vide.items():
        session.setdefault(cle, valeur)
    for cle, valeur in vide["brief"].items():
        session["brief"].setdefault(cle, valeur)
    return session


def _nouvelle_session(parcours="brief", accueil=None):
    sid = str(uuid.uuid4())
    session = _session_vide(parcours)
    if accueil:
        session["messages"][0]["content"] = accueil
    base.enregistrer_session(sid, session)
    return sid


def _session(session_id):
    session = base.charger_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session introuvable (supprimée).")
    return _completer(session)


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
def creer_session(demande: Optional[DemandeSession] = None):
    parcours = demande.parcours if demande else "brief"
    accueil = "Bonjour ! Quel est votre besoin ?" if parcours == "echantillons_demo" else rag.MESSAGE_ACCUEIL
    sid = _nouvelle_session(parcours, accueil)
    return SessionCreee(session_id=sid, message_accueil=accueil, parcours=parcours)


@app.post("/ask", response_model=ReponseQuestion, tags=["Conversation"],
          summary="Envoyer un message du client à l'assistant de brief")
def ask(demande: DemandeQuestion):
    session = base.charger_session(demande.session_id)
    if session is None:
        sid, session = str(uuid.uuid4()), _session_vide()
    else:
        sid, session = demande.session_id, _completer(session)
    if session.get("parcours") == "echantillons_demo":
        resultat = echantillons.message_demo(session, demande.question, sid,
            demande.adresse_livraison.model_dump() if demande.adresse_livraison else None)
        base.enregistrer_session(sid, session)
        return ReponseQuestion(session_id=sid, brief=Brief(**session["brief"]), ingredients_proposes=[], **resultat)
    if demande.adresse_livraison:
        raise HTTPException(409, "L'adresse de livraison concerne le parcours echantillons_demo.")
    moteur = _rag()
    try:
        r = moteur.tour(session, demande.question, session_id=sid)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erreur Ollama : {e}")
    base.enregistrer_session(sid, session)  # étape, brief et nouveaux messages
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
        parcours=s.get("parcours", "brief"), adresse_livraison=s.get("adresse_livraison"),
        confirmation_echantillon=s.get("confirmation_echantillon"), demande_echantillon=s.get("demande_echantillon"),
    )


@app.delete("/sessions/{session_id}", tags=["Conversation"], summary="Supprimer une session (recommencer)")
def supprimer_session(session_id: str):
    if not base.supprimer_session(session_id):
        raise HTTPException(status_code=404, detail="Session introuvable (supprimée).")
    return {"supprimee": True, "session_id": session_id}


class ProjetResume(BaseModel):
    session_id: str
    titre: str = Field(..., description="Projet, sinon produit, sinon « Nouveau projet ».")
    entreprise: str
    produit: str
    odeur: str
    etape: Etape
    nb_messages: int = Field(..., description="Nombre de messages du client.")
    cree_le: datetime
    modifie_le: datetime


@app.get("/projets", response_model=List[ProjetResume], tags=["Conversation"],
         summary="Mes projets : conversations enregistrées, les plus récentes d'abord")
def lister_projets(limite: int = Query(50, ge=1, le=200), avec_vides: bool = Query(
        False, description="Inclure les conversations où le client n'a encore rien écrit.")):
    projets = []
    for s in base.lister_sessions(limite):
        if s["nb_messages"] == 0 and not avec_vides:
            continue
        brief = s["etat"].get("brief", {})
        titre = brief.get("projet") or brief.get("produit") or "Nouveau projet"
        projets.append(ProjetResume(
            session_id=s["id"], titre=titre[:1].upper() + titre[1:], entreprise=brief.get("entreprise", ""),
            produit=brief.get("produit", ""), odeur=brief.get("odeur", ""),
            etape=s["etat"].get("etape", "decouverte"), nb_messages=s["nb_messages"],
            cree_le=s["cree_le"], modifie_le=s["modifie_le"],
        ))
    return projets


@app.get("/echantillons", tags=["Demandes"], summary="Mes échantillons (demandes d'échantillon enregistrées)")
def lister_echantillons(session_id: Optional[str] = Query(None, min_length=1, description="Facultatif : filtrer une session ou retrouver ses compositions de démonstration.")):
    if session_id:
        composes = echantillons.lister(session_id)
        if composes or (base.charger_session(session_id) or {}).get("parcours") == "echantillons_demo":
            return composes
        return [d for d in demandes.lister_demandes("echantillon") if d.get("session_id") == session_id]
    return demandes.lister_demandes("echantillon")


@app.get("/demandes", tags=["Demandes"], summary="Toutes les demandes enregistrées")
def lister_toutes_demandes(type: Optional[Literal["echantillon", "expert"]] = Query(None)):
    return demandes.lister_demandes(type)


@app.get("/compte", response_model=Compte, tags=["Demandes"], summary="Compte client de démo")
def compte():
    return demandes.charger_compte()


class DemandeVoix(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{
        "texte": "Bonjour, je suis l'assistant Givaudan. Parlez-moi de votre projet de parfum.",
        "voix": "fr_FR-siwis-medium",
    }]}}

    texte: str = Field(..., max_length=3000, description="Texte à lire (les identifiants FRxxx sont retirés).")
    voix: Optional[str] = Field(None, description="Identifiant d'une voix de GET /tts/voix (par défaut : Siwis).")


@app.get("/tts/voix", tags=["Voix"], summary="Voix de synthèse locales installées (Piper)")
def voix_disponibles():
    return {"disponible": tts.disponible(), "voix": tts.lister_voix()}


@app.post("/tts", tags=["Voix"], summary="Lire un texte avec une voix locale (renvoie un fichier WAV)",
          response_class=Response, responses={200: {"content": {"audio/wav": {}}}})
def synthese_vocale(demande: DemandeVoix):
    if not tts.disponible():
        raise HTTPException(status_code=503, detail="Piper n'est pas installé ou aucune voix n'est dans data/voix.")
    try:
        son = tts.synthetiser(demande.texte, demande.voix)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de synthèse vocale : {e}")
    return Response(content=son, media_type="audio/wav")


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


def _session_demo(session_id):
    session = _session(session_id)
    if session.get("parcours") != "echantillons_demo":
        raise HTTPException(409, "Créer une session avec parcours=echantillons_demo pour utiliser les compositions fictives.")
    return session


@app.get("/compositions-demo", response_model=List[echantillons.CompositionDemo], tags=["Échantillons démo"],
         summary="Les deux compositions fictives pour le scénario savon boisé")
def compositions_demo():
    return echantillons.catalogue_demo()


@app.post("/sessions/{session_id}/adresse", response_model=echantillons.AdresseSession, tags=["Échantillons démo"],
          summary="Renseigner ou modifier l'adresse fictive de cette conversation")
def adresse_session(session_id: str, adresse: echantillons.AdresseLivraison):
    session = _session_demo(session_id)
    proposition = echantillons.enregistrer_adresse(session, adresse.model_dump())
    texte = echantillons.resume(proposition) if proposition else "L'adresse est renseignée pour cette conversation. Quelle composition souhaitez-vous essayer ?"
    if proposition:
        session["etape"] = "confirmation"
    session["messages"].append({"role": "assistant", "content": texte})
    base.enregistrer_session(session_id, session)
    return echantillons.AdresseSession(adresse_livraison=adresse, confirmation_echantillon=proposition, reponse=texte)


@app.post("/echantillons/preparer", response_model=echantillons.PropositionEchantillon, tags=["Échantillons démo"],
          summary="Préparer le récapitulatif sans enregistrer de demande")
def preparer_echantillon(demande: echantillons.Preparation):
    session = _session_demo(demande.session_id)
    proposition = echantillons.preparer(session, demande.reference,
                                      demande.adresse_livraison.model_dump() if demande.adresse_livraison else None)
    session["etape"] = "confirmation"
    session["messages"].append({"role": "assistant", "content": echantillons.resume(proposition)})
    base.enregistrer_session(demande.session_id, session)
    return proposition


@app.post("/echantillons/confirmer", response_model=echantillons.DemandeEnregistree, tags=["Échantillons démo"],
          summary="Confirmer et enregistrer une demande de démonstration")
def confirmer_echantillon(demande: echantillons.Confirmation):
    session = _session_demo(demande.session_id)
    resultat = echantillons.confirmer(session, demande.session_id, demande.jeton_confirmation, demande.confirme)
    session["etape"] = "enregistree"
    base.enregistrer_session(demande.session_id, session)
    return resultat


@app.get("/echantillons/{demande_id}", response_model=echantillons.DemandeEnregistree, tags=["Échantillons démo"],
         summary="Suivre le statut d'une demande de démonstration")
def detail_echantillon(demande_id: str, session_id: str = Query(..., min_length=1)):
    for demande in echantillons.lister(session_id):
        if demande["demande_id"] == demande_id:
            return demande
    raise HTTPException(404, "Demande introuvable pour cette session.")
