"""
Compte client de démo, récapitulatifs et enregistrement des demandes (échantillon / expert).
Tout est fait par le code avec des textes modèles : fiable à 100 %, aucun appel au LLM.
"""
import json
import os
import tempfile
import threading
from datetime import datetime

DOSSIER = os.path.dirname(os.path.abspath(__file__))
COMPTE_PATH = os.path.join(DOSSIER, "data", "compte_demo.json")
DEMANDES_PATH = os.getenv("DEMANDES_JSON", os.path.join(DOSSIER, "data", "demandes.json"))

PREFIXES = {"echantillon": "ECH-DEMO", "expert": "EXP-DEMO"}
QUESTIONS_CONFIRMATION = {
    "echantillon": "Confirmez-vous cette demande et cette adresse ?",
    "expert": "Confirmez-vous que nous transmettions ce projet à un expert Givaudan ?",
}

_verrou = threading.Lock()


def charger_compte():
    with open(COMPTE_PATH, encoding="utf-8") as f:
        return json.load(f)


def lister_demandes(type_=None):
    if not os.path.exists(DEMANDES_PATH):
        return []
    try:
        with open(DEMANDES_PATH, encoding="utf-8") as f:
            demandes = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(demandes, list):
        return []
    return [d for d in demandes if not type_ or d.get("type") == type_]


def _ecrire(demandes):
    """Écriture sûre : fichier temporaire puis remplacement (jamais de fichier à moitié écrit)."""
    dossier = os.path.dirname(DEMANDES_PATH)
    os.makedirs(dossier, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dossier, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(demandes, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DEMANDES_PATH)


def enregistrer_demande(recap, session_id, brief):
    """Enregistre la demande en attente et renvoie son numéro (ECH-DEMO-001, EXP-DEMO-001...)."""
    type_ = recap["type"]
    with _verrou:
        demandes = lister_demandes()
        rang = sum(1 for d in demandes if d.get("type") == type_) + 1
        numero = f"{PREFIXES[type_]}-{rang:03d}"
        demande = {
            "numero": numero,
            "type": type_,
            "statut": "Enregistrée",
            "date": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "compte": charger_compte(),
            "brief": dict(brief),
        }
        if type_ == "echantillon":
            demande["ingredient"] = recap["ingredient"]
        else:
            demande["ingredients_interet"] = recap["ingredients_interet"]
        demandes.append(demande)
        _ecrire(demandes)
    return numero


def message_confirmation(type_, numero):
    if type_ == "echantillon":
        return (f"Votre demande {numero} est enregistrée. Vous pouvez suivre son statut "
                "dans « Mes échantillons ».")
    return (f"Votre demande {numero} est enregistrée. Un expert Givaudan vous contactera "
            f"à l'adresse {charger_compte()['email']}.")


def _texte_projet(brief, avec_contraintes=True):
    morceaux = [brief.get("produit", "")]
    if avec_contraintes:
        morceaux.append(brief.get("contraintes", ""))
    morceaux.append(brief.get("pays", ""))
    return _majuscule(", ".join(m for m in morceaux if m))


def _majuscule(texte):
    return texte[:1].upper() + texte[1:] if texte else ""


def _ingredient(a):
    return {"id": a["id"], "nom": a["nom"]}


def recap_echantillon(ingredient, brief):
    compte = charger_compte()
    lignes = [
        ("Ingrédient", f"{ingredient['id']} — {ingredient['nom']}"),
        ("Entreprise", brief.get("entreprise", "")),
        ("Contexte", _majuscule(brief.get("projet", ""))),
        ("Votre projet", _texte_projet(brief)),
        ("Demande", "Un échantillon"),
        ("Adresse", compte["adresse"]),
    ]
    return {
        "type": "echantillon",
        "ingredient": _ingredient(ingredient),
        "lignes": [{"information": i, "detail": d} for i, d in lignes if d],
    }


def recap_expert(ingredients, brief):
    compte = charger_compte()
    lignes = [
        ("Entreprise", brief.get("entreprise", "")),
        ("Contexte", _majuscule(brief.get("projet", ""))),
        ("Votre projet", _texte_projet(brief, avec_contraintes=False)),
        ("Odeur recherchée", brief.get("odeur", "")),
        ("Cible", brief.get("cible", "")),
        ("Contraintes", brief.get("contraintes", "")),
        ("Ingrédients qui vous intéressent", ", ".join(f"{a['nom']} ({a['id']})" for a in ingredients)),
        ("Contact", f"{compte['contact']} — {compte['email']}"),
    ]
    return {
        "type": "expert",
        "ingredients_interet": [_ingredient(a) for a in ingredients],
        "lignes": [{"information": i, "detail": d} for i, d in lignes if d],
    }


def texte_recap(recap):
    """Version texte du récapitulatif (le front peut afficher le tableau structuré à la place)."""
    intro = ("Voici le récapitulatif de votre demande d'échantillon :" if recap["type"] == "echantillon"
             else "Voici le récapitulatif de votre projet pour un expert Givaudan :")
    lignes = "\n".join(f"- {l['information']} : {l['detail']}" for l in recap["lignes"])
    return f"{intro}\n{lignes}\n\n{QUESTIONS_CONFIRMATION[recap['type']]}"
