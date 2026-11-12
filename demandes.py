"""
Compte client de démo, récapitulatifs et enregistrement des demandes (échantillon / expert).
Tout est fait par le code avec des textes modèles : fiable à 100 %, aucun appel au LLM.
Les demandes et le compte sont stockés dans la base SQLite (voir base.py).
"""
from datetime import datetime

import base

PREFIXES = {"echantillon": "ECH-DEMO", "expert": "EXP-DEMO"}
QUESTIONS_CONFIRMATION = {
    "echantillon": "Confirmez-vous cette demande et cette adresse ?",
    "expert": "Confirmez-vous que nous transmettions ce projet à un expert Givaudan ?",
}


def charger_compte():
    return base.compte()


def lister_demandes(type_=None):
    return base.lister_demandes(type_)


def enregistrer_demande(recap, session_id, brief):
    """Enregistre la demande en attente et renvoie son numéro (ECH-DEMO-001, EXP-DEMO-001...)."""
    type_ = recap["type"]

    def fabriquer(numero):
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
        return demande

    return base.ajouter_demande(fabriquer, type_, PREFIXES[type_])


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
