"""
Moteur RAG de l'assistant de brief fragrances.

Principe : c'est le CODE qui décide de l'étape de la conversation, pas le LLM
(qwen2.5:3b est petit et respecte mal les consignes longues).
  1. Le LLM met à jour la fiche brief à partir du message client (JSON).
  2. Le code choisit l'étape : redirection, recentrage, decouverte, pistes ou affinage.
  3. Le LLM rédige la réponse avec une consigne courte propre à l'étape
     (et les extraits du catalogue seulement aux étapes pistes et affinage).
"""
import csv
import hashlib
import json
import os
import re
import unicodedata

import numpy as np
import ollama

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
DOSSIER = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.getenv("CATALOGUE_CSV", os.path.join(DOSSIER, "data", "catalogue_fragrances.csv"))
CACHE_PATH = os.path.join(DOSSIER, "data", "embeddings_cache_fragrances.npz")

NB_MESSAGES_LLM = 10  # seuls les 10 derniers messages sont envoyés au LLM (le brief garde l'essentiel)

CHAMPS_BRIEF = ["produit", "ambiance", "cible", "contraintes", "pays"]
LIBELLES_BRIEF = {
    "produit": "Produit", "ambiance": "Ambiance / émotion", "cible": "Cible",
    "contraintes": "Contraintes", "pays": "Pays de vente",
}

MESSAGE_ACCUEIL = (
    "Bonjour et bienvenue chez Givaudan ! Parlez-moi de votre projet : "
    "quel parfum imaginez-vous, et pour quel produit ?"
)

QUESTION_RECAP = "Souhaitez-vous que je prépare un récapitulatif de votre projet pour nos experts ?"

SYSTEM_PROMPT = """Tu es l'assistant de brief de Givaudan (Fragrance & Beauty). Tu aides des clients professionnels à décrire le parfum qu'ils souhaitent pour leur produit (lessive, shampoing, savon, bougie...).
Règles :
- Tu parles uniquement du projet de parfum du client. Rien d'autre.
- Givaudan crée des parfums sur mesure : jamais de prix, de délai, de quantité ni de promesse de produit fini.
- Tu cites uniquement des ingrédients présents dans les extraits du catalogue fournis, avec leur nom et leur identifiant (ex. Adoxal, FR026). Sans extraits, tu ne cites aucun ingrédient.
- Ton simple, chaleureux et professionnel (image premium), sans jargon.
- Réponds en français, en 120 mots maximum."""


# ---------- Outils texte ----------

def _sans_accents(texte):
    """Minuscules et sans accents : "Déodorant" -> "deodorant"."""
    texte = unicodedata.normalize("NFD", (texte or "").lower())
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")


def _contient_mot(texte_normalise, mot):
    """Mot entier, pluriel accepté ("sols" trouve "sol", "soleil" ne le trouve pas)."""
    return re.search(r"\b" + re.escape(mot) + r"[sx]?\b", texte_normalise) is not None


def _liste(valeur):
    """Case du CSV "a ; b ; c" -> ["a", "b", "c"] (vide -> [])."""
    return [v.strip() for v in (valeur or "").split(";") if v.strip()]


# ---------- Règles d'usage ----------

# Mots du client (sans accents) -> applications du catalogue.
MOTS_APPLICATIONS = {
    "lessive": ["lessive liquide", "lessive en poudre"],
    "linge": ["lessive liquide", "lessive en poudre", "adoucissant"],
    "adoucissant": ["adoucissant"],
    "shampoing": ["shampoing"],
    "shampooing": ["shampoing"],
    "cheveux": ["shampoing"],
    "savon": ["savon"],
    "deodorant": ["antitranspirant"],
    "antitranspirant": ["antitranspirant"],
    "anti-transpirant": ["antitranspirant"],
    "nettoyant": ["nettoyant multi-usage", "nettoyant acide"],
    "menager": ["nettoyant multi-usage", "nettoyant acide"],
    "sol": ["nettoyant multi-usage", "nettoyant acide"],
    "surface": ["nettoyant multi-usage", "nettoyant acide"],
    "javel": ["eau de javel"],
}


def applications_demandees(texte):
    """Applications du catalogue correspondant aux mots du client."""
    texte = _sans_accents(texte)
    trouvees = []
    for mot, applications in MOTS_APPLICATIONS.items():
        if _contient_mot(texte, mot):
            trouvees += [a for a in applications if a not in trouvees]
    return trouvees


# ---------- Demandes sans rapport avec un parfum (filet de sécurité par mots-clés) ----------

MOTS_DEMANDE_AUTRE = {
    "candidature": ["postuler", "candidature", "candidat", "emploi", "embauche", "recrutement", "recrute",
                    "stage", "alternance", "cv", "travailler chez", "job"],
    "presse": ["journaliste", "presse", "interview", "reportage"],
    "fournisseur": ["fournisseur", "devenir partenaire", "proposer mes services", "proposer nos services"],
    "ethique": ["ethique", "signalement", "signaler", "lanceur d'alerte", "fraude", "corruption", "harcelement"],
}

REDIRECTIONS = {
    "candidature": "Pour une candidature, rendez-vous sur la rubrique « Carrières » du site givaudan.com, "
                   "où sont publiées toutes les offres.",
    "presse": "Pour toute demande presse, l'équipe Médias est joignable via la rubrique « Médias » du site givaudan.com.",
    "fournisseur": "Pour proposer vos services, la rubrique « Fournisseurs » du site givaudan.com vous indiquera la démarche.",
    "ethique": "Pour un signalement, Givaudan met à disposition une ligne d'alerte éthique confidentielle, "
               "accessible depuis la page « Contact » du site givaudan.com.",
}


def demande_autre_par_mots(texte):
    texte = _sans_accents(texte)
    for categorie, mots in MOTS_DEMANDE_AUTRE.items():
        if any(_contient_mot(texte, m) for m in mots):
            return categorie
    return ""


def reponse_redirection(categorie):
    orientation = REDIRECTIONS.get(
        categorie, "Vous trouverez le bon interlocuteur sur la page « Contact » du site givaudan.com."
    )
    return (
        "Merci pour votre message ! Ce chat est dédié aux projets de parfum pour vos produits, "
        f"je ne peux donc pas vous aider sur ce sujet ici. {orientation}\n\n"
        "Et si vous avez un projet de parfum, je serai ravi de vous aider à le décrire !"
    )


# ---------- Questions sur ce qui manque dans le brief ----------

def champs_manquants(brief):
    return [c for c in CHAMPS_BRIEF if not brief.get(c)]


def brief_complet(brief):
    """Assez d'informations pour proposer le récapitulatif (le pays est facultatif)."""
    return all(brief.get(c) for c in ("produit", "ambiance", "cible", "contraintes"))


def question_pour(champ, brief):
    produit = brief.get("produit") or "produit"
    return {
        "produit": "Pour quel produit imaginez-vous ce parfum : une lessive, un shampoing, un savon, une bougie… ?",
        "ambiance": f"Quelle ambiance aimeriez-vous que votre {produit} évoque : plutôt frais et léger, "
                    "ou chaud et enveloppant ?",
        "cible": f"À qui s'adresse votre {produit} : des familles, des jeunes adultes, des peaux sensibles… ?",
        "contraintes": "Avez-vous des contraintes particulières : ingrédients d'origine naturelle, "
                       "biodégradabilité… ?",
        "pays": "Dans quel pays votre produit sera-t-il vendu ?",
    }[champ]


def reponse_recentrage(brief, deja_propose):
    manquants = champs_manquants(brief)
    if manquants:
        question = question_pour(manquants[0], brief)
    elif deja_propose:
        question = QUESTION_RECAP
    else:
        question = "Voulez-vous que je vous propose quelques pistes olfactives pour votre projet ?"
    return f"Je préfère rester concentré sur votre projet de parfum, c'est là que je peux vous être utile ! {question}"


def texte_brief(brief):
    """Brief en une phrase (une liste "Produit : ..." est recopiée telle quelle par le petit modèle)."""
    connus = [f"{LIBELLES_BRIEF[c].lower()} = {brief[c]}" for c in CHAMPS_BRIEF if brief.get(c)]
    inconnus = [LIBELLES_BRIEF[c].lower() for c in CHAMPS_BRIEF if not brief.get(c)]
    texte = "Déjà connu (ne repose pas de question là-dessus) : " + ("; ".join(connus) if connus else "rien") + "."
    if inconnus:
        texte += " Encore inconnu : " + ", ".join(inconnus) + "."
    return texte


# ---------- Mise à jour du brief par le LLM ----------

PROMPT_BRIEF = """Tu mets à jour la fiche brief d'un client qui décrit le parfum souhaité pour son produit.
Tu reçois la fiche actuelle, le dernier message du conseiller et le nouveau message du client.
Renvoie UNIQUEMENT un objet JSON avec ces clés :
- "produit" : type de produit à parfumer (lessive, adoucissant, shampoing, savon, bougie, gel douche...)
- "ambiance" : ambiance, émotion, saison ou odeur souhaitée (frais, été, marin, boisé, cocooning...). "pour l'été" est une ambiance.
- "cible" : à qui s'adresse le produit (familles, enfants, jeunes adultes, peaux sensibles...)
- "contraintes" : naturel, durable, biodégradable, hypoallergénique, budget...
- "pays" : pays ou région de vente
- "hors_sujet" : true si le message n'a aucun rapport avec un projet de parfum (météo, sport, actualité, blague, culture générale...). Une réaction aux pistes proposées, un "oui", un "non", un "merci" ou une question sur un ingrédient ne sont PAS hors sujet.
- "demande_autre" : "candidature" (emploi, stage), "presse" (journaliste), "fournisseur" (proposer ses services) ou "ethique" (signalement) ; sinon "".
Règles :
1. Remplis seulement ce que le client a vraiment dit. N'invente rien, ne déduis rien.
2. Si le message ne parle pas d'un champ, recopie la valeur de la fiche actuelle.
3. Si le client précise ou change d'avis, complète ou remplace la valeur.
4. Sinon laisse "".
5. Valeurs courtes (quelques mots), en français.
Exemples de "hors_sujet" :
- "Il fait beau aujourd'hui non ?" -> true (la météo n'est pas un projet de parfum ; ne change pas la fiche)
- "Tu as vu le match hier ?" -> true
- "Qui est le président ?" -> true
- "Je préfère la deuxième piste" -> false
- "C'est pour un savon" -> false"""


def mettre_a_jour_brief(brief_actuel, message_client, dernier_message_assistant=""):
    """Renvoie (nouveau_brief, hors_sujet, demande_autre). Ne plante jamais :
    si le JSON est invalide, on garde l'ancien brief."""
    brief = {c: brief_actuel.get(c, "") for c in CHAMPS_BRIEF}
    hors_sujet, demande_autre = False, ""
    try:
        reponse = ollama.chat(
            model=CHAT_MODEL,
            format="json",
            options={"temperature": 0},
            messages=[
                {"role": "system", "content": PROMPT_BRIEF},
                {"role": "user", "content": (
                    f"Fiche actuelle : {json.dumps(brief, ensure_ascii=False)}\n"
                    f"Dernier message du conseiller : {(dernier_message_assistant or '')[:600]}\n"
                    f"Nouveau message du client : {message_client}"
                )},
            ],
        )
        donnees = json.loads(reponse["message"]["content"])
        if not isinstance(donnees, dict):
            raise ValueError("JSON inattendu")
        hors_sujet = donnees.get("hors_sujet") is True or str(donnees.get("hors_sujet")).lower() == "true"
        demande_autre = str(donnees.get("demande_autre") or "").strip().lower()
        if not hors_sujet:  # un message hors sujet ne modifie jamais le brief
            for c in CHAMPS_BRIEF:
                valeur = donnees.get(c)
                if isinstance(valeur, list):
                    valeur = ", ".join(str(v) for v in valeur)
                if isinstance(valeur, str) and valeur.strip():
                    brief[c] = valeur.strip()
    except Exception:
        pass  # JSON invalide ou Ollama capricieux : on garde l'ancien brief

    # Filets de sécurité dans le code :
    if demande_autre not in REDIRECTIONS:
        demande_autre = ""
    demande_autre = demande_autre_par_mots(message_client) or demande_autre
    produit_cite = next((m for m in MOTS_APPLICATIONS if _contient_mot(_sans_accents(message_client), m)), "")
    if produit_cite:
        hors_sujet = False  # le client parle de son produit : jamais hors sujet
        if not brief["produit"]:
            brief["produit"] = produit_cite
    return brief, hors_sujet, demande_autre


# ---------- Catalogue ----------

def charger_catalogue(chemin=CSV_PATH):
    with open(chemin, encoding="utf-8") as f:
        # Les cases vides (ou absentes) deviennent "".
        return [{k: (v or "").strip() for k, v in ligne.items()} for ligne in csv.DictReader(f)]


def fiche_texte(a):
    """Texte lisible d'un ingrédient (pour l'embedding et pour le LLM). Les lignes vides sont omises."""
    titre = f"[{a['id']}] {a['nom']}"
    if a.get("type"):
        titre += f" ({a['type']})"
    if a.get("famille_olfactive"):
        titre += f" — famille : {a['famille_olfactive']}"
    nature = " / ".join(v for v in (a.get("origine"), a.get("naturel_ou_synthetique")) if v)
    champs = [
        ("Notes", a.get("notes")),
        ("Description", a.get("description")),
        ("Recommandé pour", a.get("applications_recommandees")),
        ("Déconseillé pour", a.get("applications_deconseillees")),
        ("Tenue", a.get("tenue")),
        ("Origine / nature", nature),
        ("Durabilité", a.get("durabilite")),
    ]
    return "\n".join([titre] + [f"{nom} : {valeur}" for nom, valeur in champs if valeur])


def fiche_usage(a):
    """Texte centré sur l'ambiance et l'usage."""
    morceaux = [a["nom"]]
    if a.get("famille_olfactive"):
        morceaux.append(f"Famille : {a['famille_olfactive']}")
    if a.get("notes"):
        morceaux.append(f"Notes : {a['notes']}")
    if a.get("applications_recommandees"):
        morceaux.append(f"Recommandé pour : {a['applications_recommandees']}")
    return ". ".join(morceaux) + "."


def _normaliser(m):
    m = np.asarray(m, dtype=np.float32)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def _embed(textes):
    reponse = ollama.embed(model=EMBED_MODEL, input=textes)
    return _normaliser(reponse["embeddings"])


# ---------- Consignes par étape ----------

PAS_DE_QUESTION = "Ne pose aucune question : la question de fin sera ajoutée après ta réponse."


def consigne_etape(etape, brief, candidats=()):
    """Consigne courte et concrète pour l'étape (un petit modèle suit mieux un exemple qu'une règle)."""
    exemple = ""
    if candidats:
        a = candidats[0][0]
        exemple = f"Exemple de ligne : « Brise d'été — {a['nom']} ({a['id']}) : ce qu'il apporte, en une phrase courte. »"
    if etape == "decouverte":
        return (
            "ÉTAPE DÉCOUVERTE : tu ne proposes encore aucun ingrédient, aucune piste et aucun produit.\n"
            "Écris 1 ou 2 phrases naturelles : remercie le client et redis avec tes mots ce que tu as compris "
            f"de son projet. {PAS_DE_QUESTION}"
        )
    if etape == "pistes":
        return (
            "ÉTAPE PISTES : écris une phrase qui résume le besoin, puis 2 ou 3 pistes olfactives, une par "
            "ligne. Chaque piste a un nom évocateur et 1 ingrédient des extraits (son nom et son identifiant "
            "entre parenthèses), expliqué en une phrase courte avec tes mots (ne recopie pas la description). "
            f"{exemple}\n{PAS_DE_QUESTION}"
        )
    return (
        "ÉTAPE AFFINAGE : tiens compte de la réaction du client. Approfondis la piste qu'il préfère avec "
        "1 ou 2 ingrédients des extraits (nom et identifiant entre parenthèses) et explique en une phrase "
        f"courte ce que chacun apporte. Ne reviens pas sur ce qu'il a écarté. 80 mots maximum. {exemple}\n"
        f"{PAS_DE_QUESTION}"
    )


def question_de_fin(etape, brief):
    """La question de fin est choisie par le code, d'après ce qui manque dans le brief."""
    manquants = champs_manquants(brief)
    if etape == "decouverte":
        # En priorité ce qui manque pour proposer des pistes : produit, puis ambiance.
        return " ".join(question_pour(c, brief) for c in ("produit", "ambiance") if c in manquants)
    if etape == "pistes":
        return "Quelle piste vous parle le plus ?"
    if brief_complet(brief):
        return QUESTION_RECAP
    return question_pour(manquants[0], brief)


def nettoyer_reponse(texte, dernier_assistant):
    """Retire les questions écrites par le modèle (le code ajoute la sienne) et les
    paragraphes recopiés de la réponse précédente (défaut fréquent des petits modèles)."""
    deja_dit = {p.strip() for p in (dernier_assistant or "").split("\n\n") if p.strip()}
    paragraphes = []
    for p in texte.split("\n\n"):
        if p.strip() in deja_dit:
            continue
        p = re.sub(r"[^.!?\n]*\?", "", p)  # phrases qui se terminent par "?"
        p = "\n".join(l.rstrip() for l in p.split("\n") if l.strip())
        if p.strip():
            paragraphes.append(p.strip())
    return "\n\n".join(paragraphes)


# ---------- Moteur ----------

class CatalogueRAG:
    def __init__(self):
        self.ingredients = charger_catalogue()
        self.textes = [fiche_texte(a) for a in self.ingredients]
        self.textes_usage = [fiche_usage(a) for a in self.ingredients]
        self.vecteurs, self.vecteurs_usage = self._charger_ou_calculer_embeddings()

    def _charger_ou_calculer_embeddings(self):
        # Le cache évite de recalculer les embeddings à chaque démarrage.
        empreinte = hashlib.sha256(
            (EMBED_MODEL + json.dumps(self.textes + self.textes_usage, ensure_ascii=False)).encode("utf-8")
        ).hexdigest()
        if os.path.exists(CACHE_PATH):
            cache = np.load(CACHE_PATH)
            if str(cache["empreinte"]) == empreinte:
                return cache["vecteurs"], cache["vecteurs_usage"]
        vecteurs = _embed(self.textes)
        vecteurs_usage = _embed(self.textes_usage)
        np.savez(CACHE_PATH, vecteurs=vecteurs, vecteurs_usage=vecteurs_usage, empreinte=np.array(empreinte))
        return vecteurs, vecteurs_usage

    def rechercher(self, texte, k=3):
        """Renvoie les k ingrédients les plus proches du texte, avec un score.
        Le score mélange la fiche complète et la fiche usage (ambiance + type de produit)."""
        q = _embed([texte])[0]
        scores = 0.5 * (self.vecteurs @ q) + 0.5 * (self.vecteurs_usage @ q)

        # Règles d'usage (dans le code, car un petit modèle ne les respecte pas toujours).
        # Si aucun produit connu n'est cité (bougie, gel douche...), seule la similarité compte.
        demandees = applications_demandees(texte)
        if demandees:
            for i, a in enumerate(self.ingredients):
                if any(app in _liste(a["applications_deconseillees"]) for app in demandees):
                    scores[i] = -1  # exclu
                elif any(app in _liste(a["applications_recommandees"]) for app in demandees):
                    scores[i] += 0.15

        meilleurs = np.argsort(scores)[::-1][:k]
        return [(self.ingredients[i], float(scores[i])) for i in meilleurs if scores[i] > -1]

    def _corriger_identifiants(self, texte, candidats):
        """Le petit modèle se trompe d'identifiant ("Okoumal (FR012)") ou l'oublie ("Adoxal :").
        On corrige l'identifiant écrit juste après le nom d'un ingrédient du catalogue, puis on
        l'ajoute à la première mention d'un ingrédient des extraits qui n'en a pas."""
        for a in self.ingredients:
            nom = re.escape(a["nom"].replace("™", ""))
            texte = re.sub(rf"({nom}\W{{0,6}})FR\d{{3}}", rf"\g<1>{a['id']}", texte)
        for a, _ in candidats:
            nom = re.escape(a["nom"].replace("™", "")) + "™?"
            if a["id"] not in texte:
                texte = re.sub(rf"\b({nom})(?!\w)", rf"\g<1> ({a['id']})", texte, count=1)
        return texte

    def tour(self, session, question, k=3):
        """Traite un message du client dans une session (modifiée sur place).
        Renvoie (texte_reponse, etape, [(ingredient, score), ...])."""
        dernier_assistant = next(
            (m["content"] for m in reversed(session["messages"]) if m["role"] == "assistant"), ""
        )
        brief, hors_sujet, demande_autre = mettre_a_jour_brief(session["brief"], question, dernier_assistant)

        # 1. Demande sans rapport avec un parfum (candidature, presse...) : redirection fixe.
        if demande_autre:
            return self._enregistrer(session, question, reponse_redirection(demande_autre), "redirection", [])
        # 2. Message hors sujet : recentrage fixe, brief inchangé.
        if hors_sujet:
            texte = reponse_recentrage(session["brief"], bool(session["ingredients_proposes"]))
            return self._enregistrer(session, question, texte, "recentrage", [])

        session["brief"] = brief
        if not brief["produit"] or not brief["ambiance"]:
            etape = "decouverte"
        elif not session["ingredients_proposes"]:
            etape = "pistes"
        else:
            etape = "affinage"

        candidats = []
        if etape != "decouverte":
            texte_recherche = " ".join(
                v for v in (brief["produit"], brief["ambiance"], brief["contraintes"], brief["cible"], question) if v
            )
            candidats = self.rechercher(texte_recherche, k=max(6, k * 2))

        rappel = (
            f"{consigne_etape(etape, brief, candidats)}\n"
            f"{texte_brief(brief)}\n"
            "Reste strictement sur le projet de fragrance du client."
        )
        contenu_client = question
        if candidats:
            extraits = "\n\n".join(fiche_texte(a) for a, _ in candidats)
            contenu_client = f"Extraits du catalogue :\n{extraits}\n\nDemande du client : {question}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": m["role"], "content": m["content"]} for m in session["messages"][-NB_MESSAGES_LLM:]]
        messages.append({"role": "system", "content": rappel})
        messages.append({"role": "user", "content": contenu_client})

        reponse = ollama.chat(model=CHAT_MODEL, messages=messages, options={"temperature": 0.3})
        texte = nettoyer_reponse(reponse["message"]["content"].strip(), dernier_assistant)
        texte = self._corriger_identifiants(texte, candidats)
        texte = f"{texte}\n\n{question_de_fin(etape, brief)}".strip()

        # Seuls les ingrédients cités ET présents dans les extraits sont renvoyés (vide en découverte).
        par_id = {a["id"]: (a, s) for a, s in candidats}
        cites = []
        for id_ in re.findall(r"FR\d{3}", texte):
            if id_ in par_id and par_id[id_] not in cites:
                cites.append(par_id[id_])
        cites = cites[:k]

        session["etape"] = etape
        for a, _ in cites:
            if a["id"] not in session["ingredients_proposes"]:
                session["ingredients_proposes"].append(a["id"])
        return self._enregistrer(session, question, texte, etape, cites)

    @staticmethod
    def _enregistrer(session, question, texte, etape, cites):
        session["messages"].append({"role": "user", "content": question})
        session["messages"].append({"role": "assistant", "content": texte})
        return texte, etape, cites
