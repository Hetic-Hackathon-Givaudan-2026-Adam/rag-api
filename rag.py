"""
Moteur de l'assistant de brief fragrances.

Principe : le LLM sert à COMPRENDRE le client (fiche brief, intention) et à RÉDIGER de courtes
phrases (reformulation, description d'une odeur). Tout le reste est décidé et écrit par le CODE :
l'étape de la conversation, les questions, le choix des 2 ingrédients, les phrases de conclusion,
les récapitulatifs, la confirmation et l'enregistrement des demandes (fiable à 100 %).

Étapes : decouverte -> propositions -> recapitulatif -> terminee
         (+ recentrage pour un message hors sujet, redirection pour une demande non commerciale).
"""
import csv
import difflib
import hashlib
import json
import os
import re
import unicodedata

import numpy as np
import ollama

import demandes

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
DOSSIER = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.getenv("CATALOGUE_CSV", os.path.join(DOSSIER, "data", "catalogue_fragrances.csv"))
CACHE_PATH = os.path.join(DOSSIER, "data", "embeddings_cache_fragrances.npz")

NB_MESSAGES_LLM = 10  # seuls les 10 derniers messages sont envoyés au LLM (le brief garde l'essentiel)
NB_PROPOSITIONS = 2   # le conseiller propose exactement 2 ingrédients

CHAMPS_TEXTE = ["entreprise", "projet", "produit", "odeur", "cible", "contraintes", "pays"]
MAX_QUESTIONS_CONTEXTE = 2  # on demande le contexte (entreprise, projet) sans s'éterniser

MESSAGE_ACCUEIL = (
    "Bonjour et bienvenue chez Givaudan ! Pour commencer, présentez-moi en une phrase votre entreprise "
    "et le projet sur lequel vous travaillez."
)
MERCI_CONTEXTE = "Merci pour ces précisions, c'est un beau projet !"
QUESTION_SUITE = "Souhaitez-vous recevoir un échantillon ou être recontacté par un expert ?"
PHRASE_PARFUMEURS = "Leur intégration dans votre formule reste à valider avec nos parfumeurs."
SUGGESTIONS_SUITE = ["Recevoir un échantillon", "Être recontacté par un expert"]
SUGGESTIONS_CONFIRMATION = ["Oui, je confirme", "Annuler"]

SYSTEM_PROMPT = """Tu es l'assistant de brief de Givaudan (Fragrance & Beauty). Tu aides des clients professionnels à décrire le parfum qu'ils veulent pour leur produit (lessive, shampoing, savon...).
Règles :
- Tu parles uniquement du projet de parfum du client.
- Phrases courtes et simples, 60 mots maximum. Mots simples : frais, fruité, doux, boisé, tenace, linge propre...
- Aucun mot technique (aldéhyde, lactonique, ozonique, nitrile, molécule...) et aucun nom d'ingrédient absent des extraits.
- Jamais de prix, de délai, de quantité ni de promesse de résultat.
- Ne répète jamais une phrase déjà dite dans la conversation.
- Réponds en français."""


# ---------- Outils texte ----------

def _sans_accents(texte):
    """Minuscules et sans accents : "Déodorant" -> "deodorant"."""
    texte = unicodedata.normalize("NFD", (texte or "").lower())
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")


def _contient_mot(texte_normalise, mot):
    """Mot ou expression entière, pluriel accepté ("sols" trouve "sol", "soleil" ne le trouve pas)."""
    return re.search(r"\b" + re.escape(mot) + r"[sx]?\b", texte_normalise) is not None


def _liste(valeur):
    """Case du CSV "a ; b ; c" -> ["a", "b", "c"] (vide -> [])."""
    return [v.strip() for v in (valeur or "").split(";") if v.strip()]


def _majuscule(texte):
    return texte[:1].upper() + texte[1:] if texte else texte


def _phrase(texte):
    texte = texte.strip().rstrip(" ,;:")
    return _majuscule(texte) + ("" if texte.endswith((".", "!", "?")) else ".") if texte else ""


# Jargon -> mots simples (dans les fiches données au LLM et dans ce qu'il écrit).
JARGON = {
    "aldehyde": "propre", "aldehydee": "propre", "aldehydes": "notes propres", "lactonique": "crémeux",
    "ozonique": "aérien", "phenolique": "chaud", "balsamique": "doux et chaud", "galbanum": "vert intense",
    "graine d'ambrette": "musc doux", "musque": "doux et enveloppant", "musquee": "douce et enveloppante",
    "ambre": "chaud", "ambree": "chaude", "herbace": "herbes fraîches", "herbacee": "herbes fraîches",
    "feuillu": "feuilles vertes", "floral-vert": "floral et vert", "agreste": "campagne",
    "nitrile": "", "molecule": "ingrédient", "diffusif": "qui se diffuse bien",
}


def _simplifier_mot(mot):
    return JARGON.get(_sans_accents(mot.strip()), mot.strip())


def sans_jargon(texte):
    """Remplace les mots techniques restants (sauf dans le nom d'ingrédient « Aldehyde C11 »)."""
    def remplacer(m):
        mot = m.group(0)
        simple = JARGON.get(_sans_accents(mot))
        return mot if simple is None or mot == "Aldehyde" else simple
    return re.sub(r"[A-Za-zÀ-ÿ'-]+", remplacer, texte)


# ---------- Règles d'usage ----------

APPLICATIONS = ["nettoyant acide", "adoucissant", "antitranspirant", "shampoing", "nettoyant multi-usage",
                "lessive liquide", "savon", "lessive en poudre", "eau de javel"]
ARTICLES = {
    "lessive liquide": "la lessive liquide", "lessive en poudre": "la lessive en poudre",
    "adoucissant": "l'adoucissant", "shampoing": "le shampoing", "savon": "le savon",
    "antitranspirant": "les antitranspirants", "nettoyant multi-usage": "les nettoyants multi-usages",
    "nettoyant acide": "les nettoyants acides", "eau de javel": "l'eau de javel",
}

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


# Autres produits reconnus par le code (absents des applications du catalogue : pas de règle d'usage,
# seule la proximité de sens compte). Le mot seul « parfum » n'en fait pas partie : il est ambigu.
PRODUITS_CONNUS = [
    "eau de parfum", "eau de toilette", "eau de cologne", "extrait de parfum", "parfum fin", "fine fragrance",
    "brume parfumee", "brume", "bougie", "gel douche", "bain moussant", "creme", "lotion", "lait corporel",
    "huile de massage", "apres-shampoing", "apres shampoing", "masque capillaire", "deodorant",
    "diffuseur", "parfum d'ambiance", "spray d'ambiance", "desodorisant", "liquide vaisselle", "lingette",
    "dentifrice", "baume", "talc",
]


# Pays et zones de vente reconnus par le code (le modèle oublie parfois « pour la France »).
PAYS = {
    "france": "France", "belgique": "Belgique", "suisse": "Suisse", "luxembourg": "Luxembourg",
    "canada": "Canada", "quebec": "Québec", "allemagne": "Allemagne", "espagne": "Espagne", "italie": "Italie",
    "portugal": "Portugal", "royaume-uni": "Royaume-Uni", "angleterre": "Royaume-Uni", "pays-bas": "Pays-Bas",
    "etats-unis": "États-Unis", "usa": "États-Unis", "maroc": "Maroc", "algerie": "Algérie", "tunisie": "Tunisie",
    "senegal": "Sénégal", "cote d'ivoire": "Côte d'Ivoire", "japon": "Japon", "chine": "Chine", "inde": "Inde",
    "bresil": "Brésil", "mexique": "Mexique", "europe": "Europe", "afrique": "Afrique", "asie": "Asie",
    "monde": "Monde entier", "international": "International",
}


def pays_mentionne(texte):
    texte = _sans_accents(texte)
    trouves = [nom for cle, nom in PAYS.items() if _contient_mot(texte, cle)]
    return ", ".join(dict.fromkeys(trouves))


def produit_mentionne(texte):
    """Produit cité dans le message : application exacte du catalogue, puis produit courant, puis mot général."""
    texte = _sans_accents(texte)
    for liste in (APPLICATIONS, sorted(PRODUITS_CONNUS, key=len, reverse=True), list(MOTS_APPLICATIONS)):
        for p in liste:
            if _contient_mot(texte, _sans_accents(p)):
                return p
    return ""


def applications_demandees(texte):
    """Applications du catalogue correspondant aux mots du client. Un nom exact
    ("lessive liquide") est prioritaire sur un mot général ("lessive")."""
    texte = _sans_accents(texte)
    exactes = [a for a in APPLICATIONS if _contient_mot(texte, _sans_accents(a))]
    if exactes:
        return exactes
    trouvees = []
    for mot, applications in MOTS_APPLICATIONS.items():
        if _contient_mot(texte, mot):
            trouvees += [a for a in applications if a not in trouvees]
    return trouvees


# ---------- Demandes sans rapport avec un parfum ----------

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


# ---------- Odeur précise ou vague ----------

MOTS_ODEUR_PRECISE = [
    "linge", "propre", "coton", "talc", "savon", "agrume", "citron", "orange", "pamplemousse", "bergamote",
    "mandarine", "marin", "marine", "mer", "ocean", "iode", "aerien", "boise", "bois", "santal", "cedre",
    "floral", "fleur", "fleuri", "rose", "jasmin", "muguet", "lavande", "fruit", "fruite", "peche", "fraise",
    "framboise", "pomme", "ananas", "vanille", "gourmand", "menthe", "vert", "herbe", "epice", "epices",
    "musc", "musque", "ambre", "cuir", "tabac", "miel", "the", "rhubarbe", "safran", "iris", "poudre",
]
MOTS_ODEUR_VAGUE = ["frais", "fraiche", "fraicheur", "agreable", "bon", "bonne", "ete", "estival", "estivale",
                    "leger", "legere", "doux", "douce", "sympa", "plaisant", "naturel", "naturelle", "odeur",
                    "parfum", "une", "un", "d", "de", "l", "le", "la", "et", "tres", "plutot", "pour"]


def corriger_fautes_odeur(odeur):
    """Corrige les petites fautes de frappe sur les mots d'odeur (« lingue propre » -> « linge propre »)."""
    connus = set(MOTS_ODEUR_PRECISE) | set(MOTS_ODEUR_VAGUE)

    accents = {"boise": "boisé", "epice": "épicé", "fruite": "fruité", "aerien": "aérien", "peche": "pêche",
               "ocean": "océan", "cedre": "cèdre", "ambre": "ambré", "musque": "musqué", "the": "thé"}

    def corriger(m):
        mot = m.group(0)
        cle = _sans_accents(mot)
        # Mot connu, son pluriel ou son féminin (« agrumes », « aérienne ») : on n'y touche pas.
        if len(cle) < 4 or cle in connus or any(cle.startswith(c) for c in connus if len(c) >= 4):
            return mot
        proche = difflib.get_close_matches(cle, MOTS_ODEUR_PRECISE, n=1, cutoff=0.8)
        return accents.get(proche[0], proche[0]) if proche else mot
    return re.sub(r"[A-Za-zÀ-ÿ]+", corriger, odeur)


def odeur_est_precise(odeur, avis_llm=False):
    """Le code tranche avec des mots-clés ; l'avis du LLM ne sert que pour les mots inconnus."""
    mots = re.findall(r"[a-z]+", _sans_accents(odeur))
    if not mots:
        return False
    texte = " ".join(mots)
    # Mots longs : début du mot (« boise » trouve « boisée ») ; mots courts : mot entier (« mer », « the »).
    if any(re.search(r"\b" + re.escape(m), texte) if len(m) >= 5 else _contient_mot(texte, m)
           for m in MOTS_ODEUR_PRECISE):
        return True
    if all(m in MOTS_ODEUR_VAGUE for m in mots):
        return False
    return bool(avis_llm)


# ---------- Intentions (mots-clés, prioritaires sur le LLM) ----------

MOTS_INTENTION = {
    "echantillon": [r"echantillon", r"essayer", r"tester", r"recevoir"],
    "expert": [r"expert", r"parler a (quelqu|un|une)", r"recontact", r"rappele", r"commercial",
               r"quelqu'un", r"humain", r"etre contacte"],
    "infos": [r"c'est quoi", r"qu'est-ce", r"parle[rz]?[- ]moi", r"en savoir plus", r"plus d'info",
              r"details", r"comment sent", r"ca sent quoi"],
}
MOTS_CONFIRMER = r"\b(oui|je confirme|confirme|confirmer|c'est bon|d'accord|ok|valide|parfait)\b"
MOTS_ANNULER = r"\b(annuler|annule|annulez|non|pas maintenant|laisse tomber|plus tard)\b"


def intention_par_mots(message, recap_en_attente):
    texte = _sans_accents(message)
    if recap_en_attente:
        if re.search(MOTS_ANNULER, texte):
            return "annuler"
        if re.search(MOTS_CONFIRMER, texte):
            return "confirmer"
    for intention, motifs in MOTS_INTENTION.items():
        if any(re.search(m, texte) for m in motifs):
            return intention
    return ""


# ---------- Idées d'odeur selon le contexte du projet ----------

# (mots du contexte, libellé, directions d'odeur proposées). Chaque direction est une « odeur précise ».
INSPIRATIONS = [
    (r"sport|fitness|olymp|athlet|running|gym|energ|muscu", "un projet sportif",
     ["Fraîche et aérienne", "Agrumes énergisants", "Marine et vivifiante"]),
    (r"\bete\b|plage|soleil|vacances|estiv|tropical", "un projet estival",
     ["Agrumes pétillants", "Marine", "Fruitée et solaire"]),
    (r"luxe|haut de gamme|premium|prestige|soiree|nuit|elegan|chic", "un projet haut de gamme",
     ["Boisée et chaleureuse", "Florale élégante", "Ambrée et enveloppante"]),
    (r"bebe|enfant|famil|douceur|sensible|maman", "un projet familial",
     ["Linge propre", "Douce et poudrée", "Florale légère"]),
    (r"hiver|noel|cocooning|cosy|fetes", "un projet hivernal",
     ["Gourmande et vanillée", "Boisée et chaleureuse", "Épicée"]),
    (r"nature|ecolo|\bbio\b|durable|jardin|plante|vegetal", "un projet nature",
     ["Herbes fraîches", "Agrumes", "Florale légère"]),
]
INSPIRATION_DEFAUT = ("votre projet", ["Linge propre", "Agrumes", "Florale", "Boisée"])
MOTS_SANS_IDEE = (r"pas (trop |vraiment |encore )?d'idee|aucune idee|je (ne )?sais pas|sais pas trop|j'hesite|"
                  r"a vous de (voir|choisir|proposer)|vous (me )?conseill|que (me )?conseillez|surprenez|"
                  r"peu importe|n'importe|des idees|proposez|suggestion|inspire")


def inspiration(brief):
    """Directions d'odeur adaptées au contexte (projet, produit, cible)."""
    contexte = _sans_accents(" ".join(brief.get(c, "") for c in ("projet", "produit", "cible", "entreprise")))
    for motif, libelle, directions in INSPIRATIONS:
        if re.search(motif, contexte):
            return libelle, directions
    return INSPIRATION_DEFAUT


def client_sans_idee(message):
    return re.search(MOTS_SANS_IDEE, _sans_accents(message)) is not None


def reponse_inspiration(brief):
    libelle, directions = inspiration(brief)
    intro = ("Pas de souci, je vais vous aider ! Pour " + libelle + " comme le vôtre, ces directions fonctionnent "
             "souvent bien :") if libelle != INSPIRATION_DEFAUT[0] else \
        "Pas de souci, je vais vous aider ! Voici quelques directions qui plaisent souvent :"
    liste = "\n".join(f"- {d}" for d in directions)
    return f"{intro}\n{liste}\n\nLaquelle vous tente le plus ? Vous pouvez aussi en combiner deux.", directions


# ---------- Questions (écrites par le code) ----------

def _questions_decouverte(brief):
    """1 ou 2 questions sur ce qui manque : produit, puis odeur précise, puis pays.
    Les exemples d'odeur sont adaptés au contexte du projet (ex. sportif : fraîche et aérienne...)."""
    produit = _sans_accents(brief.get("produit", ""))
    odeur = _sans_accents(brief.get("odeur", ""))
    exemples = ", ".join(d.lower() for d in inspiration(brief)[1][:3]) + "…"
    questions = []
    if not brief.get("produit"):
        questions.append("pour quel produit imaginez-vous ce parfum (lessive, shampoing, savon, bougie…)")
    if not brief.get("odeur_precise"):
        if "frais" in odeur or "fraich" in odeur:
            questions.append("quel type de fraîcheur imaginez-vous (linge propre, agrumes, marine…)")
        elif odeur:
            questions.append(f"pouvez-vous préciser l'odeur souhaitée ({exemples})")
        else:
            questions.append(f"quelle odeur aimeriez-vous ({exemples})")
    if not brief.get("pays"):
        if "lessive" in produit:
            questions.append("dans quel pays la lessive sera-t-elle vendue")
        else:
            questions.append("dans quel pays votre produit sera-t-il vendu")
    questions = questions[:2]
    return _majuscule(" et ".join(questions)) + " ?" if questions else ""


def suggestions_decouverte(brief):
    """Boutons quand on demande l'odeur : directions adaptées au contexte + « Je n'ai pas d'idée »."""
    if brief.get("produit") and not brief.get("odeur_precise"):
        return inspiration(brief)[1][:3] + ["Je n'ai pas d'idée, conseillez-moi"]
    return []


def question_contexte(brief, nb_deja_posees):
    """Question sur l'entreprise et le projet, posée au plus 2 fois au total :
    une fois si rien n'est connu, puis une relance seulement s'il manque encore une partie."""
    entreprise, projet = brief.get("entreprise"), brief.get("projet")
    if not entreprise and not projet:
        if nb_deja_posees == 0:
            return ("Pour mieux vous accompagner, pouvez-vous me présenter en une phrase votre entreprise et "
                    "le projet sur lequel vous travaillez (occasion, public, partenaires…) ?")
        return ""
    if nb_deja_posees >= MAX_QUESTIONS_CONTEXTE:
        return ""
    if not projet:
        return (f"Merci ! Et en quelques mots, quel est le projet de {entreprise} derrière ce parfum "
                "(occasion, public, partenaires…) ?")
    if not entreprise:
        return "Pour le suivi de votre projet, pour quelle entreprise travaillez-vous ?"
    return ""


def decouverte_necessaire(brief):
    return not brief.get("produit") or not brief.get("odeur_precise") or not brief.get("pays")


# ---------- Mise à jour du brief par le LLM ----------

PROMPT_BRIEF = """Tu mets à jour la fiche brief d'un client qui décrit le parfum souhaité pour son produit.
Tu reçois la fiche actuelle, le dernier message du conseiller et le nouveau message du client.
Renvoie UNIQUEMENT un objet JSON avec ces clés :
- "entreprise" : nom de l'entreprise ou de la marque du client
- "projet" : le contexte du projet en quelques mots (occasion, objectif, gamme, partenaires...), sans répéter l'odeur
- "produit" : type de produit à parfumer (lessive liquide, adoucissant, shampoing, savon, bougie, eau de parfum, gel douche...)
- "odeur" : odeur ou ambiance souhaitée, avec les mots du client (frais, linge propre, agrumes, marine, boisé...)
- "odeur_precise" : false si l'odeur est vague ("frais", "agréable", "d'été", "bon"), true si elle est précise ("linge propre", "agrumes", "marine", "boisé chaleureux")
- "cible" : à qui s'adresse le produit (familles, enfants, jeunes adultes...)
- "contraintes" : écologique, naturel, biodégradable, hypoallergénique...
- "pays" : pays ou région de vente (France, Belgique, Europe...) ; jamais un nom d'entreprise
- "hors_sujet" : true si le message n'a aucun rapport avec un projet de parfum (météo, sport, actualité, blague...). Une réponse à une question du conseiller, un "oui", un "non" ou une question sur un ingrédient ne sont PAS hors sujet.
- "demande_autre" : "candidature", "presse", "fournisseur" ou "ethique" si le client parle d'emploi, de journalisme, de proposer ses services ou d'un signalement ; sinon "".
- "intention" : d'après le NOUVEAU message seulement : "echantillon" (veut essayer ou recevoir un ingrédient), "expert" (veut parler à quelqu'un), "confirmer" (oui, je confirme), "annuler", "infos" (pose une question sur un ingrédient) ou "".
- "ingredient_choisi" : nom ou identifiant (FRxxx) de l'ingrédient cité dans le NOUVEAU message, sinon "".
Règles :
1. Remplis seulement ce que le client a vraiment dit. N'invente rien.
2. Si le message ne parle pas d'un champ, recopie la valeur de la fiche actuelle.
3. Si le client précise ou change d'avis, remplace la valeur.
4. Valeurs courtes, en français.
Exemples :
- "Il fait beau aujourd'hui non ?" -> hors_sujet true, la fiche ne change pas
- "Je voudrais essayer Adoxal" -> intention "echantillon", ingredient_choisi "Adoxal"
- "Je préfère parler à quelqu'un" -> intention "expert"
- "Plutôt linge propre, pour la France" -> odeur "linge propre", odeur_precise true, pays "France"
- "Une eau de parfum qui sent le linge propre" -> produit "eau de parfum", odeur "linge propre"
- "Sephora, on veut une eau de parfum pour un projet sportif" -> entreprise "Sephora", projet "projet sportif", produit "eau de parfum" (pays reste "")
- "Je suis JB Industry, je veux créer une brume pour les sportifs pour les futurs Jeux olympiques en partenariat avec Nike" -> entreprise "JB Industry", projet "brume pour les Jeux olympiques, en partenariat avec Nike", produit "brume", cible "sportifs"
"""


def _ecrit_par_client(valeur, message):
    """Vrai si au moins un mot de la valeur (4 premières lettres) se retrouve dans le message :
    « fraîche » -> « frais » accepté, « familles » inventé pour un savon refusé."""
    mots_message = re.findall(r"[a-z]{3,}", _sans_accents(message))
    mots_valeur = re.findall(r"[a-z]{3,}", _sans_accents(valeur))
    return any(mv[:4] == mm[:4] for mv in mots_valeur for mm in mots_message)


def mettre_a_jour_brief(brief_actuel, message_client, dernier_message_assistant=""):
    """Renvoie (nouveau_brief, infos). infos = {hors_sujet, demande_autre, intention, ingredient_choisi}.
    Ne plante jamais : si le JSON est invalide, on garde l'ancien brief."""
    brief = {c: brief_actuel.get(c, "") for c in CHAMPS_TEXTE}
    brief["odeur_precise"] = bool(brief_actuel.get("odeur_precise"))
    infos = {"hors_sujet": False, "demande_autre": "", "intention": "", "ingredient_choisi": ""}
    avis_precise = brief["odeur_precise"]
    try:
        actuel = {c: brief[c] for c in CHAMPS_TEXTE}
        actuel["odeur_precise"] = brief["odeur_precise"]
        reponse = ollama.chat(
            model=CHAT_MODEL,
            format="json",
            options={"temperature": 0},
            messages=[
                {"role": "system", "content": PROMPT_BRIEF},
                {"role": "user", "content": (
                    f"Fiche actuelle : {json.dumps(actuel, ensure_ascii=False)}\n"
                    f"Dernier message du conseiller : {(dernier_message_assistant or '')[:600]}\n"
                    f"Nouveau message du client : {message_client}"
                )},
            ],
        )
        donnees = json.loads(reponse["message"]["content"])
        if not isinstance(donnees, dict):
            raise ValueError("JSON inattendu")
        infos["hors_sujet"] = str(donnees.get("hors_sujet")).lower() == "true"
        infos["demande_autre"] = str(donnees.get("demande_autre") or "").strip().lower()
        infos["intention"] = str(donnees.get("intention") or "").strip().lower()
        infos["ingredient_choisi"] = str(donnees.get("ingredient_choisi") or "").strip()
        if not infos["hors_sujet"]:  # un message hors sujet ne modifie jamais le brief
            for c in CHAMPS_TEXTE:
                valeur = donnees.get(c)
                if isinstance(valeur, list):
                    valeur = ", ".join(str(v) for v in valeur)
                # Une nouvelle valeur n'est acceptée que si le client l'a vraiment écrite
                # (le petit modèle invente parfois une cible ou reprend un mot du conseiller).
                if isinstance(valeur, str) and valeur.strip() and valeur.strip() != brief[c] \
                        and _ecrit_par_client(valeur, message_client):
                    brief[c] = valeur.strip()
            avis_precise = str(donnees.get("odeur_precise")).lower() == "true"
    except Exception:
        pass  # JSON invalide ou Ollama capricieux : on garde l'ancien brief

    # Filets de sécurité dans le code :
    if infos["demande_autre"] not in REDIRECTIONS:
        infos["demande_autre"] = ""
    infos["demande_autre"] = demande_autre_par_mots(message_client) or infos["demande_autre"]
    if infos["intention"] not in ("echantillon", "expert", "confirmer", "annuler", "infos"):
        infos["intention"] = ""
    produit_cite = produit_mentionne(message_client)
    if produit_cite:
        infos["hors_sujet"] = False  # le client parle de son produit : jamais hors sujet
        if not brief["produit"]:
            brief["produit"] = produit_cite
    if not infos["hors_sujet"] and pays_mentionne(message_client):
        brief["pays"] = pays_mentionne(message_client)
    # Un nom d'entreprise n'est pas un pays (le modèle a déjà rangé « Sephora » dans les deux).
    if brief["pays"] and _sans_accents(brief["pays"]).strip() in (
            _sans_accents(brief["entreprise"]).strip(), _sans_accents(brief_actuel.get("entreprise", "")).strip()):
        brief["pays"] = brief_actuel.get("pays", "") \
            if _sans_accents(brief_actuel.get("pays", "")) != _sans_accents(brief["entreprise"]) else ""
    brief["odeur"] = corriger_fautes_odeur(brief["odeur"])
    brief["odeur_precise"] = odeur_est_precise(brief["odeur"], avis_precise)
    return brief, infos


# ---------- Catalogue ----------

def charger_catalogue(chemin=CSV_PATH):
    with open(chemin, encoding="utf-8") as f:
        # Les cases vides (ou absentes) deviennent "".
        return [{k: (v or "").strip() for k, v in ligne.items()} for ligne in csv.DictReader(f)]


def fiche_texte(a):
    """Fiche complète (sert uniquement aux embeddings). Les lignes vides sont omises."""
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
    """Texte centré sur l'ambiance et l'usage (embeddings)."""
    morceaux = [a["nom"]]
    if a.get("famille_olfactive"):
        morceaux.append(f"Famille : {a['famille_olfactive']}")
    if a.get("notes"):
        morceaux.append(f"Notes : {a['notes']}")
    if a.get("applications_recommandees"):
        morceaux.append(f"Recommandé pour : {a['applications_recommandees']}")
    return ". ".join(morceaux) + "."


def notes_simples(a):
    mots = []
    for n in _liste(a.get("notes")):
        simple = _simplifier_mot(n).lower()
        if simple and simple not in ("puissant", "intense") and simple not in mots:
            mots.append(simple)
    return mots


# Adjectifs accordés avec « note » ; les autres notes sont des noms (« une touche de citron »).
FEMININ = {
    "floral": "florale", "boise": "boisée", "fruite": "fruitée", "frais": "fraîche", "marin": "marine",
    "vert": "verte", "epice": "épicée", "gourmand": "gourmande", "chaud": "chaude", "sucre": "sucrée",
    "cremeux": "crémeuse", "sec": "sèche", "propre": "propre", "aerien": "aérienne", "rose": "rosée",
    "doux et enveloppant": "douce et enveloppante", "doux et chaud": "douce et chaude",
    "floral et vert": "florale et verte", "vert intense": "verte et intense",
}


def _de(nom):
    return ("d'" if _sans_accents(nom)[:1] in "aeiouyh" or nom[:1] in "œŒ" else "de ") + nom


def phrase_odeur(a):
    """Phrase simple et juste, construite à partir des notes de la fiche :
    « note boisée et chaude, avec une touche d'iris »."""
    adjectifs, noms = [], []
    for n in notes_simples(a):
        cle = _sans_accents(n)
        if cle in FEMININ:
            adj = FEMININ[cle]
            # Pas de répétition : « verte » puis « verte et intense » -> on garde la plus complète.
            meme = next((i for i, x in enumerate(adjectifs) if x.split()[0] == adj.split()[0]), None)
            if meme is None:
                adjectifs.append(adj)
            elif len(adj) > len(adjectifs[meme]):
                adjectifs[meme] = adj
        elif n not in noms:
            noms.append(n)
    adjectifs, noms = adjectifs[:2], noms[:2]
    if adjectifs:
        phrase = "note " + (", ".join(adjectifs) if any(" et " in x for x in adjectifs) else " et ".join(adjectifs))
        if noms:
            phrase += ", avec une touche " + " et ".join(_de(n) for n in noms)
    elif noms:
        phrase = "note " + " et ".join(_de(n) for n in noms)
    else:
        phrase = "note originale"
    return phrase


def _phrase_fidele(phrase, a):
    """Vérifie qu'une phrase du LLM ne parle que d'odeurs présentes dans la fiche de CET ingrédient."""
    fiche = _sans_accents(" ".join([a["famille_olfactive"], a["notes"], a["description"]] + notes_simples(a)
                                   + [FEMININ.get(_sans_accents(n), "") for n in notes_simples(a)]))
    texte = _sans_accents(phrase)
    for mot in MOTS_ODEUR_PRECISE:
        if _contient_mot(texte, mot) and not re.search(r"\b" + re.escape(mot[:5]), fiche):
            return False
    # Une vraie phrase (pas une liste de mots : « propre, floral, frais »).
    return 3 <= len(phrase.split()) <= 30 and phrase.count(",") < 2 and re.search(r"\b(note|odeur|evoque)", texte)


def infos_utiles(a, brief):
    """Infos de la fiche utiles pour CE client (calculées par le code, pas inventées)."""
    infos = []
    contraintes = _sans_accents(brief.get("contraintes", ""))
    if re.search(r"eco|durable|biodegrad|naturel|environnement|planete|vert", contraintes):
        durabilite = _sans_accents(a.get("durabilite", ""))
        if "facilement biodegradable" in durabilite:
            infos.append("facilement biodégradable")
        elif "intrinsequement biodegradable" in durabilite:
            infos.append("biodégradable")
        if "carbone renouvelable : 100" in durabilite:
            infos.append("issu à 100 % de carbone renouvelable")
        if a.get("naturel_ou_synthetique"):
            infos.append(a["naturel_ou_synthetique"])
    tenue = _sans_accents(a.get("tenue", ""))
    sur_linge = any(app in ("lessive liquide", "lessive en poudre", "adoucissant")
                    for app in applications_demandees(brief.get("produit", "")))
    if "semaine" in tenue or "mois" in tenue:
        infos.append("très tenace sur le linge" if sur_linge else "très tenace")
    return infos


def fiche_simplifiee(a, brief):
    """Fiche courte donnée au LLM : pas de description technique (il la recopierait)."""
    lignes = [f"[{a['id']}] {a['nom']}"]
    if a.get("famille_olfactive"):
        famille = " - ".join(_simplifier_mot(f) for f in a["famille_olfactive"].split(" - "))
        lignes.append(f"Famille : {famille}")
    if notes_simples(a):
        lignes.append("Odeur : " + ", ".join(notes_simples(a)))
    if a.get("applications_recommandees"):
        lignes.append(f"Recommandé pour : {a['applications_recommandees']}")
    if a.get("applications_deconseillees"):
        lignes.append(f"Déconseillé pour : {a['applications_deconseillees']}")
    if a.get("tenue"):
        lignes.append(f"Tenue : {a['tenue']}")
    if a.get("durabilite"):
        lignes.append(f"Durabilité : {a['durabilite']}")
    if infos_utiles(a, brief):
        lignes.append("Info utile pour ce client : " + ", ".join(infos_utiles(a, brief)))
    return "\n".join(lignes)


def _normaliser(m):
    m = np.asarray(m, dtype=np.float32)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def _embed(textes):
    reponse = ollama.embed(model=EMBED_MODEL, input=textes)
    return _normaliser(reponse["embeddings"])


def nettoyer_reponse(texte, deja_dit):
    """Retire les questions écrites par le modèle (le code ajoute la sienne), les phrases déjà
    dites dans la conversation et le jargon."""
    deja = {_sans_accents(p).strip(" .!") for p in deja_dit if p.strip()}
    phrases = re.split(r"(?<=[.!?])\s+", texte.replace("\n", " "))
    gardees = [p.strip() for p in phrases
               if p.strip() and not p.strip().endswith("?") and _sans_accents(p).strip(" .!") not in deja]
    return sans_jargon(" ".join(gardees)).strip()


def _phrases_de(messages):
    return [p for m in messages if m["role"] == "assistant"
            for p in re.split(r"(?<=[.!?])\s+", m["content"].replace("\n", " "))]


# ---------- Moteur ----------

class CatalogueRAG:
    def __init__(self):
        self.ingredients = charger_catalogue()
        self.par_id = {a["id"]: a for a in self.ingredients}
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

    def rechercher(self, texte, k=3, exclure=(), odeur=""):
        """Renvoie les k ingrédients les plus proches du texte, avec un score.
        Bonus si recommandé pour le produit, exclusion si déconseillé, et petit bonus si la fiche
        contient un mot précis de l'odeur demandée (ex. « linge » pour « linge propre »)."""
        q = _embed([texte])[0]
        scores = 0.5 * (self.vecteurs @ q) + 0.5 * (self.vecteurs_usage @ q)
        demandees = applications_demandees(texte)
        odeur_n = _sans_accents(odeur)
        mots_odeur = [m for m in MOTS_ODEUR_PRECISE
                      if (re.search(r"\b" + re.escape(m), odeur_n) if len(m) >= 5 else _contient_mot(odeur_n, m))]
        if "frais" in odeur_n or "fraich" in odeur_n:
            mots_odeur.append("frais")  # « frais » seul est vague, mais aide à départager (note « Frais »)
        for i, a in enumerate(self.ingredients):
            notes = _sans_accents(" ".join([a["famille_olfactive"], a["notes"]] + notes_simples(a)))
            fiche = notes + " " + _sans_accents(a["description"])
            # Début du mot seulement : « marine » trouve « Marin », « agrumes » trouve « Agrume ».
            # Bonus cumulé : un ingrédient qui coche plusieurs mots de l'odeur passe devant.
            # « frais » ne compte que dans les notes (beaucoup de descriptions emploient le mot).
            # (en mot entier : sinon « frais » trouverait « fraise » !)
            touches = sum(1 for m in mots_odeur
                          if (re.search(r"\b(frais|fraiche)\b", notes) if m == "frais"
                              else re.search(r"\b" + re.escape(m[:5]), fiche)))
            scores[i] += min(0.08 * touches, 0.2)
            if a["id"] in exclure:
                scores[i] = -1
            elif demandees and any(app in _liste(a["applications_deconseillees"]) for app in demandees):
                scores[i] = -1  # exclu
            elif demandees and any(app in _liste(a["applications_recommandees"]) for app in demandees):
                scores[i] += 0.15
        meilleurs = np.argsort(scores)[::-1][:k]
        return [(self.ingredients[i], float(scores[i])) for i in meilleurs if scores[i] > -1]

    def trouver_ingredient(self, *textes):
        """Ingrédient du catalogue cité par son identifiant ou son nom (sans tenir compte des accents/™)."""
        for texte in textes:
            if not texte:
                continue
            m = re.search(r"FR\d{3}", texte.upper())
            if m and m.group(0) in self.par_id:
                return self.par_id[m.group(0)]
            normalise = _sans_accents(texte.replace("™", ""))
            for a in self.ingredients:
                if _contient_mot(normalise, _sans_accents(a["nom"].replace("™", ""))):
                    return a
        return None

    def _llm(self, session, consigne, contenu):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": m["role"], "content": m["content"]} for m in session["messages"][-NB_MESSAGES_LLM:]]
        messages.append({"role": "system", "content": consigne + "\nReste strictement sur le projet de fragrance du client."})
        messages.append({"role": "user", "content": contenu})
        reponse = ollama.chat(model=CHAT_MODEL, messages=messages, options={"temperature": 0.3})
        return reponse["message"]["content"].strip()

    # ----- Étapes -----

    @staticmethod
    def _decouverte(session, brief, ancien_brief, q_contexte):
        # Questions écrites par le code : le petit modèle ajoutait des détails inventés
        # dans ses reformulations (« légèrement fruité »...). Le LLM a déjà servi à comprendre
        # le message (fiche brief) ; c'est elle qui décide des questions.
        if q_contexte:
            session["questions_contexte"] = session.get("questions_contexte", 0) + 1
            return q_contexte
        merci = MERCI_CONTEXTE if brief.get("projet") and not ancien_brief.get("projet") else ""
        return f"{merci} {_questions_decouverte(brief)}".strip()

    def _propositions(self, session, question, brief):
        # L'odeur compte double : c'est elle qui doit guider le choix des ingrédients.
        texte_recherche = " ".join(
            # (le contexte du projet n'est pas utilisé ici : « Jeux olympiques, Nike » brouille la recherche d'odeur)
            v for v in (brief["produit"], brief["odeur"], brief["odeur"], brief["contraintes"], brief["cible"], question) if v
        )
        # Si le client veut autre chose, on écarte les ingrédients déjà proposés.
        exclure = session["derniers_proposes"] if re.search(
            r"\bautre|n'aime pas|pas fan|different|pas ca\b", _sans_accents(question)) else ()
        candidats = self.rechercher(texte_recherche, k=6, exclure=exclure, odeur=brief["odeur"])
        choisis = candidats[:NB_PROPOSITIONS]
        if not choisis:
            return ("Je ne trouve pas d'ingrédient de notre catalogue adapté à ce projet. "
                    f"{QUESTION_SUITE}"), []

        extraits = "\n\n".join(fiche_simplifiee(a, brief) for a, _ in choisis)
        a0 = choisis[0][0]
        # L'exemple reprend les vraies notes du 1er ingrédient (un exemple inventé est recopié tel quel).
        exemple = f"- {a0['nom']} — {a0['id']} : {phrase_odeur(a0)}."
        consigne = (
            f"ÉTAPE PROPOSITIONS : écris exactement {len(choisis)} lignes, une par ingrédient des extraits, "
            "et rien d'autre. Format : « - Nom — FRxxx : une phrase simple sur l'odeur. » Ajoute l'info utile "
            f"si elle est donnée. Exemple de forme : « {exemple} » Utilise seulement les informations de "
            "l'extrait de CHAQUE ingrédient, sans mélanger."
        )
        try:
            texte_llm = self._llm(session, consigne, f"Extraits du catalogue :\n{extraits}\n\nMessage du client : {question}")
        except Exception:
            texte_llm = ""

        lignes = []
        for a, _ in choisis:
            lignes.append(self._ligne_ingredient(a, brief, texte_llm))

        # Pour l'introduction, on garde la partie précise de l'odeur ("fraîche, linge propre" -> "linge propre").
        morceaux = [m.strip() for m in re.split(r",|/", brief["odeur"]) if m.strip()]
        odeur = next((m for m in morceaux if odeur_est_precise(m)), brief["odeur"]) or "votre projet"
        # « une odeur fraîche et marine » (adjectif) mais « une odeur de linge propre » (nom).
        adjectifs = {"frais", "fraiche", "marin", "marine", "boise", "boisee", "fruite", "fruitee", "floral",
                     "florale", "epice", "epicee", "vert", "verte", "doux", "douce", "sucre", "sucree", "chaud",
                     "chaude", "propre", "aerien", "aerienne", "gourmand", "gourmande", "poudre", "poudree",
                     "musque", "musquee", "ambre", "ambree", "leger", "legere", "fleuri", "fleurie"}
        premier = re.findall(r"[a-z]+", _sans_accents(odeur))[:1]
        if premier and premier[0] in adjectifs:
            de = ""
        else:
            de = "d'" if _sans_accents(odeur)[:1] in "aeiouyh" else "de "
        intro = (f"Deux ingrédients de notre catalogue évoquent une odeur {de}{odeur} :" if len(choisis) == 2
                 else f"Un ingrédient de notre catalogue évoque une odeur {de}{odeur} :")
        conclusion = []
        apps = applications_demandees(brief["produit"])
        commune = next((app for app in apps
                        if all(app in _liste(a["applications_recommandees"]) for a, _ in choisis)), "")
        if commune:
            sujet = "Les deux sont recommandés" if len(choisis) == 2 else "Il est recommandé"
            conclusion.append(f"{sujet} pour {ARTICLES[commune]}.")
        conclusion.append(PHRASE_PARFUMEURS if len(choisis) == 2 else
                          "Son intégration dans votre formule reste à valider avec nos parfumeurs.")
        conclusion.append(QUESTION_SUITE)
        texte = f"{intro}\n" + "\n".join(lignes) + "\n\n" + " ".join(conclusion)

        session["derniers_proposes"] = [a["id"] for a, _ in choisis]
        for a, _ in choisis:
            if a["id"] not in session["ingredients_proposes"]:
                session["ingredients_proposes"].append(a["id"])
        return texte, choisis

    def _ligne_ingredient(self, a, brief, texte_llm):
        """Ligne « - Nom — FRxxx : phrase. » : phrase du LLM si on la trouve, sinon phrase du code.
        Les infos utiles (biodégradable, tenace) sont ajoutées par le code si le LLM les a oubliées."""
        phrase = ""
        nom = re.escape(a["nom"].replace("™", "")) + "™?"
        # Le modèle écrit parfois tout sur une ligne : on découpe sur les tirets de liste.
        segments = [s for s in re.split(r"(?:^|\s)-\s+", texte_llm.replace("\n", " ")) if s.strip()]
        segment = next((s for s in segments if a["id"] in s), "") or \
            next((s for s in segments if re.search(nom, s, re.I)), "")
        apres = re.split(r"\s:\s|:\s", segment, maxsplit=1)
        if len(apres) == 2:
            phrase = re.sub(r"FR\d{3}", "", nettoyer_reponse(apres[1], [])).strip(" -—.")
        if not phrase or not _phrase_fidele(phrase, a):
            phrase = phrase_odeur(a)  # phrase sûre construite par le code
        phrase = _phrase(phrase)
        for info in infos_utiles(a, brief):
            cle = info.split()[-1] if "tenace" not in info else "tenace"
            if _sans_accents(cle) not in _sans_accents(phrase):
                phrase += f" {_phrase(info)}"
        phrase = phrase[:1].lower() + phrase[1:] if not phrase[:2].isupper() else phrase
        return f"- {a['nom']} — {a['id']} : {phrase}"

    def _infos(self, session, question, brief, cible):
        extraits = "\n\n".join(fiche_simplifiee(a, brief) for a in cible)
        try:
            texte = self._llm(
                session,
                "Le client demande des informations sur un ingrédient. Réponds en 2 phrases simples "
                "(40 mots maximum), uniquement avec les informations de l'extrait. Pas de question.",
                f"Extraits du catalogue :\n{extraits}\n\nMessage du client : {question}",
            )
            texte = nettoyer_reponse(texte, _phrases_de(session["messages"]))
        except Exception:
            texte = ""
        if not texte:
            texte = "\n".join(self._ligne_ingredient(a, brief, "") for a in cible)
        return f"{texte}\n\n{QUESTION_SUITE}"

    # ----- Tour de conversation -----

    def tour(self, session, question, session_id=""):
        """Traite un message du client (la session est modifiée sur place). Renvoie un dict :
        reponse, etape, ingredients [(ingredient, score)], recapitulatif, suggestions, numero_demande."""
        dernier_assistant = next(
            (m["content"] for m in reversed(session["messages"]) if m["role"] == "assistant"), ""
        )
        brief, infos = mettre_a_jour_brief(session["brief"], question, dernier_assistant)
        recap = session.get("recap_en_attente")
        intention = intention_par_mots(question, bool(recap)) or infos["intention"]
        if intention in ("confirmer", "annuler") and not recap:
            intention = intention if intention == "confirmer" else ""
        # Ingrédient choisi : seulement s'il est nommé dans CE message (ou « le premier / le deuxième »).
        ingredient = self.trouver_ingredient(question)
        rangs = {r"\b(premier|premiere|1er)\b": 0, r"\b(deuxieme|second|seconde|dernier|derniere|2e)\b": 1}
        for motif, rang in rangs.items():
            if not ingredient and re.search(motif, _sans_accents(question)) and len(session["derniers_proposes"]) > rang:
                ingredient = self.par_id[session["derniers_proposes"][rang]]

        def resultat(reponse, etape, ingredients=(), recapitulatif=None, suggestions=(), numero=None):
            session["messages"].append({"role": "user", "content": question})
            session["messages"].append({"role": "assistant", "content": reponse})
            if etape not in ("recentrage", "redirection"):
                session["etape"] = etape
            return {"reponse": reponse, "etape": etape, "ingredients": list(ingredients),
                    "recapitulatif": recapitulatif, "suggestions": list(suggestions), "numero_demande": numero}

        # a. Demande non commerciale : redirection fixe.
        if infos["demande_autre"]:
            return resultat(reponse_redirection(infos["demande_autre"]), "redirection")

        # b. Hors sujet : recentrage fixe, brief inchangé (« pas d'idée » n'est jamais hors sujet).
        if infos["hors_sujet"] and not client_sans_idee(question):
            ancien = session["brief"]
            if recap:
                question_retour, suggestions = demandes.QUESTIONS_CONFIRMATION[recap["type"]], SUGGESTIONS_CONFIRMATION
            elif decouverte_necessaire(ancien):
                question_retour, suggestions = _questions_decouverte(ancien), suggestions_decouverte(ancien)
            elif session["derniers_proposes"]:
                question_retour, suggestions = QUESTION_SUITE, SUGGESTIONS_SUITE
            else:
                question_retour, suggestions = "Voulez-vous que je vous propose des ingrédients pour votre projet ?", []
            texte = ("Je préfère rester concentré sur votre projet de parfum, c'est là que je peux vous être "
                     f"utile ! {question_retour}")
            return resultat(texte, "recentrage", suggestions=suggestions)

        ancien_brief = session["brief"]
        session["brief"] = brief

        # c. Récapitulatif en attente de confirmation.
        if recap:
            if intention == "confirmer":
                numero = demandes.enregistrer_demande(recap, session_id, brief)
                session["recap_en_attente"] = None
                return resultat(demandes.message_confirmation(recap["type"], numero), "terminee", numero=numero)
            if intention == "annuler":
                session["recap_en_attente"] = None
                return resultat("Pas de problème, la demande est annulée. Puis-je vous aider autrement sur "
                                "votre projet ?", "propositions", suggestions=SUGGESTIONS_SUITE)
            if intention not in ("echantillon", "expert"):
                texte = ("Pour finaliser, j'ai besoin de votre confirmation.\n\n" + demandes.texte_recap(recap))
                return resultat(texte, "recapitulatif", recapitulatif=recap, suggestions=SUGGESTIONS_CONFIRMATION)
            session["recap_en_attente"] = None  # le client change de demande : on refait le récapitulatif

        # d. Découverte : contexte (entreprise, projet) demandé sans s'éterniser,
        #    puis tant que produit, odeur précise ou pays manquent.
        # Le client n'a pas d'idée d'odeur : on s'appuie sur le contexte du projet pour proposer des directions.
        if not brief["odeur_precise"] and client_sans_idee(question):
            texte, directions = reponse_inspiration(brief)
            return resultat(texte, "decouverte", suggestions=directions)
        q_contexte = question_contexte(brief, session.get("questions_contexte", 0))
        if q_contexte or decouverte_necessaire(brief):
            return resultat(self._decouverte(session, brief, ancien_brief, q_contexte), "decouverte",
                            suggestions=[] if q_contexte else suggestions_decouverte(brief))

        # e. Demande d'échantillon.
        if intention == "echantillon" or (session.get("attente_choix_echantillon") and ingredient):
            if not ingredient and len(session["derniers_proposes"]) == 1:
                ingredient = self.par_id[session["derniers_proposes"][0]]
            if not ingredient:
                proposes = [self.par_id[i] for i in session["derniers_proposes"]]
                if not proposes:
                    return resultat(self._propositions(session, question, brief)[0], "propositions",
                                    suggestions=SUGGESTIONS_SUITE)
                session["attente_choix_echantillon"] = True
                choix = " ou ".join(f"{a['nom']} ({a['id']})" for a in proposes)
                return resultat(f"Avec plaisir ! Quel ingrédient souhaitez-vous recevoir en échantillon : {choix} ?",
                                "propositions", suggestions=[f"Essayer {a['nom']}" for a in proposes])
            session["attente_choix_echantillon"] = False
            recap = demandes.recap_echantillon(ingredient, brief)
            session["recap_en_attente"] = recap
            return resultat(demandes.texte_recap(recap), "recapitulatif", recapitulatif=recap,
                            suggestions=SUGGESTIONS_CONFIRMATION)

        # f. Demande de contact avec un expert.
        if intention == "expert":
            interet = [ingredient] if ingredient else [self.par_id[i] for i in session["ingredients_proposes"]]
            recap = demandes.recap_expert(interet, brief)
            session["recap_en_attente"] = recap
            session["attente_choix_echantillon"] = False
            return resultat(demandes.texte_recap(recap), "recapitulatif", recapitulatif=recap,
                            suggestions=SUGGESTIONS_CONFIRMATION)

        # « Oui » après des propositions, sans préciser : on demande quoi.
        if intention == "confirmer" and session["derniers_proposes"]:
            return resultat("Avec plaisir ! Préférez-vous recevoir un échantillon ou être recontacté par un expert ?",
                            "propositions", suggestions=SUGGESTIONS_SUITE)

        # Question sur un ingrédient.
        cible = [ingredient] if ingredient else [self.par_id[i] for i in session["derniers_proposes"]]
        if intention == "infos" and cible:
            return resultat(self._infos(session, question, brief, cible), "propositions",
                            ingredients=[(a, 0.0) for a in cible], suggestions=SUGGESTIONS_SUITE)

        # g. Propositions : exactement 2 ingrédients.
        texte, choisis = self._propositions(session, question, brief)
        return resultat(texte, "propositions", ingredients=choisis, suggestions=SUGGESTIONS_SUITE)
