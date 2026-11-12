"""
Moteur RAG : charge le catalogue d'ingrédients de parfumerie, crée les embeddings,
cherche les ingrédients les plus proches d'une demande et fait répondre le LLM
(Ollama, en local) dans le rôle d'assistant de brief.
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

SYSTEM_PROMPT = """Tu es l'assistant de brief de Givaudan (division Fragrance & Beauty).
Tu aides des clients professionnels (marques de lessive, de shampoing, de savon...), souvent non experts en parfumerie, à décrire le parfum qu'ils souhaitent pour leur produit.

Règles strictes :
1. Tu ne vends pas de parfum prêt à l'emploi : Givaudan crée des parfums sur mesure. Tu proposes des PISTES OLFACTIVES (ex. "une direction fraîche et marine"), chacune illustrée par 1 ou 2 ingrédients des extraits du catalogue, avec leur nom exact et leur identifiant (ex. Adoxal, FR026).
2. Tu cites UNIQUEMENT des ingrédients présents dans les extraits du catalogue fournis. Tu n'inventes jamais d'ingrédient ni de caractéristique.
3. Jamais de prix, de délai, de quantité minimum ou de promesse de résultat : c'est le rôle du commercial et du parfumeur.
4. Respecte l'usage : ne propose jamais un ingrédient "Déconseillé pour" le produit du client.
5. Pour bien comprendre le besoin, pose une ou deux questions simples à la fin de ta réponse, parmi : type de produit (lessive, shampoing, savon, bougie...), ambiance ou émotion recherchée, cible, contraintes (naturel, durable, biodégradable), pays de vente. Ne repose pas une question déjà répondue dans l'historique.
6. Si la demande ne concerne pas les fragrances (candidature, presse, fournisseur, signalement éthique...), ne donne aucun conseil ni ingrédient : indique poliment que ce chat est dédié aux projets de parfum et oriente vers la page contact appropriée du site givaudan.com.
7. Ton simple, chaleureux et professionnel (image premium). Pas de jargon sans l'expliquer en quelques mots.
8. Réponds en français, en 150 mots maximum, avec 2 à 3 pistes au plus.
"""


def _sans_accents(texte):
    """Minuscules et sans accents : "Déodorant" -> "deodorant"."""
    texte = unicodedata.normalize("NFD", texte.lower())
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")


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
    """Applications du catalogue correspondant aux mots du client (mot entier, pluriel accepté :
    "sols" trouve "sol", mais "soleil" ne le trouve pas)."""
    texte = _sans_accents(texte)
    trouvees = []
    for mot, applications in MOTS_APPLICATIONS.items():
        if re.search(r"\b" + re.escape(mot) + r"[sx]?\b", texte):
            trouvees += [a for a in applications if a not in trouvees]
    return trouvees


# Demandes hors sujet (candidature, presse, fournisseur, éthique...) : réponse fixe, sans appeler le LLM,
# car un petit modèle a tendance à proposer des ingrédients quand même.
MOTS_HORS_SUJET = [
    "postuler", "candidature", "candidat", "emploi", "embauche", "recrutement", "recrute", "stage",
    "alternance", "cv", "offre d'emploi", "travailler chez", "journaliste", "presse", "interview",
    "fournisseur", "devenir partenaire", "ethique", "signalement", "signaler", "lanceur d'alerte",
    "fraude", "corruption", "harcelement", "investisseur", "actionnaire", "action givaudan",
]

REPONSE_HORS_SUJET = (
    "Merci pour votre message ! Ce chat est dédié aux projets de parfum pour vos produits "
    "(lessive, shampoing, savon, bougie...), je ne peux donc pas vous aider sur ce sujet ici.\n\n"
    "Vous trouverez le bon interlocuteur sur le site givaudan.com : rubrique « Carrières » pour une "
    "candidature, « Médias » pour la presse, « Fournisseurs » pour proposer vos services, et la page "
    "« Contact » ou la ligne d'alerte éthique pour tout signalement.\n\n"
    "Et si vous avez un projet de parfum, je serai ravi de vous aider à le décrire !"
)


def est_hors_sujet(texte):
    texte = _sans_accents(texte)
    return any(re.search(r"\b" + re.escape(mot) + r"s?\b", texte) for mot in MOTS_HORS_SUJET)


RAPPEL_FORMAT = (
    "Consignes de réponse : une phrase d'introduction, puis 2 ou 3 pistes olfactives, une ligne chacune "
    "au format « Nom de la piste — Ingrédient (FRxxx) : pourquoi en une phrase ». Recopie l'identifiant "
    "exact écrit entre crochets devant chaque ingrédient. Termine par une ou deux questions simples pour "
    "préciser le brief. 150 mots maximum, aucun prix ni délai."
)


def _liste(valeur):
    """Case du CSV "a ; b ; c" -> ["a", "b", "c"] (vide -> [])."""
    return [v.strip() for v in (valeur or "").split(";") if v.strip()]


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
        """Renvoie les k ingrédients les plus proches de la demande, avec un score.
        Le score mélange la fiche complète et la fiche usage (ambiance + type de produit)."""
        q = _embed([texte])[0]
        scores = 0.5 * (self.vecteurs @ q) + 0.5 * (self.vecteurs_usage @ q)

        # Règles d'usage (dans le code, car un petit modèle ne les respecte pas toujours).
        # Si le client ne cite aucun produit connu (bougie, gel douche...), seule la similarité compte.
        demandees = applications_demandees(texte)
        if demandees:
            for i, a in enumerate(self.ingredients):
                recommandees = _liste(a["applications_recommandees"])
                deconseillees = _liste(a["applications_deconseillees"])
                if any(app in deconseillees for app in demandees):
                    scores[i] = -1  # exclu
                elif any(app in recommandees for app in demandees):
                    scores[i] += 0.15

        meilleurs = np.argsort(scores)[::-1][:k]
        return [(self.ingredients[i], float(scores[i])) for i in meilleurs if scores[i] > -1]

    def repondre(self, question, historique=None, k=3):
        historique = historique or []
        if est_hors_sujet(question):
            return REPONSE_HORS_SUJET, []
        # Pour la recherche, on tient compte des messages précédents du client.
        demandes_client = [m["content"] for m in historique if m.get("role") == "user"]
        texte_recherche = " ".join(demandes_client[-2:] + [question])
        # On donne plus de candidats au LLM, qui garde seulement ceux qui conviennent.
        candidats = self.rechercher(texte_recherche, k=max(6, k * 2))

        contexte = "\n\n".join(fiche_texte(a) for a, _ in candidats)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": m["role"], "content": m["content"]} for m in historique]
        messages.append({
            "role": "user",
            "content": f"Extraits du catalogue :\n{contexte}\n\n{RAPPEL_FORMAT}\n\nDemande du client : {question}",
        })

        reponse = ollama.chat(model=CHAT_MODEL, messages=messages, options={"temperature": 0.3})
        texte = reponse["message"]["content"]

        # Le petit modèle se trompe parfois d'identifiant ("Okoumal (FR012)") : on corrige
        # l'identifiant écrit juste après le nom d'un ingrédient du catalogue.
        for a in self.ingredients:
            nom = re.escape(a["nom"].replace("™", ""))
            texte = re.sub(rf"({nom}\W{{0,6}})FR\d{{3}}", rf"\g<1>{a['id']}", texte)

        # On renvoie au front uniquement les ingrédients cités (dans l'ordre de citation).
        par_id = {a["id"]: (a, s) for a, s in candidats}
        cites = []
        for id_ in re.findall(r"FR\d{3}", texte):
            if id_ in par_id and par_id[id_] not in cites:
                cites.append(par_id[id_])
        return texte, cites[:k]
