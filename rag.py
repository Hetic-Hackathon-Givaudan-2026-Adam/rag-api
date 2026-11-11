"""
Moteur RAG : charge le catalogue, crée les embeddings, cherche les arômes
les plus proches d'une demande et fait répondre le LLM (Ollama, en local).
"""
import csv
import hashlib
import json
import os
import re

import numpy as np
import ollama

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
DOSSIER = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.getenv("CATALOGUE_CSV", os.path.join(DOSSIER, "data", "catalogue_aromes_demo.csv"))
CACHE_PATH = os.path.join(DOSSIER, "data", "embeddings_cache_v2.npz")

SYSTEM_PROMPT = """Tu es le conseiller virtuel de la plateforme de commande d'arômes alimentaires Givaudan.
Tu aides des clients, souvent non experts, à trouver l'arôme adapté à leur projet.

Règles strictes :
1. Tu proposes UNIQUEMENT des arômes présents dans les extraits du catalogue fournis. Tu cites leur nom exact et leur identifiant (ex. AR007).
2. Tu n'inventes JAMAIS de produit, de prix, de délai ou de caractéristique.
3. Si aucun arôme du catalogue ne correspond vraiment, dis-le honnêtement et propose d'être mis en relation avec un commercial Givaudan.
4. Vérifie que les "Applications" de l'arôme correspondent au produit du client (glace, boisson, sauce, biscuit...). Un arôme salé ne convient jamais à un produit sucré (glace, dessert, confiserie), et inversement. L'ambiance compte, mais l'usage passe en premier. Écarte les extraits qui ne conviennent pas.
5. S'il manque une information importante pour bien conseiller (type de produit fabriqué, quantité souhaitée, préférence naturel ou non), pose UNE ou DEUX questions simples à la fin de ta réponse.
6. Parle simplement, sans jargon technique, avec un ton chaleureux et professionnel (image premium).
7. Réponds en français, en 150 mots maximum. Propose 1 à 3 arômes au plus, en expliquant en une phrase pourquoi chacun correspond.
"""


# Mots qui indiquent le type de produit du client (comparés en minuscules, au singulier
# pour attraper aussi le pluriel : "glace" trouve "glaces").
PRODUITS_SUCRES = [
    "glace", "sorbet", "dessert", "gâteau", "gateau", "pâtisserie", "patisserie", "biscuit",
    "bonbon", "confiserie", "chocolat", "yaourt", "crème dessert", "smoothie", "céréale",
    "sirop", "confiture", "pâte à tartiner", "chewing-gum", "compote",
]
PRODUITS_SALES = [
    "sauce", "chips", "soupe", "plat préparé", "plats préparés", "apéritif", "aperitif",
    "snack", "marinade", "bouillon", "risotto", "charcuterie", "poisson", "huile",
]
FAMILLES_SUCREES = ["Vanillé", "Fruits rouges", "Fruits exotiques", "Fruits du verger", "Gourmand"]


def charger_catalogue(chemin=CSV_PATH):
    with open(chemin, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fiche_texte(a):
    """Transforme une ligne du CSV en texte lisible (pour l'embedding et pour le LLM)."""
    return (
        f"[{a['id']}] {a['nom']} — famille : {a['famille']}\n"
        f"Goût : {a['profil_gout']}\n"
        f"Description : {a['description']}\n"
        f"Applications : {a['applications']}\n"
        f"Ambiances / univers : {a['ambiances_client']}\n"
        f"Forme : {a['forme']} | Origine : {a['origine']} | Dosage conseillé : {a['dosage_conseille']}\n"
        f"Quantité minimum : {a['quantite_min_kg']} kg | Échantillon sous {a['delai_echantillon_jours']} jours | "
        f"Prix indicatif : {a['prix_indicatif_eur_kg']} €/kg"
    )


def fiche_usage(a):
    """Texte centré sur l'usage du produit : sert à mieux respecter le type de produit du client."""
    return f"{a['nom']}. Utilisé pour : {a['applications']}. Goût : {a['profil_gout']}. Famille : {a['famille']}."


def _normaliser(m):
    m = np.asarray(m, dtype=np.float32)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def _embed(textes):
    reponse = ollama.embed(model=EMBED_MODEL, input=textes)
    return _normaliser(reponse["embeddings"])


class CatalogueRAG:
    def __init__(self):
        self.aromes = charger_catalogue()
        self.textes = [fiche_texte(a) for a in self.aromes]
        self.textes_usage = [fiche_usage(a) for a in self.aromes]
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
        """Renvoie les k arômes les plus proches de la demande, avec un score (0 à 1).
        Le score mélange la fiche complète (ambiance) et l'usage (type de produit)."""
        q = _embed([texte])[0]
        scores = 0.5 * (self.vecteurs @ q) + 0.5 * (self.vecteurs_usage @ q)

        # Règles métier (dans le code, car un petit modèle ne les respecte pas toujours) :
        texte_min = texte.lower()
        produits = [p for p in PRODUITS_SUCRES + PRODUITS_SALES if p in texte_min]
        demande_sucree = any(p in PRODUITS_SUCRES for p in produits)
        demande_salee = any(p in PRODUITS_SALES for p in produits)
        for i, a in enumerate(self.aromes):
            applications = a["applications"].lower()
            # 1. Bonus si l'arôme est prévu pour le produit du client (ex. "glaces").
            if any(p in applications for p in produits):
                scores[i] += 0.15
            # 2. Jamais d'arôme salé pour un produit sucré, et inversement.
            if demande_sucree and not demande_salee and a["famille"] == "Salé":
                scores[i] = -1
            if demande_salee and not demande_sucree and a["famille"] in FAMILLES_SUCREES:
                scores[i] = -1

        meilleurs = np.argsort(scores)[::-1][:k]
        return [(self.aromes[i], float(scores[i])) for i in meilleurs if scores[i] > -1]

    def repondre(self, question, historique=None, k=3):
        historique = historique or []
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
            "content": f"Extraits du catalogue :\n{contexte}\n\nDemande du client : {question}",
        })

        reponse = ollama.chat(model=CHAT_MODEL, messages=messages, options={"temperature": 0.3})
        texte = reponse["message"]["content"]

        # On renvoie au front uniquement les arômes cités par le conseiller (dans l'ordre de citation).
        par_id = {a["id"]: (a, s) for a, s in candidats}
        cites = []
        for id_ in re.findall(r"AR\d{3}", texte):
            if id_ in par_id and par_id[id_] not in cites:
                cites.append(par_id[id_])
        return texte, cites[:k]
