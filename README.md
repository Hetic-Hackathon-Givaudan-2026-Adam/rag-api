# Givaudan – Conseiller Arômes (RAG local)

API FastAPI qui recommande des arômes du catalogue à partir d'une demande client en langage courant.
Tout tourne en local avec Ollama (aucune API externe).

## 1. Prérequis
- Python 3.10 ou plus
- Ollama installé et lancé, avec les modèles :
  ```
  ollama pull qwen2.5:3b
  ollama pull bge-m3
  ```

## 2. Installation (une seule fois)
Dans le dossier du projet :
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Lancer l'API
```
venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000
```
Le premier démarrage prend un peu de temps (calcul des embeddings du catalogue), les suivants sont rapides.

## 4. Tester
Ouvrir **http://localhost:8000/docs** (Swagger), puis `POST /ask` → *Try it out* → *Execute*.

## 5. Routes
| Route | Rôle |
|---|---|
| `GET /health` | Vérifier que l'API et Ollama fonctionnent |
| `POST /ask` | Poser une question au conseiller (réponse + arômes proposés) |
| `GET /aromes` | Lister le catalogue (filtre `famille` facultatif) |
| `GET /aromes/{id}` | Détail d'un arôme |

Exemple de corps pour `POST /ask` :
```json
{
  "question": "Je cherche un arôme pour des glaces vendues dans un restaurant de plage",
  "historique": [],
  "nb_resultats": 3
}
```
Pour continuer une conversation, renvoyer les messages précédents dans `historique`
(`{"role": "user" | "assistant", "content": "..."}`).

## 6. Accès depuis un autre ordinateur (même Wi-Fi)
1. Sur le PC qui lance l'API : `ipconfig` → noter l'**Adresse IPv4** (ex. 192.168.1.25).
2. Autoriser Python si le pare-feu Windows le demande.
3. Le collègue ouvre `http://192.168.1.25:8000/docs`.

## Changer de modèle
Variables d'environnement `CHAT_MODEL` et `EMBED_MODEL` (ex. `set CHAT_MODEL=qwen2.5:7b`).

## Données
`data/catalogue_aromes_demo.csv` : catalogue **fictif** de 50 arômes créé pour la démo, inspiré de FlavorDB.
