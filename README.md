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
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Lancer l'API
```
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
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

## 7. Partager l'API avec ngrok

Le script `expose.py` ouvre une URL HTTPS vers l'API locale. Le Swagger et les routes de l'API
sont accessibles à cette même adresse. Le tunnel reste actif pendant que le script tourne.

### Préparation

Créer un compte sur [ngrok](https://dashboard.ngrok.com/signup), puis récupérer le jeton
dans la rubrique [Your Authtoken](https://dashboard.ngrok.com/get-started/your-authtoken).

Dans PowerShell, depuis le dossier du projet, installer les dépendances et configurer le jeton
localement. Remplacer `TON_JETON` par la valeur du tableau de bord ngrok.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\ngrok.exe config add-authtoken TON_JETON
```

Si ton environnement s'appelle `venv`, utilise `venv` à la place de `.venv` dans ces commandes.
Le jeton est enregistré dans la configuration locale de ngrok. Le script accepte aussi la
variable d'environnement `NGROK_AUTHTOKEN`.

### Démarrage

Dans un premier terminal, lancer l'API.

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Dans un deuxième terminal, ouvrir le tunnel.

```powershell
.\.venv\Scripts\python.exe expose.py
```

Le script affiche l'adresse publique, le lien Swagger terminé par `/docs` et le lien de test
`/health`. Transmettre le lien Swagger à Tech et garder les deux terminaux ouverts pendant
les tests. Pour utiliser un autre port, lancer l'API sur ce port et exécuter
`python expose.py --port 8001`.

### Appel de test depuis Tech

Remplacer l'adresse de l'exemple par celle affichée par le script.

```powershell
Invoke-RestMethod -Uri 'https://VOTRE-ADRESSE.ngrok-free.app/health' -Headers @{
    'ngrok-skip-browser-warning' = 'true'
}
```

Pour `POST /ask`, Tech peut utiliser le corps JSON de la section 5. Le statut `/health` permet
de vérifier séparément si le moteur RAG est prêt. Lors du premier lancement, `pyngrok`
télécharge l'agent ngrok. [Documentation pyngrok](https://pyngrok.readthedocs.io/en/latest/)

## Changer de modèle
Variables d'environnement `CHAT_MODEL` et `EMBED_MODEL` (ex. `set CHAT_MODEL=qwen2.5:7b`).

## Données
`data/catalogue_aromes_demo.csv` : catalogue **fictif** de 50 arômes créé pour la démo, inspiré de FlavorDB.
