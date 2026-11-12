# Givaudan – Assistant de brief Fragrances (RAG local)

API FastAPI qui aide un client professionnel (marque de lessive, de shampoing, de savon...) à décrire
le parfum qu'il souhaite pour son produit. Givaudan crée des parfums **sur mesure** : l'assistant ne vend
rien, il propose 2 ou 3 **pistes olfactives** illustrées par de vrais ingrédients du catalogue, puis pose
une ou deux questions pour préparer le brief envoyé au commercial. Jamais de prix, de délai ni de promesse
de produit fini.

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
Le premier démarrage prend un peu de temps (calcul des embeddings du catalogue, mis en cache dans
`data/embeddings_cache_fragrances.npz`), les suivants sont rapides.

## 4. Tester
Ouvrir **http://localhost:8000/docs** (Swagger), puis `POST /ask` → *Try it out* → *Execute*.

## 5. Routes
| Route | Rôle |
|---|---|
| `GET /health` | Vérifier que l'API et Ollama fonctionnent (`nb_ingredients`) |
| `POST /ask` | Décrire son projet : réponse de l'assistant + ingrédients cités |
| `GET /ingredients` | Lister le catalogue (filtre `famille` facultatif, ex. `?famille=Boisé`) |
| `GET /ingredients/{id}` | Détail d'un ingrédient (ex. `/ingredients/FR026`) |

Exemple de corps pour `POST /ask` :
```json
{
  "question": "Je cherche un parfum frais et estival pour ma lessive",
  "historique": [],
  "nb_resultats": 3
}
```
Exemple de réponse (raccourcie) :
```json
{
  "reponse": "Bonjour, pour un parfum frais et estival pour votre lessive, voici quelques pistes : ... Florhydral™ (FR003) ... Quelle est la cible de votre produit ?",
  "ingredients_proposes": [
    { "id": "FR003", "nom": "Florhydral™", "famille_olfactive": "Floral", "notes": "Floral ; Vert ; Muguet ; Frais ; Puissant", "...": "...", "score": 0.71 }
  ]
}
```
`ingredients_proposes` est vide quand l'assistant pose seulement une question ou que la demande
ne concerne pas un projet de parfum (candidature, presse... → redirection polie vers givaudan.com).

Pour continuer une conversation, renvoyer les messages précédents dans `historique`
(`{"role": "user" | "assistant", "content": "..."}`).

## 6. Règles appliquées dans le code
- **Usage** : si le client cite son produit (lessive, linge, shampoing, savon, déodorant, nettoyant,
  javel...), les ingrédients recommandés pour ce produit sont favorisés et ceux déconseillés sont exclus.
  Pour un produit inconnu du catalogue (bougie, gel douche...), seule la proximité de sens compte.
- **Hors sujet** : candidature, presse, fournisseur, signalement éthique... → réponse fixe, sans conseil.
- Les identifiants d'ingrédients mal recopiés par le modèle sont corrigés.

## 7. Accès depuis un autre ordinateur (même Wi-Fi)
1. Sur le PC qui lance l'API : `ipconfig` → noter l'**Adresse IPv4** (ex. 192.168.1.25).
2. Autoriser Python si le pare-feu Windows le demande.
3. Le collègue ouvre `http://192.168.1.25:8000/docs`.

## 8. Partager l'API avec ngrok

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

## Changer de modèle ou de catalogue
Variables d'environnement `CHAT_MODEL`, `EMBED_MODEL` et `CATALOGUE_CSV`
(ex. `set CHAT_MODEL=qwen2.5:7b`).

## Données
`data/catalogue_fragrances.csv` : 30 ingrédients de parfumerie (molécules) issus de la bibliothèque
de molécules du site **givaudan.com** (lien source dans la colonne `url_source`). Données utilisées
uniquement pour une démo de hackathon (HETIC x Givaudan).
