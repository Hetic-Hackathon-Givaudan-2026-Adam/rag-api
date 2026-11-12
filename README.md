# Givaudan – Assistant de brief Fragrances (RAG local)

API FastAPI qui aide un client professionnel (marque de lessive, de shampoing, de savon...) à décrire
le parfum qu'il souhaite pour son produit. Givaudan crée des parfums **sur mesure** : l'assistant ne vend
rien. Il commence par comprendre le besoin, propose ensuite 2 ou 3 **pistes olfactives** illustrées par de
vrais ingrédients du catalogue, puis affine avec le client pour préparer le brief envoyé au commercial.
Jamais de prix, de délai ni de promesse de produit fini.

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
Ouvrir **http://localhost:8000/docs** (Swagger) :
1. `POST /sessions` → *Try it out* → *Execute* : copier le `session_id`.
2. `POST /ask` → coller le `session_id` dans le corps, écrire la `question` → *Execute*.
3. Recommencer l'étape 2 avec le même `session_id` pour continuer la conversation.

## 5. Routes
| Route | Rôle |
|---|---|
| `GET /health` | Vérifier que l'API et Ollama fonctionnent (`nb_ingredients`) |
| `POST /sessions` | Démarrer une conversation : renvoie `session_id` et le message d'accueil |
| `POST /ask` | Envoyer un message du client : réponse, étape, brief, ingrédients cités |
| `GET /sessions/{id}` | Historique complet, brief et étape d'une conversation |
| `DELETE /sessions/{id}` | Supprimer une conversation (bouton « recommencer ») |
| `GET /ingredients` | Lister le catalogue (filtre `famille` facultatif, ex. `?famille=Boisé`) |
| `GET /ingredients/{id}` | Détail d'un ingrédient (ex. `/ingredients/FR026`) |

## 6. Fonctionnement : sessions et étapes

**L'API garde la conversation.** Le front n'envoie que le `session_id` et le nouveau message ; il n'a plus
besoin de renvoyer l'historique. Si `session_id` est absent ou inconnu, une nouvelle session est créée et
son identifiant est renvoyé. Les sessions sont gardées en mémoire (pas de base de données : elles sont
perdues au redémarrage) et supprimées au bout de 2 heures.

**Une fiche brief est remplie au fil des messages** : `produit`, `ambiance`, `cible`, `contraintes`, `pays`.
Seules les informations vraiment données par le client sont notées. Le modèle ne reçoit que les 10 derniers
messages, mais le brief garde l'essentiel.

**C'est le code qui choisit l'étape**, pas le modèle (le petit modèle `qwen2.5:3b` suit mal les consignes
longues) :

| Étape (`etape`) | Quand | Ce que fait l'assistant |
|---|---|---|
| `decouverte` | le produit ou l'ambiance n'est pas encore connu | reformule et pose une question ; **aucun ingrédient** (le catalogue ne lui est même pas fourni) |
| `pistes` | produit et ambiance connus, rien encore proposé | résume le besoin, propose 2 ou 3 pistes avec des ingrédients, demande laquelle plaît le plus |
| `affinage` | des pistes ont déjà été proposées | tient compte de la réaction du client, approfondit, demande ce qui manque encore (cible, contraintes...) ; quand le brief est complet, propose un récapitulatif pour les experts |
| `recentrage` | message hors sujet (météo, sport...) | phrase polie puis question pour revenir au projet ; brief inchangé |
| `redirection` | candidature, presse, fournisseur, signalement éthique | oriente vers la bonne page de givaudan.com |

La question de fin de chaque réponse est choisie par le code d'après ce qui manque dans le brief :
l'assistant ne repose donc jamais une question déjà répondue.

### Exemple de conversation en 3 messages

**1.** `POST /sessions` → `{"session_id": "63205d24-…", "message_accueil": "Bonjour et bienvenue chez Givaudan ! Parlez-moi de votre projet : quel parfum imaginez-vous, et pour quel produit ?"}`

**2.** `POST /ask`
```json
{ "session_id": "63205d24-…", "question": "Je cherche un parfum d'été", "nb_resultats": 3 }
```
```json
{
  "session_id": "63205d24-…",
  "reponse": "Parfait, vous souhaitez un parfum qui évoque l'été pour votre produit. C'est une idée très lumineuse.\n\nPour quel produit imaginez-vous ce parfum : une lessive, un shampoing, un savon, une bougie… ?",
  "etape": "decouverte",
  "brief": { "produit": "", "ambiance": "été", "cible": "", "contraintes": "", "pays": "" },
  "ingredients_proposes": []
}
```

**3.** `POST /ask`
```json
{ "session_id": "63205d24-…", "question": "C'est pour une lessive liquide", "nb_resultats": 3 }
```
```json
{
  "session_id": "63205d24-…",
  "reponse": "Pour une lessive liquide, voici quelques pistes olfactives qui évoquent l'été :\n\n… — Nectaryl (FR011) : une note fruitée et naturelle…\n\n… — Lemonile™ (FR006) : une note intense et fraîche…\n\nMenthe fraîche — Freskomenthe™ (FR024) : une note fraîche…\n\nQuelle piste vous parle le plus ?",
  "etape": "pistes",
  "brief": { "produit": "lessive", "ambiance": "été", "cible": "", "contraintes": "", "pays": "" },
  "ingredients_proposes": [
    { "id": "FR011", "nom": "Nectaryl", "famille_olfactive": "Fruité", "...": "...", "score": 0.68 },
    { "id": "FR006", "nom": "Lemonile™", "...": "..." },
    { "id": "FR024", "nom": "Freskomenthe™", "...": "..." }
  ]
}
```

**4.** `POST /ask`
```json
{ "session_id": "63205d24-…", "question": "Je préfère la piste marine, et ce serait pour des familles", "nb_resultats": 3 }
```
```json
{
  "session_id": "63205d24-…",
  "reponse": "Ultrazur™ (FR027) : une note fraîche et ozonique…\n\nNectaryl (FR011) : …idéale pour des familles.\n\nAvez-vous des contraintes particulières : ingrédients d'origine naturelle, biodégradabilité… ?",
  "etape": "affinage",
  "brief": { "produit": "lessive", "ambiance": "marine", "cible": "familles", "contraintes": "", "pays": "" },
  "ingredients_proposes": [ { "id": "FR027", "nom": "Ultrazur™", "...": "..." }, { "id": "FR011", "nom": "Nectaryl", "...": "..." } ]
}
```

`ingredients_proposes` ne contient que les ingrédients cités dans la réponse **et** fournis par le
catalogue : la liste est vide en découverte, recentrage et redirection.

## 7. Autres règles appliquées dans le code
- **Usage** : si le client cite son produit (lessive, linge, shampoing, savon, déodorant, nettoyant,
  javel...), les ingrédients recommandés pour ce produit sont favorisés et ceux déconseillés sont exclus.
  Pour un produit inconnu du catalogue (bougie, gel douche...), seule la proximité de sens compte.
- Les identifiants d'ingrédients oubliés ou mal recopiés par le modèle sont corrigés, et les passages
  recopiés de la réponse précédente sont retirés.

## 8. Accès depuis un autre ordinateur (même Wi-Fi)
1. Sur le PC qui lance l'API : `ipconfig` → noter l'**Adresse IPv4** (ex. 192.168.1.25).
2. Autoriser Python si le pare-feu Windows le demande.
3. Le collègue ouvre `http://192.168.1.25:8000/docs`.

## 9. Partager l'API avec ngrok

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

Pour `POST /ask`, Tech peut utiliser les corps JSON de la section 6. Le statut `/health` permet
de vérifier séparément si le moteur RAG est prêt. Lors du premier lancement, `pyngrok`
télécharge l'agent ngrok. [Documentation pyngrok](https://pyngrok.readthedocs.io/en/latest/)

## Changer de modèle ou de catalogue
Variables d'environnement `CHAT_MODEL`, `EMBED_MODEL` et `CATALOGUE_CSV`
(ex. `set CHAT_MODEL=qwen2.5:7b`).

## Données
`data/catalogue_fragrances.csv` : 30 ingrédients de parfumerie (molécules) issus de la bibliothèque
de molécules du site **givaudan.com** (lien source dans la colonne `url_source`). Données utilisées
uniquement pour une démo de hackathon (HETIC x Givaudan).
