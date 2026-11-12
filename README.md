# Givaudan – Assistant de brief Fragrances (RAG local)

API FastAPI qui aide un client professionnel (marque de lessive, de shampoing, de savon...) à décrire
le parfum qu'il souhaite pour son produit. Givaudan crée des parfums **sur mesure** : l'assistant ne vend
rien. Il commence par comprendre le besoin, propose ensuite **2 ingrédients** du catalogue, puis enregistre
une demande d'échantillon ou de contact avec un expert (compte client fictif pour la démo).
Jamais de prix, de délai ni de promesse de produit fini.

Tout tourne en local avec Ollama (aucune API externe).

## 1. Prérequis
- Python 3.10 ou plus
- Ollama installé et lancé, avec les modèles :
  ```
  ollama pull qwen3.5:4b
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
**Le plus simple : la page de chat.** Ouvrir **http://localhost:8000/chat** pour discuter avec l'assistant
comme un client. La page affiche l'étape et les ingrédients sous chaque réponse, les **réponses rapides**
en boutons, le **récapitulatif en tableau**, la fiche brief à droite, des messages de test en un clic,
un bouton **« Mes échantillons »** et un bouton « Recommencer ». Elle fonctionne aussi via ngrok
(`https://VOTRE-ADRESSE.ngrok-free.app/chat`). Le fichier est `chat_test.html`.

Avec le Swagger, ouvrir **http://localhost:8000/docs** :
1. `POST /sessions` → *Try it out* → *Execute* : copier le `session_id`.
2. `POST /ask` → coller le `session_id` dans le corps, écrire la `question` → *Execute*.
3. Recommencer l'étape 2 avec le même `session_id` pour continuer la conversation.

## 5. Routes
| Route | Rôle |
|---|---|
| `GET /health` | Vérifier que l'API et Ollama fonctionnent (`nb_ingredients`) |
| `POST /sessions` | Démarrer une conversation : renvoie `session_id` et le message d'accueil |
| `POST /ask` | Envoyer un message du client (voir les champs de réponse ci-dessous) |
| `GET /sessions/{id}` | Historique complet, brief, étape et récapitulatif en attente |
| `DELETE /sessions/{id}` | Supprimer une conversation (bouton « recommencer ») |
| `GET /echantillons` | Page « Mes échantillons » : demandes d'échantillon enregistrées |
| `GET /demandes` | Toutes les demandes (filtre `?type=echantillon` ou `?type=expert`) |
| `GET /compte` | Compte client fictif de la démo (`data/compte_demo.json`) |
| `GET /ingredients` | Lister le catalogue (filtre `famille` facultatif, ex. `?famille=Boisé`) |
| `GET /ingredients/{id}` | Détail d'un ingrédient (ex. `/ingredients/FR026`) |
| `GET /chat` | Page web de test |

**Champs de la réponse de `POST /ask`**

| Champ | Contenu |
|---|---|
| `session_id` | à renvoyer au message suivant |
| `reponse` | texte de l'assistant (récapitulatif compris) |
| `etape` | `decouverte`, `propositions`, `recapitulatif`, `terminee`, `recentrage` ou `redirection` |
| `brief` | `entreprise`, `projet`, `produit`, `odeur`, `odeur_precise`, `cible`, `contraintes`, `pays` |
| `ingredients_proposes` | fiches des ingrédients cités (vide si l'assistant pose seulement une question) |
| `recapitulatif` | `{type, lignes: [{information, detail}]}` à afficher en tableau, sinon `null` |
| `suggestions` | réponses rapides à afficher en boutons (clic = envoyer ce texte) |
| `numero_demande` | `ECH-DEMO-001`, `EXP-DEMO-001`... une fois la demande confirmée, sinon `null` |

## 6. Fonctionnement : sessions et étapes

**L'API garde la conversation.** Le front n'envoie que le `session_id` et le nouveau message. Les sessions
sont en mémoire (perdues au redémarrage) et supprimées au bout de 2 heures.

**Le LLM sert à comprendre le client et à rédiger** : à chaque message, il met à jour la fiche brief
(entreprise, projet, produit, odeur, cible, contraintes, pays) et repère l'intention (échantillon, expert, confirmer, annuler,
infos) et l'ingrédient choisi. Une information n'est retenue que si le client l'a vraiment écrite.

**Le code décide de tout le reste**, avec des textes modèles :

| Étape (`etape`) | Quand | Ce que fait l'assistant |
|---|---|---|
| `redirection` | candidature, presse, fournisseur, signalement éthique | oriente vers la bonne page de givaudan.com |
| `recentrage` | message hors sujet (météo, sport...) | phrase polie puis question pour revenir au projet ; brief inchangé |
| `decouverte` | contexte (entreprise, projet) pas encore demandé, ou produit, odeur **précise** ou pays manquant | d'abord le contexte (voir ci-dessous), puis 1 ou 2 questions sur ce qui manque (« frais » est trop vague : il demande quel type de fraîcheur) ; aucun ingrédient |
| `propositions` | brief suffisant | **exactement 2 ingrédients** du catalogue, puis « Souhaitez-vous recevoir un échantillon ou être recontacté par un expert ? » |
| `recapitulatif` | le client veut un échantillon ou un expert | tableau à confirmer (adresse ou e-mail du compte démo) |
| `terminee` | le client confirme | demande enregistrée dans `data/demandes.json`, avec un numéro |

Détails des propositions : les ingrédients déconseillés pour le produit sont exclus et ceux recommandés
favorisés ; une fiche simplifiée, sans jargon, est donnée au modèle ; chaque phrase qu'il écrit est vérifiée
(elle ne doit parler que d'odeurs présentes dans la fiche de l'ingrédient), sinon le code la remplace par
une phrase construite à partir des notes. Les infos utiles (« facilement biodégradable » si le client veut
de l'écologique, « très tenace sur le linge »...) sont ajoutées par le code.

**Contexte du projet, sans s'éterniser.** Le message d'accueil demande l'entreprise et le projet en une
phrase. Si le client n'en parle pas, l'assistant le demande **une fois** ; s'il manque encore une partie
(ex. l'entreprise est donnée mais pas le projet), il relance **une fois**. Au plus 2 questions, puis il passe
à l'odeur quoi qu'il arrive. Quand le contexte arrive, il répond « Merci pour ces précisions, c'est un beau
projet ! ». L'entreprise et le contexte apparaissent dans les récapitulatifs ; ils ne servent pas à choisir
les ingrédients (seuls le produit, l'odeur, la cible et les contraintes comptent).

**Le contexte aide le client qui n'a pas d'idée.** Les exemples de la question sur l'odeur sont adaptés au
projet (sportif : « fraîche et aérienne, agrumes énergisants, marine et vivifiante » ; estival, haut de gamme,
familial, hivernal, nature...) et proposés en boutons, avec « Je n'ai pas d'idée, conseillez-moi ». Si le
client répond « pas d'idée », « je ne sais pas », « conseillez-moi »..., l'assistant propose 3 directions
adaptées au contexte au lieu de reposer la même question :

> Pas de souci, je vais vous aider ! Pour un projet sportif comme le vôtre, ces directions fonctionnent souvent bien :
> - Fraîche et aérienne
> - Agrumes énergisants
> - Marine et vivifiante
>
> Laquelle vous tente le plus ? Vous pouvez aussi en combiner deux.

**Filets de sécurité du code sur la fiche** : produits courants (eau de parfum, bougie, gel douche...) et
pays reconnus directement, petites fautes corrigées sur les odeurs (« lingue » → linge), un nom d'entreprise
n'est jamais pris pour un pays.

« Annuler » au récapitulatif ramène aux propositions. Jamais de prix, de délai ni de quantité.

### Exemple avec contexte en une phrase

| Client | Assistant | `etape` |
|---|---|---|
| *(accueil)* | Bonjour et bienvenue chez Givaudan ! Pour commencer, présentez-moi en une phrase votre entreprise et le projet sur lequel vous travaillez. | |
| Je suis JB Industry, je veux créer une brume pour les sportifs pour les futurs Jeux olympiques en partenariat avec Nike. | Merci pour ces précisions, c'est un beau projet ! Quelle odeur aimeriez-vous (linge propre, agrumes, fleurs, boisé…) et dans quel pays votre produit sera-t-il vendu ? | `decouverte` |
| Une brume fraîche et marine, vendue en France | Deux ingrédients de notre catalogue évoquent une odeur fraîche et marine :<br>- Ultrazur™ — FR027 : note marine et aérienne, avec une touche d'agrumes. Très tenace.<br>- Adoxal — FR026 : …<br>Leur intégration dans votre formule reste à valider avec nos parfumeurs. Souhaitez-vous recevoir un échantillon ou être recontacté par un expert ? | `propositions` |
| Je préfère parler à un expert | Récapitulatif : Entreprise JB Industry ; Contexte Création d'une brume pour les Jeux olympiques, en partenariat avec Nike ; Votre projet Brume, France ; Odeur fraîche et marine ; Cible sportifs ; Ingrédients… ; Contact… Confirmez-vous que nous transmettions ce projet à un expert Givaudan ? | `recapitulatif` |

### Exemple complet (lessive, client qui donne son entreprise)

| Client | Assistant | `etape` |
|---|---|---|
| Bonjour, je développe une lessive liquide écologique pour les familles. Je cherche une odeur fraîche. | Pour le suivi de votre projet, pour quelle entreprise travaillez-vous ? | `decouverte` |
| Je travaille chez Maison Lumen, on lance une nouvelle gamme de lessives en supermarché | Quel type de fraîcheur imaginez-vous (linge propre, agrumes, marine…) et dans quel pays la lessive sera-t-elle vendue ? | `decouverte` |
| Plutôt linge propre, pour la France. | Deux ingrédients de notre catalogue évoquent une odeur de linge propre :<br>- Peonile™ — FR004 : note florale et fraîche, avec une touche de géranium et de pamplemousse. Très tenace sur le linge.<br>- Adoxal — FR026 : note marine et propre. Facilement biodégradable.<br>Les deux sont recommandés pour la lessive liquide. Leur intégration dans votre formule reste à valider avec nos parfumeurs. Souhaitez-vous recevoir un échantillon ou être recontacté par un expert ? | `propositions` |
| Je voudrais essayer Adoxal | Voici le récapitulatif de votre demande d'échantillon : Ingrédient FR026 — Adoxal ; Votre projet Lessive liquide, écologique, France ; Demande Un échantillon ; Adresse 12 rue des Lilas, 75011 Paris, France. Confirmez-vous cette demande et cette adresse ? | `recapitulatif` |
| Oui, je confirme | Votre demande ECH-DEMO-001 est enregistrée. Vous pouvez suivre son statut dans « Mes échantillons ». | `terminee` |

Requête :
```json
{ "session_id": "c0ff749a-…", "question": "Je voudrais essayer Adoxal" }
```
Réponse (raccourcie) :
```json
{
  "session_id": "c0ff749a-…",
  "reponse": "Voici le récapitulatif de votre demande d'échantillon :\n- Ingrédient : FR026 — Adoxal\n- …\n\nConfirmez-vous cette demande et cette adresse ?",
  "etape": "recapitulatif",
  "brief": { "produit": "lessive liquide", "odeur": "linge propre", "odeur_precise": true, "cible": "familles", "contraintes": "écologique", "pays": "France" },
  "ingredients_proposes": [],
  "recapitulatif": {
    "type": "echantillon",
    "lignes": [
      { "information": "Ingrédient", "detail": "FR026 — Adoxal" },
      { "information": "Votre projet", "detail": "Lessive liquide, écologique, France" },
      { "information": "Demande", "detail": "Un échantillon" },
      { "information": "Adresse", "detail": "12 rue des Lilas, 75011 Paris, France" }
    ]
  },
  "suggestions": ["Oui, je confirme", "Annuler"],
  "numero_demande": null
}
```

## 7. Autres règles appliquées dans le code
- **Usage** : si le client cite son produit (lessive liquide, linge, shampoing, savon, déodorant,
  nettoyant, javel...), les ingrédients recommandés pour ce produit sont favorisés et ceux déconseillés
  exclus. Pour un produit inconnu du catalogue (bougie, gel douche...), seule la proximité de sens compte.
- **Jargon** : les mots techniques du catalogue (aldéhydé, lactonique, ozonique...) sont traduits en mots
  simples (propre, crémeux, aérien...).
- **Demandes** : `data/demandes.json` est créé au premier enregistrement (écriture sûre). Pour repartir de
  zéro, supprimer ce fichier.

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

Les identifiants `FR001` à `FR030` sont internes à la démonstration : ce ne sont pas les codes
commerciaux Givaudan. Le champ `tenue` correspond à la tenue sur mouillette, pas à celle du produit
fini. Une application indiquée dans le CSV ne remplace pas une validation réglementaire ou des essais.
Une case vide représente une information inconnue ; les origines et les données de durabilité
restent notamment incomplètes. Les liens sources permettent de consulter les fiches d'origine.

## Évaluer les réponses avec Ollama

Après le démarrage de l'API, exécuter dans un autre terminal :

```powershell
.\.venv\Scripts\python.exe scripts\evaluer_rag.py --url http://127.0.0.1:8000
```

Le script envoie dix scénarios et enregistre les questions, réponses, durées et contrôles dans
`evaluation/resultats_rag.json` et une version lisible `evaluation/resultats_rag.html`.
Les contrôles automatiques portent sur les références, les sources, les exclusions d'usage,
les prix inventés et certaines demandes de clarification. Ils ne mesurent pas à eux seuls la qualité
métier ; il faut relire les réponses. `--cas savon_boise,reference_inconnue` limite un essai.

Les salutations simples, les références inconnues et les informations commerciales ou de conformité
manquantes reçoivent une réponse fixe, conservée dans la session. Les briefs incomplets suivent
l'étape découverte de la conversation. Les pistes olfactives sont rédigées par le modèle lorsque
le produit et l'ambiance sont connus. Le script de test crée des sessions et réutilise leurs identifiants.

Les contrôles du contrat et de la progression des sessions se lancent avec :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Ils utilisent le catalogue local indexé. Les réponses Ollama sont simulées dans les tests de progression
pour isoler la gestion des étapes ; les essais du script d'évaluation appellent réellement Ollama.

## Parcours échantillons de démonstration

Le catalogue réel contient des **ingrédients**, pas des compositions commandables. Le fichier
`data/compositions_demo.json` ajoute deux **compositions fictives** : `DEMO-BOIS-01` Bois doux
et `DEMO-BOIS-02` Bois frais. Leur compatibilité reste à valider. Elles ne sont pas mélangées aux
références `FRxxx` et n'ont ni prix ni stock réel.

Le parcours est guidé par le code et ne demande pas au LLM de décider d'un enregistrement.
Toutes les routes sont accessibles dans Swagger, sous **Échantillons démo** et **Conversation**.

1. `POST /sessions` avec `{"parcours":"echantillons_demo"}`. Copier le `session_id`.
2. `POST /ask` avec le même `session_id` et le message
   « Bonjour, je développe une gamme de savons. Je cherche une senteur boisée et discrète. ».
3. Envoyer « Des savons solides, pour le marché français. ». Les deux compositions sont proposées.
4. Envoyer « Je voudrais essayer Bois doux. ». L'assistant demande une adresse.
5. `POST /sessions/{session_id}/adresse` avec une **adresse fictive**, par exemple :

```json
{
  "entreprise": "Entreprise Démo",
  "ligne1": "10 rue Exemple",
  "code_postal": "75000",
  "ville": "Paris",
  "pays": "France"
}
```

6. Lire le récapitulatif, puis envoyer « Oui, je confirme. » via `POST /ask` avec le même identifiant.
   L'API renvoie un numéro `ECH-DEMO-...` seulement après l'enregistrement SQLite réussi.
7. `GET /echantillons?session_id=...` représente **Mes échantillons**.
   `GET /echantillons/{demande_id}?session_id=...` donne le statut de la demande.

Autre possibilité pour Tech : `POST /echantillons/preparer` avec `session_id`, `reference` et
`adresse_livraison`. Ce n'est qu'un récapitulatif. Copier son `jeton_confirmation`, puis appeler
`POST /echantillons/confirmer` avec `session_id`, `jeton_confirmation` et `confirme: true`.
Une nouvelle adresse invalide le jeton précédent. Répéter la même confirmation ne crée pas de doublon.

Ce parcours de compositions n'utilise pas le compte fictif de `/compte` : son adresse est fournie
pour la conversation. Le parcours ingrédients du nouveau chat conserve le compte fictif et les
demandes échantillon ou expert de `demandes.py`. Utiliser exclusivement des données fictives pour ce prototype.
Les demandes sont conservées dans `data/echantillons_demo.sqlite3` (ignoré par Git ; chemin configurable
avec `ECHANTILLONS_DB`). Les sessions restent en mémoire et sont perdues au redémarrage, mais les demandes
enregistrées restent consultables avec l'ancien `session_id`. Cet identifiant sert au filtrage du prototype ;
il ne remplace pas une authentification de compte client.

Aucune expédition ni commande commerciale n'est déclenchée. Le seul statut disponible est `enregistree`.
Le scénario ne démontre pas les performances du LLM ou la compatibilité technique d'une composition.
