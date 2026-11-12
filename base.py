"""
Base de données SQLite du prototype (un seul fichier : data/givaudan.db, créé automatiquement).

Tables :
- sessions  : une conversation (étape, fiche brief, ingrédients proposés...) ;
- messages  : l'historique de chaque conversation ;
- demandes  : demandes d'échantillon / de contact expert (ECH-DEMO-001, EXP-DEMO-001...) ;
- comptes   : compte client (un compte de démo, repris de data/compte_demo.json).

Le catalogue des ingrédients reste dans data/catalogue_fragrances.csv (mis à jour par l'équipe Data).
"""
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime

DOSSIER = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.getenv("BASE_SQLITE", os.path.join(DOSSIER, "data", "givaudan.db"))
COMPTE_DEMO_PATH = os.path.join(DOSSIER, "data", "compte_demo.json")
ANCIEN_DEMANDES_PATH = os.path.join(DOSSIER, "data", "demandes.json")

_verrou = threading.Lock()  # une écriture à la fois (l'API traite plusieurs requêtes en parallèle)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    etat        TEXT NOT NULL,          -- JSON : étape, brief, ingrédients proposés, récapitulatif en attente...
    cree_le     TEXT NOT NULL,
    modifie_le  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    contenu     TEXT NOT NULL,
    cree_le     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS demandes (
    numero      TEXT PRIMARY KEY,       -- ECH-DEMO-001, EXP-DEMO-001...
    type        TEXT NOT NULL CHECK (type IN ('echantillon', 'expert')),
    statut      TEXT NOT NULL,
    date        TEXT NOT NULL,
    session_id  TEXT,
    donnees     TEXT NOT NULL           -- JSON complet de la demande (compte, brief, ingrédient(s))
);
CREATE TABLE IF NOT EXISTS comptes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entreprise  TEXT NOT NULL,
    contact     TEXT NOT NULL,
    email       TEXT NOT NULL,
    adresse     TEXT NOT NULL
);
"""


def maintenant():
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connexion():
    con = sqlite3.connect(BASE_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def initialiser():
    """Crée les tables si besoin, ajoute le compte de démo et reprend l'ancien data/demandes.json."""
    os.makedirs(os.path.dirname(BASE_PATH), exist_ok=True)
    with _verrou, connexion() as con:
        con.execute("PRAGMA journal_mode = WAL")  # lectures et écritures en même temps sans blocage
        con.executescript(SCHEMA)
        if con.execute("SELECT COUNT(*) FROM comptes").fetchone()[0] == 0 and os.path.exists(COMPTE_DEMO_PATH):
            with open(COMPTE_DEMO_PATH, encoding="utf-8") as f:
                c = json.load(f)
            con.execute("INSERT INTO comptes (entreprise, contact, email, adresse) VALUES (?, ?, ?, ?)",
                        (c["entreprise"], c["contact"], c["email"], c["adresse"]))
        if con.execute("SELECT COUNT(*) FROM demandes").fetchone()[0] == 0 and os.path.exists(ANCIEN_DEMANDES_PATH):
            try:
                with open(ANCIEN_DEMANDES_PATH, encoding="utf-8") as f:
                    anciennes = json.load(f)
            except (OSError, json.JSONDecodeError):
                anciennes = []
            for d in anciennes if isinstance(anciennes, list) else []:
                con.execute("INSERT OR IGNORE INTO demandes VALUES (?, ?, ?, ?, ?, ?)",
                            (d["numero"], d["type"], d.get("statut", "Enregistrée"), d.get("date", maintenant()),
                             d.get("session_id"), json.dumps(d, ensure_ascii=False)))


# ---------- Sessions et messages ----------

def _etat(session):
    """Ce qui est stocké dans sessions.etat : tout sauf les messages (table à part) et la date."""
    return {k: v for k, v in session.items() if k not in ("messages", "cree_le")}


def enregistrer_session(session_id, session):
    """Crée ou met à jour la session, et ajoute les nouveaux messages de l'historique."""
    cree_le = session["cree_le"]
    cree_le = cree_le.isoformat(timespec="seconds") if isinstance(cree_le, datetime) else str(cree_le)
    with _verrou, connexion() as con:
        con.execute(
            "INSERT INTO sessions (id, etat, cree_le, modifie_le) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET etat = excluded.etat, modifie_le = excluded.modifie_le",
            (session_id, json.dumps(_etat(session), ensure_ascii=False), cree_le, maintenant()),
        )
        deja = con.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
        for m in session["messages"][deja:]:
            con.execute("INSERT INTO messages (session_id, role, contenu, cree_le) VALUES (?, ?, ?, ?)",
                        (session_id, m["role"], m["content"], maintenant()))


def charger_session(session_id):
    """Session complète (avec son historique), ou None si elle n'existe pas."""
    if not session_id:
        return None
    with connexion() as con:
        ligne = con.execute("SELECT etat, cree_le FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if ligne is None:
            return None
        messages = con.execute("SELECT role, contenu FROM messages WHERE session_id = ? ORDER BY id",
                               (session_id,)).fetchall()
    session = json.loads(ligne["etat"])
    session["messages"] = [{"role": m["role"], "content": m["contenu"]} for m in messages]
    session["cree_le"] = datetime.fromisoformat(ligne["cree_le"])
    return session


def supprimer_session(session_id):
    with _verrou, connexion() as con:
        return con.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount > 0


def lister_sessions(limite=50):
    """Conversations les plus récentes, avec un résumé (pour la page « Mes projets »)."""
    with connexion() as con:
        lignes = con.execute(
            "SELECT s.id, s.etat, s.cree_le, s.modifie_le, "
            "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id AND m.role = 'user') AS nb_messages "
            "FROM sessions s ORDER BY s.modifie_le DESC LIMIT ?", (limite,)).fetchall()
    return [{"id": l["id"], "etat": json.loads(l["etat"]), "cree_le": l["cree_le"],
             "modifie_le": l["modifie_le"], "nb_messages": l["nb_messages"]} for l in lignes]


# ---------- Demandes ----------

def lister_demandes(type_=None):
    with connexion() as con:
        if type_:
            lignes = con.execute("SELECT donnees FROM demandes WHERE type = ? ORDER BY date, numero", (type_,))
        else:
            lignes = con.execute("SELECT donnees FROM demandes ORDER BY date, numero")
        return [json.loads(l["donnees"]) for l in lignes.fetchall()]


def ajouter_demande(fabriquer, type_, prefixe):
    """Numérote et enregistre une demande en une seule opération (pas de doublon de numéro).
    `fabriquer(numero)` renvoie le dictionnaire complet de la demande."""
    with _verrou, connexion() as con:
        rang = con.execute("SELECT COUNT(*) FROM demandes WHERE type = ?", (type_,)).fetchone()[0] + 1
        numero = f"{prefixe}-{rang:03d}"
        while con.execute("SELECT 1 FROM demandes WHERE numero = ?", (numero,)).fetchone():
            rang += 1
            numero = f"{prefixe}-{rang:03d}"
        demande = fabriquer(numero)
        con.execute("INSERT INTO demandes VALUES (?, ?, ?, ?, ?, ?)",
                    (numero, type_, demande["statut"], demande["date"], demande.get("session_id"),
                     json.dumps(demande, ensure_ascii=False)))
    return numero


# ---------- Compte ----------

def compte():
    """Compte client de la démo (le premier de la table)."""
    with connexion() as con:
        ligne = con.execute("SELECT entreprise, contact, email, adresse FROM comptes ORDER BY id LIMIT 1").fetchone()
    return dict(ligne) if ligne else {"entreprise": "", "contact": "", "email": "", "adresse": ""}
