"""Compositions fictives et demandes d'échantillons de démonstration, sans expédition."""
import json
import os
import re
import sqlite3
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool

ROOT = Path(__file__).parent
DB_PATH = Path(os.getenv('ECHANTILLONS_DB', str(ROOT / 'data' / 'echantillons_demo.sqlite3')))


class AdresseLivraison(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid', json_schema_extra={'examples': [{
        'entreprise': 'Entreprise Démo', 'ligne1': '10 rue Exemple', 'code_postal': '75000',
        'ville': 'Paris', 'pays': 'France'
    }]})
    entreprise: str = Field(min_length=1, max_length=160)
    ligne1: str = Field(min_length=1, max_length=240)
    complement: str = Field(default='', max_length=240)
    code_postal: str = Field(min_length=1, max_length=20)
    ville: str = Field(min_length=1, max_length=120)
    pays: str = Field(min_length=1, max_length=100)


class CompositionDemo(BaseModel):
    reference: str
    nom: str
    description: str
    direction_olfactive: str
    application_demo: str
    compatibilite: Literal['a_valider'] = 'a_valider'
    demonstration: Literal[True] = True


class PropositionEchantillon(BaseModel):
    jeton_confirmation: str
    reference: str
    nom: str
    adresse_livraison: AdresseLivraison
    statut: Literal['a_confirmer'] = 'a_confirmer'
    demonstration: Literal[True] = True


class DemandeEnregistree(BaseModel):
    demande_id: str
    session_id: str
    reference: str
    nom: str
    adresse_livraison: AdresseLivraison
    brief: dict[str, str | bool]
    statut: Literal['enregistree'] = 'enregistree'
    cree_le: str
    demonstration: Literal[True] = True


class Preparation(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    session_id: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    adresse_livraison: Optional[AdresseLivraison] = None


class Confirmation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(min_length=1)
    jeton_confirmation: str = Field(min_length=1)
    confirme: StrictBool


class AdresseSession(BaseModel):
    adresse_livraison: AdresseLivraison
    confirmation_echantillon: Optional[PropositionEchantillon] = None
    reponse: str


def catalogue_demo():
    rows = json.loads((ROOT / 'data' / 'compositions_demo.json').read_text(encoding='utf-8'))
    return [CompositionDemo.model_validate(r).model_dump() for r in rows]


def composition(reference):
    for row in catalogue_demo():
        if row['reference'] == reference.upper():
            return row
    raise HTTPException(404, 'Composition de démonstration introuvable. Une référence FR est un ingrédient, pas une composition commandable.')


def adresse_texte(adresse):
    return ', '.join(v for v in [adresse['entreprise'], adresse['ligne1'], adresse.get('complement'),
                                adresse['code_postal'] + ' ' + adresse['ville'], adresse['pays']] if v)


def resume(proposition):
    return (
        f"Vous souhaitez demander un échantillon de {proposition['reference']} — {proposition['nom']}.\n"
        f"Adresse : {adresse_texte(proposition['adresse_livraison'])}.\n\n"
        "Il s'agit d'une demande de démonstration, sans expédition réelle. "
        "Confirmez-vous cette demande et cette adresse, ou souhaitez-vous modifier l'adresse ?"
    )


def preparer(session, reference, adresse=None):
    fiche = composition(reference)
    destination = adresse or session.get('adresse_livraison')
    if not destination:
        raise HTTPException(422, 'Une adresse de livraison doit être renseignée avant la confirmation.')
    destination = AdresseLivraison.model_validate(destination).model_dump()
    session['adresse_livraison'] = destination
    session['composition_selectionnee'] = fiche['reference']
    proposition = PropositionEchantillon(jeton_confirmation=str(uuid.uuid4()), reference=fiche['reference'],
                                       nom=fiche['nom'], adresse_livraison=destination).model_dump()
    session['confirmation_echantillon'] = proposition
    session['demande_echantillon'] = None
    return proposition


def enregistrer_adresse(session, adresse):
    destination = AdresseLivraison.model_validate(adresse).model_dump()
    session['adresse_livraison'] = destination
    if session.get('composition_selectionnee'):
        return preparer(session, session['composition_selectionnee'], destination)
    return None


def _connexion():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute('''CREATE TABLE IF NOT EXISTS echantillons (
        id INTEGER PRIMARY KEY AUTOINCREMENT, jeton TEXT NOT NULL UNIQUE,
        session_id TEXT NOT NULL, reference TEXT NOT NULL, nom TEXT NOT NULL,
        adresse_json TEXT NOT NULL, brief_json TEXT NOT NULL, cree_le TEXT NOT NULL)''')
    return conn


def _demande(row):
    return DemandeEnregistree(
        demande_id=f"ECH-DEMO-{row['id']:03d}", session_id=row['session_id'],
        reference=row['reference'], nom=row['nom'],
        adresse_livraison=json.loads(row['adresse_json']), brief=json.loads(row['brief_json']),
        cree_le=row['cree_le'],
    ).model_dump()


def confirmer(session, session_id, jeton, confirme):
    if confirme is not True:
        raise HTTPException(422, 'La confirmation explicite de la demande et de l’adresse est requise.')
    proposition = session.get('confirmation_echantillon')
    if not proposition or proposition['jeton_confirmation'] != jeton:
        raise HTTPException(409, 'Confirmation absente ou périmée. Vérifiez le nouveau récapitulatif avant de confirmer.')
    try:
        conn = _connexion()
        try:
            with conn:
                conn.execute('BEGIN IMMEDIATE')
                row = conn.execute('SELECT * FROM echantillons WHERE jeton=? AND session_id=?', (jeton, session_id)).fetchone()
                if row is None:
                    conn.execute('INSERT INTO echantillons (jeton,session_id,reference,nom,adresse_json,brief_json,cree_le) VALUES (?,?,?,?,?,?,?)',
                                 (jeton, session_id, proposition['reference'], proposition['nom'],
                                  json.dumps(proposition['adresse_livraison'], ensure_ascii=False),
                                  json.dumps(session['brief'], ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
                    row = conn.execute('SELECT * FROM echantillons WHERE jeton=?', (jeton,)).fetchone()
                demande = _demande(row)
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(503, "L'enregistrement n'a pas pu être confirmé. Vous pouvez réessayer.") from exc
    session['demande_echantillon'] = demande
    return demande


def lister(session_id):
    if not DB_PATH.exists():
        return []
    try:
        conn = _connexion()
        try:
            return [_demande(r) for r in conn.execute('SELECT * FROM echantillons WHERE session_id=? ORDER BY id DESC', (session_id,))]
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(503, 'Le suivi des demandes est temporairement indisponible.') from exc


def _normaliser(texte):
    return ''.join(c for c in unicodedata.normalize('NFD', texte.lower()) if unicodedata.category(c) != 'Mn')


def message_demo(session, question, session_id, adresse=None):
    """Parcours guidé de démonstration ; les confirmations ne sont jamais décidées par le LLM."""
    q = _normaliser(question)
    simple = ' '.join(re.sub(r'[^a-z\s]', ' ', q).split())
    if adresse:
        enregistrer_adresse(session, adresse)
    brief = session['brief']
    if re.search(r'\bsavons?\b', q):
        brief['produit'] = 'savon'
    autre_produit = re.search(r'\b(lessive|shampoing|shampooing|bougie|boisson)\b', q)
    if autre_produit and not re.search(r'\bsavons?\b', q):
        brief['produit'] = autre_produit.group(1)
        session['confirmation_echantillon'] = None
        session['composition_selectionnee'] = None
        session['compositions_proposees'] = []
        session['demande_echantillon'] = None
    if brief['produit'].startswith('savon'):
        if re.search(r'\bsolides?\b', q):
            brief['produit'] = 'savon solide'
        elif re.search(r'\bliquides?\b', q):
            brief['produit'] = 'savon liquide'
    if re.search(r'\bboise(?:e|es|s)?\b', q):
        brief['odeur'] = 'boisé' + (' et discret' if 'discret' in q else '')
        brief['odeur_precise'] = True
    if re.search(r'\b(france|francais|francaise)\b', q):
        brief['pays'] = 'France'

    def retour(texte, etape, compositions=()):
        session['messages'].extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': texte}])
        session['etape'] = etape
        return {'reponse': texte, 'etape': etape, 'compositions_proposees': list(compositions),
                'confirmation_echantillon': session.get('confirmation_echantillon'),
                'demande_echantillon': session.get('demande_echantillon')}

    if simple in {'bonjour', 'salut', 'bonsoir', 'hello'}:
        return retour('Bonjour ! Quel est votre besoin ?', session['etape'])
    if session.get('confirmation_echantillon') and simple in {'oui', 'je confirme', 'oui je confirme', 'confirme'}:
        demande = confirmer(session, session_id, session['confirmation_echantillon']['jeton_confirmation'], True)
        return retour(f"Votre demande {demande['demande_id']} est enregistrée pour la démonstration. "
                      'Vous pouvez suivre son statut dans « Mes échantillons ». Aucune expédition réelle n’est déclenchée.', 'enregistree')
    if session.get('confirmation_echantillon') and simple in {'non', 'annuler', 'j annule'}:
        session['confirmation_echantillon'] = None
        return retour("La préparation est annulée. Aucune nouvelle demande n'a été enregistrée. Quelle composition souhaitez-vous explorer ?", 'selection')

    catalogue = catalogue_demo()
    selection = [f for f in catalogue if f['reference'].lower() in q or _normaliser(f['nom']) in q]
    references = re.findall(r'\b(?:DEMO-[A-Z]+-\d+|FR\d{3})\b', question.upper())
    if references and any(ref not in {f['reference'] for f in catalogue} for ref in references):
        return retour('Cette référence ne correspond pas à une composition de notre catalogue de démonstration. Choisissez Bois doux ou Bois frais.', 'selection', catalogue)
    if len(selection) > 1:
        session['confirmation_echantillon'] = None
        session['composition_selectionnee'] = None
        return retour('Choisissez une seule composition pour cette demande : Bois doux ou Bois frais.', 'selection', catalogue)
    if len(selection) == 1 and not re.search(r'\b(pas|ni|sans|eviter)\b', q):
        if (selection[0]['reference'] not in session.get('compositions_proposees', [])
                or brief['produit'] != 'savon solide' or not brief['pays']):
            return retour("Précisons d'abord le projet : s'agit-il de savons solides ou liquides, et dans quel pays seront-ils commercialisés ?", 'decouverte')
        session['composition_selectionnee'] = selection[0]['reference']
        session['demande_echantillon'] = None
        if session.get('adresse_livraison'):
            proposition = preparer(session, selection[0]['reference'])
            return retour(resume(proposition), 'confirmation')
        return retour(f"Vous avez choisi {selection[0]['nom']} — {selection[0]['reference']}. "
                      "À quelle adresse d'entreprise souhaitez-vous recevoir l'échantillon ? Pour cet essai, utilisez une adresse fictive.", 'adresse')
    if len(selection) == 1 and re.search(r'\b(pas|ni|sans|eviter)\b', q):
        session['composition_selectionnee'] = None
        session['confirmation_echantillon'] = None
        return retour('Quelle autre composition souhaitez-vous explorer ?', 'selection', catalogue)
    if session.get('confirmation_echantillon') and not session.get('demande_echantillon'):
        return retour(resume(session['confirmation_echantillon']), 'confirmation')
    if session.get('demande_echantillon'):
        d = session['demande_echantillon']
        return retour(f"La demande {d['demande_id']} est enregistrée pour la démonstration. Son statut est « enregistrée ». Aucune expédition réelle n'est prévue.", 'enregistree')
    if session.get('composition_selectionnee') and not session.get('adresse_livraison'):
        return retour("Il manque l'adresse de livraison. Renseignez une adresse fictive pour voir le récapitulatif avant de confirmer.", 'adresse')
    if not brief['produit'].startswith('savon'):
        return retour('Pour quel produit recherchez-vous une fragrance ? Cette démonstration propose deux compositions boisées pour des savons solides.', 'decouverte')
    if brief['produit'] == 'savon' or not brief['pays']:
        questions = []
        if brief['produit'] == 'savon':
            questions.append('S’agit-il de savons solides ou liquides ?')
        if not brief['pays']:
            questions.append('Dans quel pays seront-ils commercialisés ?')
        return retour(' '.join(questions), 'decouverte')
    if brief['produit'] != 'savon solide':
        return retour('Ce catalogue de démonstration contient uniquement des pistes pour un savon solide. Une autre application nécessite une sélection et des essais adaptés.', 'decouverte')
    if 'boise' not in _normaliser(brief['odeur']):
        return retour('Souhaitez-vous explorer une direction boisée ? Les deux compositions de ce catalogue de démonstration sont boisées.', 'decouverte')
    session['compositions_proposees'] = [f['reference'] for f in catalogue]
    return retour('Deux compositions fictives de notre catalogue de démonstration pourraient correspondre à la direction recherchée :\n\n'
                  + '\n'.join(f"• {f['nom']} — {f['reference']} : {f['description']}" for f in catalogue)
                  + '\n\nLeur compatibilité avec votre savon et l’intensité obtenue restent à valider par des essais. Laquelle souhaitez-vous essayer ?', 'selection', catalogue)
