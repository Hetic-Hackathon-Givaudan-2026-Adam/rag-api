"""Parcours réel de démonstration avec stockage SQLite temporaire, sans appel LLM."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import echantillons
import main

ADRESSE = {'entreprise': 'Entreprise Démo', 'ligne1': '10 rue Exemple',
           'code_postal': '75000', 'ville': 'Paris', 'pays': 'France'}


class EchantillonsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def setUp(self):
        # Base SQLite temporaire : les tests ne touchent pas data/givaudan.db
        self._tmp_base = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_base.cleanup)
        base_test = patch.object(main.base, 'BASE_PATH', str(Path(self._tmp_base.name) / 'test.db'))
        base_test.start()
        self.addCleanup(base_test.stop)
        main.base.initialiser()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = patch.object(echantillons, 'DB_PATH', Path(self.tmp.name) / 'demandes.sqlite3')
        self.db.start()
        self.addCleanup(self.db.stop)
        self.sid = self.client.post('/sessions', json={'parcours': 'echantillons_demo'}).json()['session_id']

    def ask(self, question, **extra):
        response = self.client.post('/ask', json={'session_id': self.sid, 'question': question, **extra})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def prepare(self, **extra):
        return self.client.post('/echantillons/preparer', json={
            'session_id': self.sid, 'reference': 'DEMO-BOIS-01', 'adresse_livraison': ADRESSE, **extra})

    def confirm(self, jeton, **extra):
        return self.client.post('/echantillons/confirmer', json={
            'session_id': self.sid, 'jeton_confirmation': jeton, 'confirme': True, **extra})

    def test_scenario_complet_et_suivi(self):
        with patch('rag.ollama.chat', side_effect=AssertionError('Le parcours demo ne doit pas inventer via le LLM')):
            first = self.ask('Bonjour, je développe une gamme de savons. Je cherche une senteur boisée et discrète.')
            self.assertEqual(first['etape'], 'decouverte')
            self.assertEqual(first['compositions_proposees'], [])
            second = self.ask('Des savons solides, pour le marché français.')
            self.assertEqual(second['etape'], 'selection')
            self.assertEqual(len(second['compositions_proposees']), 2)
            self.assertEqual(second['ingredients_proposes'], [])
            third = self.ask('Je voudrais essayer Bois doux.')
            self.assertEqual(third['etape'], 'adresse')
            self.assertEqual(self.client.get('/echantillons', params={'session_id': self.sid}).json(), [])
            recap = self.client.post(f'/sessions/{self.sid}/adresse', json=ADRESSE).json()
            self.assertEqual(recap['confirmation_echantillon']['reference'], 'DEMO-BOIS-01')
            self.assertIn('10 rue Exemple', recap['reponse'])
            final = self.ask('Oui, je confirme.')
            self.assertEqual(final['etape'], 'enregistree')
            sample = final['demande_echantillon']
            self.assertTrue(sample['demande_id'].startswith('ECH-DEMO-'))
            self.assertEqual(sample['adresse_livraison']['entreprise'], 'Entreprise Démo')
            suivi = self.client.get('/echantillons/' + sample['demande_id'], params={'session_id': self.sid})
            self.assertEqual(suivi.json()['statut'], 'enregistree')
            self.assertEqual(len(self.client.get('/echantillons', params={'session_id': self.sid}).json()), 1)

    def test_confirmation_obligatoire_et_pas_de_doublon(self):
        proposal = self.prepare().json()
        self.assertEqual(self.client.get('/echantillons', params={'session_id': self.sid}).json(), [])
        self.assertEqual(self.confirm(proposal['jeton_confirmation'], confirme=False).status_code, 422)
        first = self.confirm(proposal['jeton_confirmation']).json()
        again = self.confirm(proposal['jeton_confirmation']).json()
        self.assertEqual(first['demande_id'], again['demande_id'])
        self.assertEqual(len(echantillons.lister(self.sid)), 1)

    def test_changement_adresse_invalide_ancienne_confirmation(self):
        old = self.prepare().json()['jeton_confirmation']
        update = self.client.post(f'/sessions/{self.sid}/adresse', json={**ADRESSE, 'ligne1': '20 rue Exemple'}).json()
        new = update['confirmation_echantillon']['jeton_confirmation']
        self.assertNotEqual(old, new)
        self.assertEqual(self.confirm(old).status_code, 409)
        final = self.confirm(new).json()
        self.assertEqual(final['adresse_livraison']['ligne1'], '20 rue Exemple')

    def test_adresse_et_reference_valides_requises(self):
        self.assertEqual(self.client.post('/echantillons/preparer', json={'session_id': self.sid, 'reference': 'DEMO-BOIS-01'}).status_code, 422)
        self.assertEqual(self.prepare(reference='FR014').status_code, 404)
        self.assertEqual(self.prepare(reference='DEMO-BOIS-99').status_code, 404)
        self.assertEqual(self.client.post(f'/sessions/{self.sid}/adresse', json={**ADRESSE, 'ville': '   '}).status_code, 422)

    def test_echec_stockage_ne_produit_pas_de_confirmation_reussie(self):
        proposal = self.prepare().json()
        with patch('echantillons._connexion', side_effect=sqlite3.OperationalError('indisponible')):
            response = self.confirm(proposal['jeton_confirmation'])
        self.assertEqual(response.status_code, 503)
        self.assertIsNone(main.base.charger_session(self.sid)['demande_echantillon'])
        self.assertEqual(echantillons.lister(self.sid), [])

    def test_sessions_distinctes_et_persistance_sqlite(self):
        proposal = self.prepare().json()
        demande = self.confirm(proposal['jeton_confirmation']).json()
        other_sid = self.client.post('/sessions', json={'parcours': 'echantillons_demo'}).json()['session_id']
        self.assertEqual(self.client.get('/echantillons', params={'session_id': other_sid}).json(), [])
        self.assertEqual(self.client.get('/echantillons/' + demande['demande_id'], params={'session_id': other_sid}).status_code, 404)
        self.client.delete('/sessions/' + self.sid)
        self.assertEqual(echantillons.lister(self.sid)[0]['demande_id'], demande['demande_id'])

    def test_pas_de_confirmation_sans_adresse_dans_le_chat(self):
        self.ask('Je développe un savon solide boisé pour la France.')
        self.ask('Je voudrais essayer Bois doux.')
        response = self.ask('Oui, je confirme.')
        self.assertEqual(response['etape'], 'adresse')
        self.assertIsNone(response['demande_echantillon'])

    def test_changement_produit_invalide_confirmation(self):
        self.ask('Un savon solide boisé pour la France.')
        self.ask('Je voudrais essayer Bois doux.', adresse_livraison=ADRESSE)
        token = main.base.charger_session(self.sid)['confirmation_echantillon']['jeton_confirmation']
        self.ask('Finalement, ce sera une lessive.')
        self.assertEqual(self.confirm(token).status_code, 409)
        self.assertEqual(echantillons.lister(self.sid), [])

    def test_plusieurs_choix_ne_confirment_pas_ancienne_selection(self):
        token = self.prepare().json()['jeton_confirmation']
        response = self.ask('Je voudrais Bois doux et Bois frais.')
        self.assertEqual(response['etape'], 'selection')
        self.assertEqual(self.confirm(token).status_code, 409)
        self.assertEqual(echantillons.lister(self.sid), [])

    def test_endpoints_documentes_et_donnees_fictives(self):
        schema = self.client.get('/openapi.json').json()
        for path in ['/compositions-demo', '/sessions/{session_id}/adresse', '/echantillons/preparer', '/echantillons/confirmer', '/echantillons', '/echantillons/{demande_id}']:
            self.assertIn(path, schema['paths'])
        rows = self.client.get('/compositions-demo').json()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['demonstration'] is True and r['compatibilite'] == 'a_valider' for r in rows))


if __name__ == '__main__':
    unittest.main()
