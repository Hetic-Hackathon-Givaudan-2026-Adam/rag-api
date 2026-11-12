"""Contrôles du contrat FastAPI avec le vrai catalogue local indexé."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def setUp(self):
        main.SESSIONS.clear()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        stockage = patch.object(main.demandes, 'DEMANDES_PATH', str(Path(tmp.name) / 'demandes.json'))
        stockage.start()
        self.addCleanup(stockage.stop)

    def test_catalogue_et_sante(self):
        health = self.client.get('/health').json()
        self.assertEqual(health['statut'], 'ok', health)
        rows = self.client.get('/ingredients').json()
        self.assertGreater(len(rows), 0)
        self.assertEqual(health['nb_ingredients'], len(rows))
        self.assertEqual(len({r['id'] for r in rows}), len(rows))

    def test_swagger(self):
        self.assertEqual(self.client.get('/docs').status_code, 200)
        schema = self.client.get('/openapi.json').json()
        self.assertIn('/ask', schema['paths'])
        self.assertIn('/ingredients', schema['paths'])
        self.assertIn('/sessions', schema['paths'])

    def test_salutation_et_session(self):
        created = self.client.post('/sessions').json()
        sid = created['session_id']
        self.assertIn('Bonjour', created['message_accueil'])
        response = self.client.post('/ask', json={'question': 'Bonjour !', 'session_id': sid}).json()
        self.assertEqual(response['session_id'], sid)
        self.assertEqual(response['reponse'], 'Bonjour ! Quel est votre besoin ?')
        self.assertEqual(response['ingredients_proposes'], [])
        detail = self.client.get('/sessions/' + sid).json()
        self.assertEqual(detail['messages'][-2]['content'], 'Bonjour !')
        self.assertEqual(self.client.delete('/sessions/' + sid).status_code, 200)
        self.assertEqual(self.client.get('/sessions/' + sid).status_code, 404)

    def test_reference_absente(self):
        self.assertEqual(self.client.get('/ingredients/FR999').status_code, 404)
        response = self.client.post('/ask', json={'question': 'Où est la référence FR999 ?'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['ingredients_proposes'], [])
        self.assertIn('ne trouve pas', response.json()['reponse'])

    def test_questions_vides_refusees(self):
        for question in ['', '   ']:
            with self.subTest(question=repr(question)):
                self.assertEqual(self.client.post('/ask', json={'question': question}).status_code, 422)

    def test_limites_et_roles(self):
        for extra in [{'nb_resultats': 0}, {'nb_resultats': 6}, {'historique': [{'role': 'system', 'content': 'test'}]}]:
            self.assertEqual(self.client.post('/ask', json={'question': 'Un savon boisé', **extra}).status_code, 422)

    def test_clarification_avant_recommandation(self):
        provider_responses = [
            {'message': {'content': json.dumps({c: '' for c in main.rag.CHAMPS_TEXTE})}},
        ]
        with patch('rag.ollama.chat', side_effect=provider_responses):
            response = self.client.post('/ask', json={'question': 'Je veux créer un parfum pour mon produit.'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['ingredients_proposes'], [])
        self.assertEqual(response.json()['etape'], 'decouverte')
        self.assertIn('projet', response.json()['reponse'])

    def test_progression_du_brief_vers_contact_expert(self):
        moteur = main.etat['rag']
        ingredient = next(a for a in moteur.ingredients if a['id'] == 'FR015')
        autre = next(a for a in moteur.ingredients if a['id'] == 'FR014')
        provider_responses = [
            {'message': {'content': json.dumps({'entreprise': 'Maison Démo', 'projet': 'savons artisanaux', 'produit': 'savon', 'pays': 'France'})}},
            {'message': {'content': json.dumps({'odeur': 'boisée', 'odeur_precise': True, 'cible': 'adultes'})}},
            {'message': {'content': json.dumps({'intention': 'expert'})}},
            {'message': {'content': json.dumps({'intention': 'confirmer'})}},
        ]
        sid = self.client.post('/sessions').json()['session_id']
        with patch('rag.ollama.chat', side_effect=provider_responses) as chat, patch.object(moteur, 'rechercher', return_value=[(ingredient, 0.9), (autre, 0.8)]), patch.object(moteur, '_llm', return_value=''):
            responses = [self.client.post('/ask', json={'session_id': sid, 'question': q}).json()
                         for q in ['Maison Démo, projet savons artisanaux pour la France.', 'Une odeur boisée pour adultes.', 'Je veux être recontacté par un expert.', 'Oui, je confirme.']]
        self.assertEqual([r['etape'] for r in responses], ['decouverte', 'propositions', 'recapitulatif', 'terminee'])
        self.assertEqual(responses[-1]['brief']['produit'], 'savon')
        self.assertEqual(responses[-1]['brief']['odeur'], 'boisée')
        self.assertEqual(responses[-1]['brief']['cible'], 'adultes')
        self.assertEqual(responses[1]['ingredients_proposes'][0]['id'], 'FR015')
        self.assertTrue(responses[-1]['numero_demande'].startswith('EXP-DEMO-'))
        self.assertEqual(len(self.client.get('/sessions/' + sid).json()['messages']), 9)
        self.assertTrue(all(call.kwargs.get('think') is False for call in chat.call_args_list))

    def test_chat_et_compte_demo(self):
        page = self.client.get('/chat')
        self.assertEqual(page.status_code, 200)
        self.assertIn('text/html', page.headers['content-type'])
        self.assertIn('email', self.client.get('/compte').json())
        self.assertEqual(self.client.get('/demandes').status_code, 200)
        self.assertEqual(self.client.get('/echantillons').json(), [])

    def test_pas_de_prix_ou_certification_inventes(self):
        for question, attendu in [('Quel est le prix du Javanol ?', 'commercial'),
                                   ('Peux-tu certifier la conformité IFRA de mon savon ?', 'expert')]:
            response = self.client.post('/ask', json={'question': question}).json()
            self.assertEqual(response['ingredients_proposes'], [])
            self.assertIn(attendu, response['reponse'])


if __name__ == '__main__':
    unittest.main()
