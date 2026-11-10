"""Общий режим через HTTP: отдельные формализация и проверка, без оплаты LLM."""

import copy
import json
from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from factlab.web.app import create_app
from factlab.web.llm_client import LLMResponse, LLMSettings


ROOT = Path(__file__).resolve().parents[2] / "src" / "factlab"


def example(name):
    folder = ROOT / 'data' / 'general_examples' / name
    return ((folder / 'text.txt').read_text(encoding='utf-8'),
            json.loads((folder / 'bundle.json').read_text(encoding='utf-8')))


class GeneralAPITests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.source, self.bundle = example('legend')

        async def fake_formalizer(**kwargs):
            self.calls.append(kwargs)
            return LLMResponse(json.dumps(self.bundle, ensure_ascii=False), 'test-model', {
                'prompt_tokens': 100, 'completion_tokens': 200, 'total_tokens': 300,
            })

        self.settings = LLMSettings('test-only-general-secret', 'https://provider.invalid/v1', 'test-model')
        self.client = TestClient(create_app(self.settings, formalizer=fake_formalizer))

    def analyze(self, source, bundle, accepted=()):
        return self.client.post('/api/general/analyze', json={
            'text': source, 'bundle_json': json.dumps(bundle, ensure_ascii=False),
            'accepted_proposal_ids': list(accepted),
        })

    def test_general_page_is_default_and_profiles_are_preserved(self):
        self.assertIn('general.js', self.client.get('/').text)
        self.assertIn('general.js', self.client.get('/general').text)
        self.assertIn('app.js', self.client.get('/profiles').text)
        self.assertEqual(self.client.get('/static/general.js').status_code, 200)

    def test_bootstrap_has_dynamic_examples_without_exposing_key(self):
        response = self.client.get('/api/general/bootstrap')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(self.settings.api_key, response.text)
        data = response.json()
        self.assertEqual({e['id'] for e in data['examples']}, {'discovery', 'legend', 'siblings', 'disjunction'})
        self.assertEqual(data['defaults']['text'], example('discovery')[0])
        self.assertTrue(data['configuration']['has_api_key'])
        self.assertEqual(self.calls, [])

    def test_formalization_makes_one_call_and_does_not_run_solver(self):
        response = self.client.post('/api/general/extract', json={
            'text': self.source, 'model': 'selected-model',
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNone(data['report'])
        self.assertEqual(len(data['preview']['premises']), 8)
        self.assertTrue(all(p['formula_text'] for p in data['preview']['premises']))
        self.assertEqual(data['usage']['total_tokens'], 300)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]['model_override'], 'selected-model')

    def test_default_demo_discovers_positive_and_negative_facts_without_questions(self):
        response = self.client.post('/api/general/demo')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['source_text'], example('discovery')[0])
        self.assertEqual(data['bundle']['queries'], [])
        self.assertEqual(data['report']['queries'], [])
        self.assertEqual(data['report']['consistency'], 'sat')
        found = {
            (f['predicate'], tuple(f['args']), f['negated'])
            for f in data['report']['facts'] if f['origin'] == 'derived'
        }
        self.assertEqual(found, {
            ('mechanism', ('ray',), False),
            ('device', ('ray',), False),
            ('robot', ('nora',), True),
        })
        self.assertEqual(self.calls, [])

    def test_extract_and_discover_do_not_require_questions_or_another_llm_call(self):
        self.source, self.bundle = example('discovery')
        response = self.client.post('/api/general/extract', json={'text': self.source})
        self.assertEqual(response.status_code, 200)
        extracted = response.json()
        self.assertEqual(extracted['bundle']['queries'], [])
        self.assertIsNone(extracted['report'])
        response = self.analyze(self.source, extracted['bundle'])
        self.assertEqual(response.status_code, 200)
        report = response.json()['report']
        self.assertEqual(report['queries'], [])
        self.assertEqual(len([f for f in report['facts'] if f['origin'] == 'derived']), 3)
        self.assertEqual(len(self.calls), 1)

    def test_legend_demo_checks_true_and_false_queries_and_quantifiers(self):
        response = self.client.post('/api/general/demo', json={'example_id': 'legend'})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['demo'])
        report = data['report']
        self.assertEqual(report['consistency'], 'sat')
        self.assertEqual([q['status'] for q in report['queries']], [
            'entailed', 'refuted', 'entailed', 'entailed',
        ])
        derived = [f for f in report['facts'] if f['origin'] == 'derived']
        self.assertEqual(len(derived), 2)
        self.assertTrue(all(f['negated'] and f['support'] for f in derived))
        self.assertEqual(report['statistics']['accepted_proposals'], 0)
        self.assertEqual(len(data['bundle']['entities']), 2)
        self.assertGreaterEqual(len(report['unparsed']), 2)
        self.assertEqual(self.calls, [])

    def test_sibling_definitions_require_explicit_selection(self):
        source, bundle = example('siblings')
        first = self.analyze(source, bundle).json()['report']
        self.assertEqual(first['queries'][0]['status'], 'undetermined')
        selected = [p['id'] for p in bundle['proposals']]
        second = self.analyze(source, bundle, selected).json()['report']
        self.assertEqual(second['queries'][0]['status'], 'entailed')
        self.assertEqual(second['queries'][1]['status'], 'undetermined')
        self.assertEqual(second['queries'][2]['status'], 'undetermined')
        self.assertEqual(second['statistics']['accepted_proposals'], len(selected))
        # Selection is local to one request; it does not alter later analyses.
        third = self.analyze(source, bundle).json()['report']
        self.assertEqual(third['queries'][0]['status'], 'undetermined')
        self.assertEqual(self.calls, [])

    def test_disjunction_is_one_premise_and_does_not_become_two_facts(self):
        response = self.client.post('/api/general/demo', json={'example_id': 'disjunction'})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['bundle']['premises'][0]['formula']['op'], 'or')
        self.assertEqual([q['status'] for q in data['report']['queries']], ['entailed', 'refuted'])
        self.assertEqual(len([f for f in data['report']['facts'] if f['origin'] == 'derived']), 1)

    def test_conflicting_premises_do_not_turn_every_query_into_truth(self):
        bundle = copy.deepcopy(self.bundle)
        first_ground = next(p for p in bundle['premises'] if p['formula']['op'] == 'atom')
        opposite = copy.deepcopy(first_ground)
        opposite['id'] = 'contradiction'
        opposite['formula'] = {'op': 'not', 'arg': opposite['formula']}
        bundle['premises'].append(opposite)
        response = self.analyze(self.source, bundle)
        self.assertEqual(response.status_code, 200)
        report = response.json()['report']
        self.assertEqual(report['consistency'], 'unsat')
        self.assertTrue(report['consistency_support'])
        self.assertEqual(report['facts'], [])
        self.assertTrue(all(q['status'] == 'inconsistent' for q in report['queries']))

    def test_invalid_llm_result_is_kept_for_editing_without_retry(self):
        async def invalid(**kwargs):
            self.calls.append(kwargs)
            return LLMResponse('not json', 'test-model', {'total_tokens': 123})

        client = TestClient(create_app(self.settings, formalizer=invalid))
        response = client.post('/api/general/extract', json={'text': self.source})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()['error']['raw_output'], 'not json')
        self.assertEqual(response.json()['usage']['total_tokens'], 123)
        self.assertEqual(len(self.calls), 1)

    def test_bad_quotes_or_code_operations_are_rejected_before_solver(self):
        for change in ('quote', 'code'):
            with self.subTest(change=change):
                bundle = copy.deepcopy(self.bundle)
                if change == 'quote':
                    bundle['premises'][0]['evidence'] = ['Несуществующая цитата.']
                else:
                    bundle['premises'][0]['formula'] = {'op': 'eval', 'code': 'print(1)'}
                response = self.analyze(self.source, bundle)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()['error']['code'], 'invalid_formalization')
        self.assertEqual(self.calls, [])

    def test_unknown_example_and_proposal_are_not_silently_accepted(self):
        response = self.client.post('/api/general/demo', json={'example_id': '../../.env'})
        self.assertEqual(response.status_code, 404)
        response = self.analyze(self.source, self.bundle, ['missing'])
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()['error']['code'], 'invalid_logic_input')
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
