"""Профили изолируют словари, правила и входные факты; API вызывается подставной."""

import json
import unittest

from fastapi.testclient import TestClient

from factlab.web.app import create_app, load_context
from factlab.web.llm_client import LLMResponse, LLMSettings


class WebProfileTests(unittest.TestCase):
    def setUp(self):
        self.contexts = {name: load_context(name) for name in ('basic', 'university', 'family')}
        self.calls = []

        async def fake_extractor(**kwargs):
            self.calls.append(kwargs)
            names = {p['name'] for p in json.loads(kwargs['schema_json'])['predicates']}
            selected = 'family' if 'parent' in names else 'university' if 'studies' in names else 'basic'
            return LLMResponse(self.contexts[selected].demo_json, 'test-model', {'total_tokens': 100})

        settings = LLMSettings('test-only-key', 'https://provider.invalid/v1', 'test-model')
        self.client = TestClient(create_app(settings, fake_extractor))

    def test_bootstrap_lists_profiles_and_selects_matching_schema(self):
        data = self.client.get('/api/bootstrap').json()
        self.assertEqual(data['profile']['id'], 'university')
        self.assertEqual({p['id'] for p in data['profiles']}, set(self.contexts))
        for name, context in self.contexts.items():
            with self.subTest(profile=name):
                response = self.client.get('/api/bootstrap', params={'profile': name})
                self.assertEqual(response.status_code, 200)
                selected = response.json()
                self.assertEqual(selected['profile']['id'], name)
                self.assertEqual(selected['defaults']['schema_json'], context.schema_json)
                self.assertEqual(selected['defaults']['text'], context.demo_text)
                self.assertEqual(
                    set(selected['profile']['rule_descriptions']), {r.id for r in context.rules},
                )
        self.assertEqual(self.calls, [])

    def test_university_demo_combines_different_kinds_of_rules(self):
        response = self.client.post('/api/demo', json={'profile': 'university'})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        report = data['report']
        derived = {f['display'] for f in report['facts'] if f['origin'] == 'derived'}
        self.assertTrue({
            'teacher(Анна)',
            'teaches(Анна, Логика)',
            'teaches_at(Анна, Логика, Университет)',
            'needs_course(Борис, Теория множеств)',
            'collaborates(Борис, Анна)',
            '¬may_teach(Вера, Логика)',
        } <= derived)
        self.assertEqual(report['statistics']['initial_facts'], 12)
        self.assertEqual(report['statistics']['new_facts'], 20)
        self.assertEqual(report['status'], 'ok')
        graph = {g['predicate']: g for g in report['graph_checks']}
        self.assertEqual(graph['part_of']['comparison'], 'passed')
        self.assertEqual(graph['requires']['comparison'], 'skipped')
        self.assertEqual(data['profile']['id'], 'university')
        self.assertIsNone(data['usage'])
        self.assertEqual(self.calls, [])

    def test_family_demo_uses_other_types_and_preserves_direct_parenthood(self):
        response = self.client.post('/api/demo', json={'profile': 'family'})
        self.assertEqual(response.status_code, 200)
        report = response.json()['report']
        facts = {f['display'] for f in report['facts']}
        self.assertTrue({
            'ancestor(Анна, Глеб)', 'grandparent(Анна, Вера)',
            'child(Борис, Анна)', 'spouse(Дарья, Анна)',
        } <= facts)
        self.assertNotIn('parent(Анна, Вера)', facts)
        self.assertEqual(report['statistics']['new_facts'], 12)
        self.assertEqual(report['status'], 'ok')
        graph = report['graph_checks'][0]
        self.assertTrue(graph['acyclic_required'])
        self.assertFalse(graph['strict_order'])
        self.assertNotIn('prerequisite_order', graph)
        self.assertEqual(self.calls, [])

    def test_get_profile_does_not_change_default_for_other_requests(self):
        self.client.get('/api/bootstrap', params={'profile': 'family'})
        data = self.client.post('/api/demo').json()
        self.assertEqual(data['profile']['id'], 'university')

    def test_each_llm_call_uses_explicit_profile_without_previous_profile_leak(self):
        for name in ('family', 'university', 'basic'):
            with self.subTest(profile=name):
                response = self.client.post('/api/analyze', json={
                    'profile': name, 'text': self.contexts[name].demo_text,
                })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['profile']['id'], name)
                self.assertEqual(self.calls[-1]['schema_json'], self.contexts[name].schema_json)
        self.assertEqual(len(self.calls), 3)

    def test_json_from_another_profile_is_rejected_without_llm(self):
        context = self.contexts['family']
        response = self.client.post('/api/infer', json={
            'profile': 'university', 'text': context.demo_text,
            'extraction_json': context.demo_json,
        })
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()['error']['code'], 'invalid_extraction')
        self.assertEqual(self.calls, [])

    def test_unknown_profile_is_rejected_before_paid_call(self):
        for name in ('unknown', '../../family', ''):
            with self.subTest(profile=name):
                response = self.client.get('/api/bootstrap', params={'profile': name})
                self.assertEqual(response.status_code, 404)
        for route in ('analyze', 'extract', 'infer', 'demo'):
            payload = {'profile': '../../family'}
            if route != 'demo':
                payload['text'] = self.contexts['family'].demo_text
            if route == 'infer':
                payload['extraction_json'] = self.contexts['family'].demo_json
            response = self.client.post('/api/' + route, json=payload)
            self.assertEqual(response.status_code, 404)
        self.assertEqual(self.calls, [])

    def test_cycle_created_by_inverse_input_is_reported_as_acyclic_violation(self):
        context = self.contexts['family']
        data = json.loads(context.demo_json)
        labels = {e['label']: e['id'] for e in data['entities']}
        # child(Анна, Глеб) adds parent(Глеб, Анна), closing the ancestor chain.
        quote = 'Анна — ребёнок Глеба.'
        data['facts'].append({
            'predicate': 'child', 'args': [labels['Анна'], labels['Глеб']],
            'negated': False, 'evidence': [quote],
        })
        response = self.client.post('/api/infer', json={
            'profile': 'family', 'text': context.demo_text + '\n' + quote,
            'extraction_json': json.dumps(data, ensure_ascii=False),
        })
        self.assertEqual(response.status_code, 200)
        report = response.json()['report']
        self.assertEqual(report['status'], 'needs_review')
        graph = report['graph_checks'][0]
        self.assertFalse(graph['strict_order'])
        self.assertTrue(graph['acyclic_required'])
        self.assertEqual(len(graph['acyclic_cycle_vertices']), 4)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
