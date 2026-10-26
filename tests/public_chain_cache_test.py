import ast
import copy
import json
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace as NS

ROOT = pathlib.Path(__file__).resolve().parent
Q = pathlib.Path(__file__).resolve().parents[1]
foundation = {'__name__': 'frozen_fixture_only', '__file__': str(Q / 'tests/load_time_lora_test.py')}
exec(compile((Q / 'tests/load_time_lora_test.py').read_text(), str(Q / 'tests/load_time_lora_test.py'), 'exec'), foundation)
install = foundation['install']
function = foundation['function']

# Reuse the existing actual cache/lazy fixture setup up to its scenario body.
# Do not run its already-completed retry test or any old test methods.
method = copy.deepcopy(function(Q / 'tests/load_time_lora_test.py', 'test_failed_create_retry_reacquires_fresh_resource_and_publishes_once'))
cut = next(i for i, n in enumerate(method.body) if isinstance(n, ast.With))
method.body = method.body[:cut] + [ast.Return(value=ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[]))]
method.name = 'make_actual_fixture'
method.decorator_list = []
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<existing-fixture-prefix>', 'exec'), foundation)

MODE = os.environ.get('I15_PORTABLE_MUTANT', '')
OBSERVATIONS = []

class PublicChain(foundation['LoadTime']):
    def setUp(self):
        super().setUp()
        self.fixture = foundation['make_actual_fixture'](self)
        f = self.fixture
        qfe, handle, cache = f['qfe'], f['Handle'], f['engine_env']
        qfe.info = lambda *args, **kwargs: None
        qfe.console_safe = str
        handle.calls = 1  # skip the existing fixture's first-create refusal arm
        self.creates, self.updates = [], []
        original_create = handle.create
        def create(cls, lib, **kwargs):
            result = original_create(lib, **kwargs)
            cfg = copy.deepcopy(kwargs['create_params'].observed_cfg)
            self.creates.append(cfg)
            result.observed_cfg = cfg
            result.pipeline_update = lambda payload: self.updates.append(copy.deepcopy(payload))
            return result
        handle.create = classmethod(create)
        cache['_prepare_params'] = lambda model, cfg, device, key, path: NS(device_idx=device, observed_cfg=copy.deepcopy(cfg))
        self.env['QFLazyEngine'] = f['Lazy']
        self.env['qfe'] = qfe
        self.deps['get_engine'] = cache['_get_engine']
        family = copy.deepcopy(function(Q / 'qf_modelpatcher.py', 'family_build'))
        if MODE:
            class Mutate(ast.NodeTransformer):
                hits = 0
                def visit_Assign(m, n):
                    text = ast.unparse(n)
                    if MODE == 'omit_baked_stack' and text == 'create_cfg = dict(create_cfg or {}, lora=[dict(entry) for entry in lora_entries])':
                        m.hits += 1
                        return ast.copy_location(ast.Pass(), n)
                    if MODE == 'duplicate_runtime_overlay' and text == 'model._qf._lora = []':
                        m.hits += 1
                        return ast.copy_location(ast.Pass(), n)
                    return n
            mutation = Mutate()
            family = mutation.visit(family)
            if mutation.hits != 1:
                raise RuntimeError('Mutation did not target exactly one actual assignment')
        install([ast.fix_missing_locations(family)], self.env)
        for name in ['rebuild_of', 'lora_stack_of']:
            install([function(Q / 'qf_modelpatcher.py', name)], self.env)
        source = ast.parse((Q / '__init__.py').read_text())
        cls = next(n for n in ast.walk(source) if isinstance(n, ast.ClassDef) and n.name == 'QuantFuncNativeLoRA')
        env = {'qfmp': NS(rebuild_of=self.env['rebuild_of'], lora_stack_of=self.env['lora_stack_of']),
               '_resolve_lora': lambda name: str(self.path), '_lora_choices': lambda: ['adapter']}
        install([cls], env)
        self.node = env['QuantFuncNativeLoRA']()
        self.node._refuse_foreign_lora_format = lambda path: None
        self.base = self.build()([])

    def apply(self, patcher, strength=1.0, mode=None):
        kw = {} if mode is None else {'application': mode}
        return self.node.apply(patcher, 'adapter', strength, **kw)[0]

    def tearDown(self):
        OBSERVATIONS.append({'test': self._testMethodName, 'create_cfgs': self.creates,
                             'runtime_updates': self.updates, 'prepared_entries': len(self.fixture['entries']),
                             'published_handles': len(self.fixture['engine_env']['_PIPELINE_CACHE'])})
        super().tearDown()

    def test_public_identical_baked_stacks_reuse_one_handle(self):
        first = self.apply(self.base, mode='load_time')
        second = self.apply(self.base, mode='load_time')
        third = self.apply(self.base, mode='load_time')
        self.assertEqual(self.creates, [])
        real = first.model._qf.ensure()
        self.assertIs(second.model._qf.ensure(), real)
        self.assertIs(third.model._qf.ensure(), real)
        self.assertEqual(len(self.creates), 1)
        self.assertEqual(self.creates[0].get('lora'), [self.entry])
        self.assertEqual(self.updates, [])
        self.assertEqual(len(self.fixture['entries']), 1)

    def test_public_two_baked_nodes_create_complete_stack_once(self):
        first = self.apply(self.base, mode='load_time')
        final = self.apply(first, .5, 'load_time')
        self.assertIs(final.adopted, first)
        self.assertEqual(self.creates, [])
        final.model._qf.ensure()
        expected = [self.entry, dict(self.entry, scale=.5)]
        self.assertEqual([x.get('lora') for x in self.creates], [expected])
        self.assertEqual(final.model._stack, expected)
        self.assertEqual(self.updates, [])

    def test_public_baked_then_default_runtime_applies_full_stack_once(self):
        first = self.apply(self.base, mode='load_time')
        final = self.apply(first, .5)
        final.model._qf.ensure()
        final.model._qf.ensure()
        self.assertEqual(len(self.creates), 1)
        self.assertNotIn('lora', self.creates[0])
        self.assertEqual(self.updates, [{'lora': [self.entry, dict(self.entry, scale=.5)]}])

    def test_public_runtime_then_baked_leaves_no_runtime_merge(self):
        first = self.apply(self.base)
        final = self.apply(first, .5, 'load_time')
        final.model._qf.ensure()
        self.assertEqual([x.get('lora') for x in self.creates], [[self.entry, dict(self.entry, scale=.5)]])
        self.assertEqual(self.updates, [])

    def test_public_strength_and_order_distinguish_actual_cached_handles(self):
        one = self.apply(self.base, mode='load_time')
        half = self.apply(self.base, .5, 'load_time')
        ab = self.apply(one, .5, 'load_time')
        ba = self.apply(half, 1.0, 'load_time')
        retained = [x.model._qf.ensure() for x in [one, half, ab, ba]]
        self.assertEqual(len({id(x) for x in retained}), 4)
        self.assertEqual(len(self.creates), 4)
        self.assertEqual(self.updates, [])

    def test_public_changed_file_bytes_between_acquisitions_new_handle(self):
        first = self.apply(self.base, mode='load_time')
        first_real = first.model._qf.ensure()
        self.path.write_bytes(b'new lora same bytes')
        second = self.apply(self.base, mode='load_time')
        second_real = second.model._qf.ensure()
        self.assertIsNot(second_real, first_real)
        self.assertEqual(len(self.creates), 2)
        self.assertEqual(self.updates, [])

if __name__ == '__main__':
    names = sorted(n for n in PublicChain.__dict__ if n.startswith('test_'))
    suite = unittest.TestSuite(PublicChain(n) for n in names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'mode': MODE or 'original_candidate', 'tests': result.testsRun,
                      'failures': len(result.failures), 'errors': len(result.errors),
                      'observations': OBSERVATIONS,
                      'scope': 'Actual public apply/family builder/cache/lazy Python bodies; fake model/patcher/prepared/native/auth seams. No Torch/native/model. No lifetime/snapshot/merge math claim.'}, indent=2))
    sys.exit(not result.wasSuccessful())
