import ast,copy,json,os,tempfile,unittest,threading,weakref
from pathlib import Path
from types import SimpleNamespace as NS
Q=Path(__file__).resolve().parents[1]
B=Q
def function(path,name):return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
def install(nodes,env):
    exec(compile(ast.Module(body=copy.deepcopy(nodes),type_ignores=[]),'actual_candidate_ast','exec'),env)
class Lazy:
    def __init__(self,factory,**kwargs):self.factory=factory;self._lora=[]
class Model:
    def __init__(self,engine):self._qf=engine
class Patcher:
    def __init__(self,model,**kwargs):self.model=model;self.adopted=None
    def adopt_comfy_state_from(self,old):self.adopted=old;return self
class LoadTime(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'adapter.safetensors';self.path.write_bytes(b'original lora bytes')
        self.captured=[]
        self.env=dict(QFLazyEngine=Lazy,QFModelPatcher=Patcher,QF_LORA_STACK_ATTR='_stack',QF_LORA_REBUILD_ATTR='_rebuild',LOADER_SIG=NS(get=lambda:'sig'),LOADER_API_KEY=NS(get=lambda:None),current_torch_device=lambda:('cpu',0),ensure_model_config_attrs=lambda m:None,comfy=NS(model_management=NS(unet_offload_device=lambda:'cpu')),qfe=NS(info=lambda *a,**k:None))
        install([function(Q/'qf_modelpatcher.py',n) for n in ['make_engine_factory','tag_lora_rebuild','family_build']],self.env)
        self.deps={'get_engine':lambda *a,**k:self.captured.append(k) or (NS(),'key'),'bind_pipeline_model':lambda *a:None,'read_auth':lambda *a:None}
        self.entry={'path':str(self.path),'scale':1.0,'target':'all'}
    def tearDown(self):self.tmp.cleanup()
    def build(self,bf16=True):
        return self.env['family_build'](self.deps,'model',{'transformer_precision':'bf16'} if bf16 else None,lambda c:NS(),{},lambda c,e,d:Model(e),'label')
    def keyenv(self):
        env=dict(os=os,json=json,qfe=NS(load_lib=lambda:'lib',loaded_so_path=lambda:'engine',library_identity=lambda x:'lib-id'))
        install([function(Q/'__init__.py',n) for n in ['_load_time_lora_content_identity','_engine_recipe']],env);return env
    def test_default_runtime_recipe_overlay_unchanged(self):
        patch=self.build()([self.entry]);patch.model._qf.factory()
        self.assertNotIn('lora',self.captured[-1]['create_cfg']);self.assertEqual(patch.model._qf._lora,[self.entry])
    def test_loadtime_sent_once_empty_runtime_overlay(self):
        patch=self.build()([self.entry],application='load_time');self.assertEqual(self.captured,[])
        patch.model._qf.factory();self.assertEqual(self.captured[-1]['create_cfg']['lora'],[self.entry]);self.assertEqual(patch.model._qf._lora,[]);self.assertEqual(patch.model._stack,[self.entry])
    def test_chain_rebuild_keeps_entire_stack(self):
        patch=self.build()([self.entry],application='load_time');second=dict(self.entry,scale=.5)
        chain=patch.model._rebuild(patch.model._stack+[second],application='load_time');chain.model._qf.factory()
        self.assertEqual(self.captured[-1]['create_cfg']['lora'],[self.entry,second]);self.assertEqual(chain.model._qf._lora,[])
    def test_nonbf16_explicit_refusal_and_runtime_preserved(self):
        build=self.build(False);self.assertFalse(build.supports_load_time_lora)
        with self.assertRaisesRegex(ValueError,'BF16'):build([self.entry],application='load_time')
        self.assertEqual(build([self.entry]).model._qf._lora,[self.entry])
    def test_recipe_cache_separates_base_strength_order_and_content(self):
        recipe=self.keyenv()['_engine_recipe'];key=lambda entries:recipe('model',{'lora':entries})[1]
        a,b=self.entry,dict(self.entry,scale=.5)
        keys=[key([]),key([a]),key([b]),key([a,b]),key([b,a])];self.assertEqual(len(set(keys)),5)
        other=self.path.with_name('same-bytes-other-path.safetensors');other.write_bytes(self.path.read_bytes())
        self.assertNotEqual(key([a]),key([dict(a,path=str(other))]))
        cfg={'lora':[a]};self.assertNotEqual(recipe('model',cfg)[1],recipe('other-model',cfg)[1])
        first=key([a]);self.path.write_bytes(b'new lora same bytes');self.assertNotEqual(first,key([a]))
    def test_legacy_key_unchanged_without_lora(self):
        recipe=self.keyenv()['_engine_recipe']
        for cfg in [None,{}, {'transformer_precision':'bf16'}]:
            key=('engine','model',None,'svdq',0,json.dumps(cfg or {},sort_keys=True))
            self.assertEqual(recipe('model',cfg),('lib',key,('lib-id',key),cfg))
    def test_public_node_dispatch_and_comfy_state_adoption(self):
        cls=next(n for n in ast.walk(ast.parse((Q/'__init__.py').read_text())) if isinstance(n,ast.ClassDef) and n.name=='QuantFuncNativeLoRA')
        env={'qfmp':NS(rebuild_of=lambda p:p.model._rebuild,lora_stack_of=lambda p:list(p.model._stack)), '_resolve_lora':lambda n:str(self.path),'_lora_choices':lambda:['adapter']}
        install([cls],env);node=env['QuantFuncNativeLoRA']();node._refuse_foreign_lora_format=lambda p:None
        base=self.build()([]);out=node.apply(base,'adapter',1,application='load_time')[0]
        self.assertIs(out.adopted,base);out.model._qf.factory();self.assertEqual(self.captured[-1]['create_cfg']['lora'],[self.entry]);self.assertEqual(out.model._qf._lora,[])
        runtime=node.apply(base,'adapter',1)[0];runtime.model._qf.factory();self.assertNotIn('lora',self.captured[-1]['create_cfg'])
        with self.assertRaisesRegex(RuntimeError,'BF16'):node.apply(self.build(False)([]),'adapter',1,application='load_time')
    def test_actual_runtime_update_consumer_does_not_reapply_baked_stack(self):
        patch=self.build()([self.entry],application='load_time')
        calls=[]
        real=NS(applied_lora_sig='[]',pipeline_update=lambda cfg:calls.append(cfg))
        lazy=patch.model._qf;lazy._real=real
        env={};install([function(B/'qf_modelpatcher.py','_apply_runtime_lora')],env)
        lazy._lora_sig=lambda:json.dumps(lazy._lora,sort_keys=True)
        lazy.lora_set=lambda:list(lazy._lora)
        self.assertFalse(env['_apply_runtime_lora'](lazy));self.assertEqual(calls,[])
    def test_mixed_chains_last_explicit_mode_controls_whole_stack(self):
        build=self.build();first=build([self.entry],application='load_time')
        second=dict(self.entry,scale=.25)
        runtime=first.model._rebuild(first.model._stack+[second])
        runtime.model._qf.factory()
        self.assertNotIn('lora',self.captured[-1]['create_cfg'])
        self.assertEqual(runtime.model._qf._lora,[self.entry,second])
        baked=runtime.model._rebuild(runtime.model._stack,application='load_time')
        baked.model._qf.factory()
        self.assertEqual(self.captured[-1]['create_cfg']['lora'],[self.entry,second])
        self.assertEqual(baked.model._qf._lora,[])
    def test_content_change_during_hash_is_refused(self):
        env=self.keyenv();original_open=open;target=self.path
        class ChangingFile:
            def __init__(self,path,mode):self.file=original_open(path,mode);self.changed=False
            def __enter__(self):return self
            def __exit__(self,*args):return self.file.__exit__(*args)
            def fileno(self):return self.file.fileno()
            def read(self,n):
                data=self.file.read(n)
                if data and not self.changed:
                    self.changed=True
                    with original_open(target,'ab') as other:other.write(b'changed')
                return data
        env['open']=ChangingFile
        with self.assertRaisesRegex(RuntimeError,'changed while computing'):
            env['_load_time_lora_content_identity']({'lora':[self.entry]})
    def test_failed_create_retires_prepared_entry_without_publishing(self):
        consumer=Model(None);retired=[];pipeline_cache={};prepared_cache={}
        def fail_create(*args,**kwargs):raise RuntimeError('native create refused')
        entry=NS(_materialize_lock=threading.Lock(),_cache_usable=True,_materializers=[consumer],create_params=object(),capacity_bytes=1,resource=object(),retire_materialized=lambda:retired.append(True))
        prepared_cache[('lib-id','key')]=entry
        env=dict(qfmp=NS(_current_engine_cache_acquisition=lambda:NS(consumer=consumer)),qfe=NS(NativeContractUnavailable=RuntimeError,library_identity=lambda x:'lib-id',QFEngineHandle=NS(create=fail_create)),weakref=weakref,_ENGINE_IDENTITY_LOCK=threading.Lock(),_PIPELINE_CACHE=pipeline_cache,_PREPARED_CACHE=prepared_cache,_pin_pipeline_consumer_locked=lambda k:None,_sweep_dead_pipelines=lambda k:None)
        install([function(Q/'__init__.py','_materialize_engine')],env)
        with self.assertRaisesRegex(RuntimeError,'native create refused'):env['_materialize_engine'](entry,'lib','key')
        self.assertEqual(retired,[True]);self.assertEqual(prepared_cache,{});self.assertEqual(pipeline_cache,{})
    def test_failed_create_retry_reacquires_fresh_resource_and_publishes_once(self):
        import contextvars, contextlib
        def node(file, name):
            return function(Q / file, name)
        acquisition=contextvars.ContextVar('acquisition',default=None)
        @contextlib.contextmanager
        def acquiring(consumer):
            token=acquisition.set(NS(consumer=consumer))
            try: yield
            finally: acquisition.reset(token)
        class Handle:
            calls=0
            @classmethod
            def create(cls,lib,*,create_params,capacity_bytes,prepared_resource):
                cls.calls+=1
                if cls.calls==1: raise RuntimeError('injected first create refusal')
                result=cls(); result.lib=lib; result.pipeline=object();result.resource=prepared_resource
                result.capacity_bytes=capacity_bytes;result.applied_lora_sig='[]'
                return result
        qfe=NS(load_lib=lambda:'lib',loaded_so_path=lambda:'image',library_identity=lambda lib:'image-id',NativeContractUnavailable=RuntimeError,QFEngineHandle=Handle,FACTORY_PREPARE_ONLY=contextvars.ContextVar('prepare-only',default=False))
        entries=[]
        class Entry:
            def __init__(self,lib,device,*,create_params,api_key):
                self.lib=lib;self.create_params=create_params;self.api_key=api_key
                self.capacity_bytes=64;self.cache_key=None;self._cache_usable=True
                self._materializers=weakref.WeakSet();self._materialize_lock=threading.RLock()
                self.resource=object();self._qf_resource_adapters={};entries.append(self)
            def retire_unpublished(self): self._cache_usable=False
        entry_env={}
        for name in ['retire_materialized','bind_materializer','materialize']:
            env={'qfe':qfe};install([node('qf_modelpatcher.py',name)],env);setattr(Entry,name,env[name])
        qfmp=NS(QFPreparedEntry=Entry,_current_engine_cache_acquisition=lambda:acquisition.get())
        engine_env=dict(os=os,json=json,qfe=qfe,qfmp=qfmp,weakref=weakref,_ENGINE_IDENTITY_LOCK=threading.RLock(),_PIPELINE_CACHE={},_PREPARED_CACHE={},_prepare_params=lambda *a,**k:NS(device_idx=0),_pin_pipeline_consumer_locked=lambda k:None,_sweep_dead_pipelines=lambda k:None)
        install([node('__init__.py',name) for name in ['_load_time_lora_content_identity','_engine_recipe','_get_or_prepare_entry','_materialize_engine','_get_engine']],engine_env)
        # Extract the actual lazy lifecycle, supplying native/resource objects only at their external seams.
        class Lazy: pass
        for name in ['__init__','prepare_resource','_bind_capacity','_adopt_real','_materialize_prepared','ensure','_apply_runtime_lora','lora_set','_lora_sig']:
            n=next(n for n in ast.walk(ast.parse((Q/'qf_modelpatcher.py').read_text())) if isinstance(n,ast.ClassDef) and n.name=='QFLazyEngine')
            method=next(c for c in n.body if isinstance(c,ast.FunctionDef) and c.name==name)
            env=dict(qfe=qfe,QFPreparedEntry=Entry,_engine_cache_acquisition=acquiring,json=json)
            install([method],env);setattr(Lazy,name,env[name])
        Lazy._apply_runtime_api_key=lambda self:False
        with __import__('tempfile').TemporaryDirectory(dir=self.tmp.name) as temp:
            path=Path(temp)/'adapter';path.write_bytes(b'stable synthetic adapter')
            lazy=Lazy(lambda:engine_env['_get_engine']('model',{'transformer_precision':'bf16','lora':[{'path':str(path),'scale':1.0,'target':'all'}]}))
            first_failure=None
            try:lazy.ensure()
            except RuntimeError as exc:first_failure=str(exc)
            first=entries[0]
            after_failure={'error':first_failure,'published':len(engine_env['_PIPELINE_CACHE']),'prepared_cached':len(engine_env['_PREPARED_CACHE']),'old_entry_usable':first._cache_usable,'old_materializers':len(first._materializers),'lazy_has_real':lazy._real is not None}
            real=lazy.ensure()
            result={'scope':'stdlib synthetic native seam; actual frozen Python cache/lazy lifecycle bodies; not native/Torch/E2E','after_first_failure':after_failure,'retry':{'create_calls':Handle.calls,'prepared_epochs':len(entries),'fresh_resource':entries[1].resource is not first.resource,'prepared_entry_rebound':lazy._prepared_entry is entries[1],'published':len(engine_env['_PIPELINE_CACHE']),'lazy_real_is_result':lazy._real is real,'runtime_overlay':lazy._lora}}
            result['passed']=after_failure=={'error':'injected first create refusal','published':0,'prepared_cached':0,'old_entry_usable':False,'old_materializers':0,'lazy_has_real':False} and Handle.calls==2 and len(entries)==2 and entries[1].resource is not first.resource and lazy._prepared_entry is entries[1] and len(engine_env['_PIPELINE_CACHE'])==1 and lazy._real is real
            self.assertTrue(result['passed'], result)
            third_consumer = Lazy(lazy._factory)
            third_real = third_consumer.ensure()
            self.assertIs(third_real, real)
            self.assertEqual(Handle.calls, 2)
            self.assertEqual(len(entries), 2)
            self.assertEqual(len(engine_env['_PIPELINE_CACHE']), 1)
if __name__=='__main__':unittest.main(verbosity=2)
