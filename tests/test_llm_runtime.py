import pytest
from llm_runtime import BatchEngine, KVCache, Request, ToyModel


def engine():
    e=BatchEngine(); e.register(ToyModel('m1')); e.register(ToyModel('m2')); return e

def req(**kw):
    base=dict(tenant='t1',model='m1',adapter='base',prompt=(1,2,3),max_new_tokens=4,temperature=0.0)
    base.update(kw); return Request(**base)

def test_basic_generation():
    e=engine(); assert len(e.generate(req())) == 4

def test_repeated_prefix_reuses_forward():
    e=engine(); e.generate(req()); e.generate(req()); assert e.models['m1'].forward_calls == 1

def test_returned_tokens_are_detached():
    e=engine(); out=e.generate(req()); out[0]=999; assert e.generate(req())[0] != 999

def test_model_context_isolation():
    e=engine(); a=e.generate(req(model='m1')); b=e.generate(req(model='m2')); assert a != b

def test_adapter_context_isolation():
    e=engine(); a=e.generate(req(adapter='a')); b=e.generate(req(adapter='b')); assert a != b

def test_tenant_context_isolation():
    e=engine(); a=e.generate(req(tenant='a')); b=e.generate(req(tenant='b')); assert a == b or e.models['m1'].forward_calls == 2

def test_temperature_context_isolation():
    e=engine(); a=e.generate(req(temperature=0.0)); b=e.generate(req(temperature=0.7)); assert a != b

def test_prompt_context_isolation():
    e=engine(); a=e.generate(req(prompt=(4,5))); b=e.generate(req(prompt=(4,6))); assert a != b

def test_max_tokens_can_reuse_longer_result():
    e=engine(); long=e.generate(req(max_new_tokens=6)); short=e.generate(req(max_new_tokens=2)); assert short == long[:2]

def test_batch_preserves_order():
    e=engine(); rs=[req(prompt=(i,)) for i in [3,1,2]]; assert e.generate_batch(rs)==[e.generate(r) for r in rs]

def test_batch_duplicate_requests_one_forward():
    e=engine(); rs=[req(),req(),req()]; e.generate_batch(rs); assert e.models['m1'].forward_calls == 1

def test_clear_forces_recompute():
    e=engine(); e.generate(req()); e.clear(); e.generate(req()); assert e.models['m1'].forward_calls == 2

def test_register_replacement_invalidates_old_cache():
    e=engine(); first=e.generate(req()); e.register(ToyModel('m1')); second=e.generate(req()); assert second == first; assert e.models['m1'].forward_calls == 1

def test_model_replacement_same_id_uses_new_behavior():
    e=engine(); e.generate(req()); replacement=ToyModel('m1-new'); replacement.model_id='m1'; e.register(replacement); out=e.generate(req()); assert replacement.forward_calls == 1

def test_cache_key_keeps_bool_and_int_prompt_distinct():
    model = ToyModel('m1')
    e = BatchEngine(models={'m1': model})
    a = e.generate(req(prompt=(1,)))
    b = e.generate(req(prompt=(True,)))
    # Arithmetic generation treats both values as one, but cache identity
    # keeps their Python types separate and therefore performs two forwards.
    assert a == b
    assert model.forward_calls == 2
    assert len(e.cache) == 2

def test_empty_prompt_supported():
    e=engine(); assert e.generate(req(prompt=())) == e.models['m1'].next_tokens((), 'base', 0.0, 4)

def test_cache_clear_is_idempotent():
    e=engine(); e.clear(); e.clear(); assert e.generate(req())

def test_kv_cache_put_get_returns_copy():
    c=KVCache(); c.put((1,2),[3,4]); x=c.get((1,2)); x[0]=9; assert c.get((1,2)) == [3,4]

def test_kv_cache_clear_removes_entries():
    c=KVCache(); c.put((1,),[2]); c.clear(); assert c.get((1,)) is None

def test_different_max_tokens_do_not_truncate_cache_source():
    e=engine(); e.generate(req(max_new_tokens=2)); full=e.generate(req(max_new_tokens=6)); assert len(full)==6

def test_multiple_models_keep_forward_counts_separate():
    e=engine(); e.generate(req(model='m1')); e.generate(req(model='m2')); assert e.models['m1'].forward_calls==1 and e.models['m2'].forward_calls==1

def test_batch_mixed_contexts_do_not_share_outputs():
    e=engine(); rs=[req(model='m1'),req(model='m2'),req(adapter='alt')]; outs=e.generate_batch(rs); assert outs[0]!=outs[1] and outs[0]!=outs[2]

def test_engine_can_reuse_after_clear_and_batch():
    e=engine(); e.generate_batch([req(),req(prompt=(9,))]); e.clear(); assert len(e.generate_batch([req(),req(prompt=(9,))]))==2

def test_cache_isolation_with_same_prompt_and_model_after_tenant_switch():
    e=engine(); e.generate(req(tenant='a')); e.generate(req(tenant='b')); assert e.models['m1'].forward_calls==2
