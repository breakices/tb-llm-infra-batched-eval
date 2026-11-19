"""Observe public runtime behavior with a deterministic, instrumented model."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


class Backend:
    model_id = "small"

    def __init__(self, offset=0):
        self.offset = offset
        self.forwards = 0

    def row(self, prompt, size):
        # Tokenizers can use distinct vocabulary entries for True and 1.
        value = sum(19 if type(token) is bool else int(token) for token in prompt)
        return [int((self.offset + value + index) % 31) for index in range(size)]

    def next_tokens(self, prompt, adapter, temperature, max_new_tokens):
        self.forwards += 1
        return self.row(prompt, max_new_tokens)

    def next_tokens_batch(self, prompts, adapter, temperature, max_new_tokens, mode="padded"):
        self.forwards += 1
        return [self.row(prompt, max_new_tokens) for prompt in prompts]

    def prefill(self, tokens, adapter, temperature):
        return tuple(tokens)


def observe(repo):
    sys.path.insert(0, str(repo.resolve()))
    from llm_runtime import BatchEngine, KVCache, Request
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import CachingLM
    from lm_eval.api.registry import get_model

    def request(tenant, token):
        return Request(tenant, "small", "base", (token,), 2, 0.0)

    first = Backend()
    engine = BatchEngine(models={"small": first})
    original = engine.generate(request("east", True))
    integer = engine.generate(request("east", 1))
    before = first.forwards
    engine.generate(request("west", 1))
    tenant_cost = first.forwards - before
    replacement = Backend(offset=10)
    engine.register(replacement)
    replaced = engine.generate(request("east", True))
    cache = KVCache()
    cache.put((1,), [3, 4])
    cache.get((1,))[0] = 23
    observed = {
        "typed_token_outputs": {
            "expected": [first.row((True,), 2), first.row((1,), 2)],
            "actual": [original, integer],
        },
        "new_tenant_cold_forwards": {"expected": 1, "actual": tenant_cost},
        "replacement_output": {"expected": replacement.row((True,), 2), "actual": replaced},
        "detached_cache_read": {"expected": [3, 4], "actual": cache.get((1,))},
    }

    instance = Instance(
        request_type="generate_until", doc={}, arguments=("note", {"max_gen_toks": 3}),
        idx=0, metadata=("offline", 0, 1),
    )
    lm_class = get_model("codesense-batch")
    with TemporaryDirectory() as directory:
        database = str(Path(directory) / "replay.sqlite")
        for step, tenant in enumerate(("east", "west", "east")):
            backend = Backend()
            lm = lm_class(model="small", tenant=tenant, engine=BatchEngine({"small": backend}))
            cached = CachingLM(lm, database)
            try:
                if step == 0:
                    text = lm.generate_until([instance])[0]
                    lm.cache_hook.add_partial("generate_until", instance.args, text)
                else:
                    cached.generate_until([instance])
                    observed[f"reopen_partial_forwards_{tenant}"] = {
                        "expected": 1 if tenant == "west" else 0,
                        "actual": backend.forwards,
                    }
            finally:
                cached.dbdict.close()
    return observed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="working candidate source root")
    parser.add_argument("--require", choices=("correct", "defect"), required=True)
    args = parser.parse_args()
    observations = observe(args.repo)
    print(json.dumps(observations, indent=2))
    passes = all(item["actual"] == item["expected"] for item in observations.values())
    sys.exit(0 if passes == (args.require == "correct") else 1)
