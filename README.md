# Offline generation in LM Evaluation Harness

CS-XX-003 v4.0.0 is a Code Sense pilot about maintaining a deterministic CPU generation backend through the public Language Model Evaluation Harness interface. The author repository includes tests and reference answers. Give solvers only the task instruction and the prepared source tree.

## Build and score

From a clone of this repository, initialize the pinned upstream source and materialize the starter before building the image:

```sh
git submodule update --init --depth 1 environment/upstream/lm-evaluation-harness
python3 environment/prepare_source.py --dest environment/workspace/repo
docker build --platform linux/amd64 -t cs-xx-003:4.0.0 environment
```

The upstream checkout is needed for preparation. Docker resolves the locked CPU dependencies at build time. An installed image can run the model interface and verifier without network or credentials. Start the solver in a fresh container as the default unprivileged user. Supply the content of `instruction.md` as its only prompt. The writable Python checkout inside the container is `/workspace/repo`. Its participant guide is `docs/model_guide/codesense-batch.md`; `instruction.md` includes a runnable registry/Instance example.

To run the untouched starter (NOP) and the reference (Golden), use a separate, offline container per answer:

```sh
docker run --rm --network none --user root \
  -v "$PWD/tests:/tests:ro" cs-xx-003:4.0.0 \
  sh -c 'sh /tests/test.sh && cat /logs/verifier/reward.txt'

docker run --rm --network none --user root \
  -v "$PWD/tests:/tests:ro" -v "$PWD/solution:/solution:ro" cs-xx-003:4.0.0 \
  sh -c 'sh /solution/solve.sh && sh /tests/test.sh && cat /logs/verifier/reward.txt'
```

The reference score is 1 and the untouched score is 0. Only candidate-facing tests count. Inspect `/logs/verifier/result.json` for each test method and its parameter rows. A failed assertion yields reward 0; a verifier error makes the run invalid. `tests/assets/selfcheck.md` identifies the control results and `CS-XX-003-external/06_baseline.md` gives the model-calibration status.

For an existing local workflow that invokes `tests/score.py repo result.json`, materialize the same source with `python3 environment/prepare_source.py --dest repo`, use the pinned Python dependencies from the environment, and run the command with that interpreter. The canonical Golden is a patch at `solution/golden/golden.patch`.

## Export for review

```sh
python3.12 tools/export_task.py ./dist
```

The exporter uses the selected Git commit (default `HEAD`) and its pinned upstream source. It writes a batch archive containing `CS-XX-003/{instruction.md,task.toml,environment,solution,tests}` and `manifest.csv`, plus a separate external reviewer archive. The batch includes the complete prepared checkout and third-party notices. Follow the source and distribution notes in `CS-XX-003-external/07_provenance_similarity.md` before sharing the materialized batch. The workload itself never downloads models or datasets.

## Pilot status

The 3.x public repository defined `repo/llm_runtime.py` as the candidate although that file matched its Golden byte for byte. Its evaluator also counted checks executed entirely on verifier-side code. Those commits remain in Git history. Version 4 uses new source and a new scorer; earlier Opus runs describe development on 3.x. No 4.0.0 five-model calibration or procurement import is claimed.
