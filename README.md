# CS-XX-003 · Offline batch generation

Version 4.0.0. [Task](instruction.md) · [Batch manifest](manifest.csv)

## Build

```sh
git submodule update --init --depth 1 environment/upstream/lm-evaluation-harness
python3 environment/prepare_source.py --dest environment/workspace/repo
docker build --platform linux/amd64 -t cs-xx-003:4.0.0 environment
```

Give the solver `instruction.md` and the prepared checkout at `/workspace/repo`. Keep the public author repository outside the solver’s filesystem and network access. Mount `tests/` after the solver exits.

## Verify

Run the untouched starter and reference solution in separate containers.

```sh
docker run --rm --network none --user root \
  -v "$PWD/tests:/tests:ro" cs-xx-003:4.0.0 \
  sh -c 'sh /tests/test.sh && cat /logs/verifier/reward.txt'

docker run --rm --network none --user root \
  -v "$PWD/tests:/tests:ro" -v "$PWD/solution:/solution:ro" cs-xx-003:4.0.0 \
  sh -c 'sh /solution/solve.sh && sh /tests/test.sh && cat /logs/verifier/reward.txt'
```

## Export

```sh
python3.12 tools/export_task.py ./dist
```

The exporter writes the task archive and a separate reviewer archive. The prepared checkout includes third-party material covered by the [source and distribution notes](CS-XX-003-external/07_provenance_similarity.md).
