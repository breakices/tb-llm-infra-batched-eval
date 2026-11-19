# Development evidence from earlier tasks

The following excerpts come from local Claude Code submissions and failed pytest assertions. All three were collected on 2026-11-18 from `claude-opus-5-5`. The paths below identify files in each historical submission.

| Case | Run | Submitted code and observation |
|---|---|---|
| FM1 | `eval-llm-v3-contract-fixed`, session `92caf506-07db-4161-9bf5-f3f727c7f19d` | `repo/llm_runtime.py:380` groups by `(_typed(row.model), _typed(row.adapter), row.temperature)`. With identical rows owned by two tenants the assertion on `model.forward_calls` was `assert 1 >= 2`. |
| FM2 | `eval-llm-v3-contract-fixed-pre-final`, session `2da6f75b-749f-4142-8931-2eb8770815d4` | `repo/llm_runtime.py:216` returns `len(self._entries) + len(self._prefixes)`. With prompts `(1,)` and `(True,)` the result at `len(e.cache)` was 4; expected 2. The assertions for equal output values and two forwards both succeeded. |
| FM3 | `eval-hard-llm-infra`, session `a4d05357-6016-4bd2-8754-eaf2713dfba3` | `repo/llm_runtime.py:251` rejects prompts at `if not all(_is_int(token) for token in prompt):`. The helper excludes `bool`. Both typed-prompt tests failed before checking cache entries. |

In v3 one of the tenant assertions required two forward invocations. V4 permits independent tenant rows to share an invocation if both rows do their own backend work. The v3 call-count failure alone does not establish a v4 semantic failure.
