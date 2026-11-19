#!/usr/bin/env python3
"""Synchronize Harbor metadata and reviewer tables with the actual verifier index."""
import argparse
import csv
import io
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TASK = "CS-XX-003"
EXTERNAL = TASK + "-external"
VERSION = "4.0.0"
PROMPT_START = "# BEGIN embedded solver instruction"
PROMPT_END = "# END embedded solver instruction"
INDEX_START = "# BEGIN verifier index"

# All entries in the public verifier registry carry these procurement fields.
FIELDS = ("id", "ut_name", "turn", "dimension", "description", "criterion_necessity",
          "judge_type", "targets_badpattern", "change_points", "evidence_required")

CP_REQUIREMENTS = {
    "CP1": "兼容单请求、输入校验、输出上限、截断和停止条件",
    "CP2": "批次与逐条生成一致，按输入位置返回，节省模型工作",
    "CP3": "相容工作可复用，运行主体、不同配置及模型变更隔离",
    "CP4": "注册模型的 generate_until 输入、续写文本与位置契约",
    "CP5": "两种布局按实际输入计量",
    "CP6": "进程重启、配置切换、部分结果和旧模型磁盘缓存兼容",
}


def literal(value):
    return json.dumps(value, ensure_ascii=False)


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def generate():
    metadata_path = ROOT / "task.toml"
    source = metadata_path.read_text()
    prefix, rest = source.split(PROMPT_START, 1)
    _, rest = rest.split(PROMPT_END, 1)
    stable, _ = rest.split(INDEX_START, 1)
    instruction = (ROOT / "instruction.md").read_text()
    if "'''" in instruction:
        raise ValueError("instruction contains the TOML literal delimiter")
    entries = json.loads((ROOT / "tests/ut_index.json").read_text())
    if len(entries) < 21 or len({e["ut_name"] for e in entries}) != len(entries):
        raise ValueError("verifier must declare at least 21 distinct methods")
    if [e["id"] for e in entries] != [f"U{i+1}" for i in range(len(entries))]:
        raise ValueError("UT IDs are out of order")
    by_dim = {}
    for e in entries:
        by_dim.setdefault(e["dimension"], []).append(e["ut_name"])
    flagged = sum(e["targets_badpattern"] for e in entries)
    header = [prefix + PROMPT_START, "prompt = '''", instruction + "'''",
              PROMPT_END + stable + INDEX_START, "[metadata.unit_tests]",
              f"count = {len(entries)}", f"targets_badpattern_count = {flagged}"]
    for dim, names in by_dim.items():
        header.append(f"{literal(dim)} = {literal(names)}")
    header += ["", "[metadata.verification]", "human_required = false",
               "test_is_sufficient = true", 'judge_type = "unit_test"',
               'scoring_scripts = ["tests/test.sh", "tests/score.py"]',
               f"ut_count = {len(entries)}"]
    for e in entries:
        header.extend(["", "[[metadata.ut_index]]"])
        header.extend(f"{f} = {literal(e[f])}" for f in FIELDS)
    metadata = "\n".join(header) + "\n"
    data = tomllib.loads(metadata)["metadata"]
    assert data["prompt"] == instruction
    fm_coverage = {i for fm in data["model_failure_modes"] for i in fm["unit_tests"]}
    assert {e["id"] for e in entries if e["targets_badpattern"]} <= fm_coverage

    catalog = ["# 单测清单", "", f"具名方法 {len(entries)} 个，其中 {flagged} 个直接针对历史试答中的行为。每项的参数行均通过才记该方法通过。", "",
               "| ID | 行数 | 方法 | 维度 | 判定行为 |", "|---|---:|---|---|---|"]
    mapping = ["# 题面、单测与证据映射", "",
               "| 单测 | 对应业务要求 | 改动/回归点 | 可观测证据 |", "|---|---|---|---|"]
    for e in entries:
        cps = e["change_points"].split("+")
        catalog.append(f"| {e['id']} | {len(e['cases'])} | `{e['ut_name']}` | {e['dimension']} | {cell(e['description'])} |")
        mapping.append(f"| {e['id']} | {'、'.join(CP_REQUIREMENTS[c] for c in cps)} | {e['change_points']} | {cell(e['evidence_required'])} |")

    manifest = io.StringIO()
    writer = csv.writer(manifest, lineterminator="\n")
    writer.writerow("id,title,version,badpattern,trap_type,platform,focus_points,has_embedded_challenge,challenge_premise_is_false,model_failure_modes,ut_count,badpattern_ut_count,score_full,score_equivalent,score_guessed,score_defective,score_golden,score_nop,precheck_evidence,determinism_3_runs_identical,code_origin,code_license,desensitisation,similarity_check,known_risks,rework_history,status".split(","))
    results_file = ROOT / "tests/assets/controls/matrix.json"
    matrix = json.loads(results_file.read_text()) if results_file.exists() else {}
    profiles = matrix.get("profiles", {})
    scores = [profiles.get(n, {}).get("reward", "") for n in
              ("full_score", "equivalent", "guessed", "defective", "golden", "nop")]
    stable_three = matrix.get("repetitions", 0) >= 3 and len(profiles) == 8
    writer.writerow([TASK, data["title"], VERSION, data["badpattern"], data["trap_type"],
                     data["platform"], "②;④;⑥", "true", "true", "FM1;FM2;FM3", len(entries),
                     flagged, *scores, "tests/assets/selfcheck.md", "是" if stable_three else "待完成",
                     "opensource", data["code_license"], "公开上游代码与新写回放后端",
                     EXTERNAL + "/07_provenance_similarity.md",
                     "未完成当前题面五模型各五次独立校准及平台样包导入",
                     "4.0.0 使用真实 harness 和候选驱动的独立验收，3.x 成绩仅供开发参考",
                     "calibration_incomplete"])
    return {"task.toml": metadata, EXTERNAL + "/02_ut_list.md": "\n".join(catalog) + "\n",
            EXTERNAL + "/03_prompt_ut_evidence_map.md": "\n".join(mapping) + "\n",
            "manifest.csv": manifest.getvalue()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    check = parser.parse_args().check
    for name, content in generate().items():
        path = ROOT / name
        if check:
            if not path.is_file() or path.read_bytes() != content.encode():
                raise SystemExit(f"regenerate {name} with {Path(__file__).name}")
        else:
            path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
