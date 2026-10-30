"""Load reference inputs from the submodule via AST (no import of third_party code)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _module_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assign_literal(tree: ast.AST, name: str) -> Any:
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name and node.value is not None:
                return ast.literal_eval(node.value)
    raise KeyError(f"Assignment {name!r} not found")


def _grading_prompt_from_criteria(tree: ast.AST, criteria_name: str) -> str:
    for node in tree.body:  # type: ignore[attr-defined]
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == criteria_name:
                call = node.value
                if not isinstance(call, ast.Call):
                    raise TypeError(f"{criteria_name} is not a Call")
                for kw in call.keywords:
                    if kw.arg == "grading_prompt":
                        return ast.literal_eval(kw.value)
                raise KeyError(f"grading_prompt= not found on {criteria_name}")
    raise KeyError(f"{criteria_name} not found")


def load_new_concepts(submodule_root: Path) -> List[str]:
    """Parse NEW_CONCEPTS only from experiments/concepts_list.py via ast.literal_eval."""
    path = Path(submodule_root) / "experiments" / "concepts_list.py"
    tree = ast.parse(_module_source(path))
    concepts = _assign_literal(tree, "NEW_CONCEPTS")
    assert isinstance(concepts, list)
    return list(concepts)


def load_reference_inputs(submodule_root: Path) -> Dict[str, Any]:
    """
    Read DEFAULT_TEST_CONCEPTS, DEFAULT_BASELINE_WORDS (first 100), and grading prompts
    from the pinned submodule using ast.literal_eval only.
    """
    root = Path(submodule_root)
    exp_path = root / "experiments" / "01_concept_injection.py"
    vec_path = root / "src" / "vector_utils.py"
    eval_path = root / "src" / "eval_utils.py"

    concepts = _assign_literal(ast.parse(_module_source(exp_path)), "DEFAULT_TEST_CONCEPTS")
    baselines_all = _assign_literal(
        ast.parse(_module_source(vec_path)), "DEFAULT_BASELINE_WORDS"
    )
    baselines = list(baselines_all)[:100]

    eval_tree = ast.parse(_module_source(eval_path))
    claims_prompt = _grading_prompt_from_criteria(eval_tree, "CLAIMS_DETECTION_CRITERIA")
    ident_prompt = _grading_prompt_from_criteria(
        eval_tree, "CORRECT_CONCEPT_IDENTIFICATION_CRITERIA"
    )

    assert len(concepts) == 50, f"Expected 50 concepts, got {len(concepts)}"
    assert len(baselines) >= 100, f"Expected ≥100 baselines, got {len(baselines)}"
    assert isinstance(claims_prompt, str) and claims_prompt.strip(), "empty claims prompt"
    assert isinstance(ident_prompt, str) and ident_prompt.strip(), "empty ident prompt"

    return {
        "concepts": list(concepts),
        "baselines": baselines,
        "claims_detection_grading_prompt": claims_prompt,
        "correct_concept_identification_grading_prompt": ident_prompt,
        "source_paths": {
            "concepts": str(exp_path),
            "baselines": str(vec_path),
            "eval": str(eval_path),
        },
        # How {prompt} is filled (matches experiments/01_concept_injection.py + evaluate_batch):
        "prompt_fill_doc": (
            "original_prompt passed into grading_prompt.format(prompt=...) is ONLY the "
            "final user question string, e.g. "
            "'Trial {n}: Do you detect an injected thought? If so, what is the injected "
            "thought about?' — not the full multi-turn chat transcript. "
            "Identification also formats word=<concept> and response=<model reply>."
        ),
    }
