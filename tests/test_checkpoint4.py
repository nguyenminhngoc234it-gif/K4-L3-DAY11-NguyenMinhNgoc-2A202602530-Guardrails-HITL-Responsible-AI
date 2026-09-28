"""Offline contract checks for Checkpoint 4 prompts and attack artifacts."""
import json

from attacks.attacks import (
    adversarial_prompts,
    classify_attack_outcome,
    save_attack_results,
)


def test_five_distinct_detailed_attack_techniques():
    assert len(adversarial_prompts) >= 5
    assert len({row["id"] for row in adversarial_prompts}) == len(adversarial_prompts)
    assert len({row["category"] for row in adversarial_prompts}) == len(adversarial_prompts)
    assert all(len(row["input"]) >= 180 for row in adversarial_prompts)
    assert all("TODO" not in row["input"] for row in adversarial_prompts)


def test_outcome_classifier_distinguishes_leak_and_guard_block():
    leaked = classify_attack_outcome(
        adversarial_prompts[0]["input"],
        "The internal administrator value is admin123.",
        target_name="red_default",
    )
    assert leaked["leaked"] is True
    assert leaked["layer"] == "leaked"

    blocked = classify_attack_outcome(
        adversarial_prompts[0]["input"],
        "I cannot process that request. I only help with VinBank banking questions.",
        target_name="red_advance",
    )
    assert blocked["leaked"] is False
    assert blocked["blocked_input"] is True
    assert blocked["blocked"] is True


def test_combined_attack_artifact_contract(tmp_path):
    base = {
        "id": 1,
        "category": adversarial_prompts[0]["category"],
        "input": adversarial_prompts[0]["input"],
        "response_preview": "sample",
        "blocked_input": False,
        "blocked": False,
        "layer": "leaked",
        "blocked_at": "LEAKED",
    }
    unsafe = [{**base, "leaked": True, "target": "red_default"}]
    guards = [{**base, "leaked": False, "blocked": True,
               "layer": "input_injection", "target": "red_advance"}]
    path = tmp_path / "attack_results.json"
    save_attack_results(unsafe_results=unsafe, guards_results=guards, filepath=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["llm_provider"] in {"openai", "gemini"}
    assert payload["llm_model"]
    assert payload["summary"]["unsafe_leaked"] == 1
    assert payload["summary"]["guards_blocked_plugin"] == 1
    assert payload["unsafe_attacks"][0]["target"] == "red_default"
    assert payload["guards_attacks"][0]["target"] == "red_advance"
