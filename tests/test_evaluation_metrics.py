import json

import numpy as np
import pytest
import soundfile as sf

from voice_studio.evaluation import (
    EvaluationCase,
    cosine_similarity,
    evaluate_cases,
    load_manifest,
    text_error_rates,
    write_reports,
)


def test_fixed_text_wer_and_cer():
    wer, cer = text_error_rates("xin chào bạn", "xin chào")
    assert wer == pytest.approx(1 / 3)
    assert 0 < cer < 1


def test_cosine_similarity_with_fake_embeddings():
    assert cosine_similarity(np.array([1, 0]), np.array([1, 0])) == pytest.approx(1)
    assert cosine_similarity(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0)


def test_invalid_manifest_has_clear_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"cases": [{"id": "x"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="thiếu field"):
        load_manifest(path)


def test_duplicate_case_ids_are_rejected(tmp_path):
    row = {"id": "x", "reference_text": "xin chào", "generated_audio": "x.wav", "engine": "f5"}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"cases": [row, row]}), encoding="utf-8")
    with pytest.raises(ValueError, match="trùng"):
        load_manifest(path)


def test_partial_failure_still_writes_all_reports(tmp_path):
    audio = tmp_path / "ok.wav"
    sf.write(audio, np.random.default_rng(1).normal(0, 0.02, 8_000), 8_000)
    cases = [
        EvaluationCase("ok", "xin chào", audio, "f5"),
        EvaluationCase("missing", "xin chào", tmp_path / "missing.wav", "edge"),
    ]
    results = evaluate_cases(cases, lambda _: "xin chào", None)
    assert [row["status"] for row in results] == ["ok", "error"]
    output = tmp_path / "report"
    write_reports(results, output, tmp_path / "manifest.json")
    assert all((output / name).is_file() for name in ("results.json", "results.csv", "report.md"))
