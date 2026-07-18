"""Vietnamese text normalization and boundary-aware long-text chunking."""

from __future__ import annotations

import re
import unicodedata

_PROTECTED = re.compile(
    r"(?:https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"\b\d+[.,]\d+\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|"
    r"\b(?:tp\.hcm|tp\.|ts\.|ths\.|pgs\.|gs\.|bs\.|mr\.|mrs\.|dr\.|v\.v\.|"
    r"v\.d\.|etc\.)|(?:\b[A-ZĐ](?:\.[A-ZĐ]){1,}\.?))",
    re.IGNORECASE,
)


def normalize_vietnamese(text: str, *, lowercase: bool = True, add_end_punct: bool = False) -> str:
    """Normalize Unicode and whitespace while preserving Vietnamese diacritics."""

    value = unicodedata.normalize("NFC", text or "").strip()
    value = value.replace("…", ".").replace("“", '"').replace("”", '"').replace("’", "'")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.!?;:])", r"\1", value)
    if lowercase:
        value = value.lower()
    if add_end_punct and value and value[-1] not in ".!?,;:\"')":
        value += "."
    return value


def _mask_protected(text: str) -> tuple[str, dict[str, str]]:
    values: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        key = f"\ue000{len(values)}\ue001"
        values[key] = match.group(0)
        return key

    return _PROTECTED.sub(replace, text), values


def _restore(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace(key, value)
    return text


def _hard_split(text: str, max_chars: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(word[i : i + max_chars] for i in range(0, len(word), max_chars))
            continue
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_long_text(text: str, max_chars: int = 280, mode: str = "auto") -> list[str]:
    """Split text in order at sentence/clause boundaries without breaking protected forms."""

    if mode not in {"off", "auto", "aggressive"}:
        raise ValueError("mode phải là off, auto hoặc aggressive")
    if max_chars < 80:
        raise ValueError("max_chars phải >= 80")
    normalized = normalize_vietnamese(text, lowercase=True, add_end_punct=True)
    if not normalized:
        return []
    if mode == "off" or len(normalized) <= max_chars:
        return [normalized]

    masked, protected = _mask_protected(normalized)
    boundary = r"(?<=[.!?])\s+"
    if mode == "aggressive":
        boundary = r"(?<=[.!?;:,])\s+"
    sentences = re.split(boundary, masked)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(_restore(candidate, protected)) > max_chars:
            chunks.append(current)
            current = ""
        if len(_restore(sentence, protected)) <= max_chars:
            current = f"{current} {sentence}".strip()
            continue
        clauses = re.split(r"(?<=[;:,])\s+", sentence)
        for clause in clauses:
            candidate = f"{current} {clause}".strip()
            if current and len(_restore(candidate, protected)) > max_chars:
                chunks.append(current)
                current = ""
            if len(_restore(clause, protected)) > max_chars:
                chunks.extend(_hard_split(clause, max_chars))
            else:
                current = f"{current} {clause}".strip()
    if current:
        chunks.append(current)
    return [_restore(chunk, protected).strip() for chunk in chunks if chunk.strip()]
