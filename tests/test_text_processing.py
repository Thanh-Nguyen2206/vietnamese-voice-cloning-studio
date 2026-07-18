from voice_studio.text_processing import normalize_vietnamese, split_long_text


def test_unicode_vietnamese_is_preserved():
    assert normalize_vietnamese("  Tiếng Việt rất đẹp!  ") == "tiếng việt rất đẹp!"


def test_long_text_chunks_keep_order_and_limit():
    text = "Câu thứ nhất rất rõ ràng. Câu thứ hai cũng rõ ràng. " * 8
    chunks = split_long_text(text, max_chars=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert " ".join(chunks).startswith("câu thứ nhất")


def test_does_not_split_decimal_date_abbreviation_email_or_url():
    text = (
        "TS. An đo được 3.14 vào ngày 18/07/2026 và gửi tới a.b@example.com. "
        "Xem https://example.com/a.b để biết thêm thông tin chi tiết."
    )
    chunks = split_long_text(text, max_chars=100)
    joined = " ".join(chunks)
    for protected in ("ts. an", "3.14", "18/07/2026", "a.b@example.com", "https://example.com/a.b"):
        assert protected in joined


def test_empty_and_invalid_mode():
    assert split_long_text("   ") == []
    try:
        split_long_text("xin chào", mode="bad")
    except ValueError as exc:
        assert "mode" in str(exc)
    else:
        raise AssertionError("invalid mode was accepted")


def test_over_2000_chars_is_chunked_by_policy():
    chunks = split_long_text("đây là một câu dài. " * 150, max_chars=180, mode="auto")
    assert len(chunks) > 10
    assert all(len(chunk) <= 180 for chunk in chunks)
