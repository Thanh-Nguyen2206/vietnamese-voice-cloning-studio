import torch

from scripts.train import create_collate_fn


def test_training_collate_uses_f5_padding_and_integer_mel_lengths():
    vocab = {" ": 0, "a": 1, "b": 2}
    collate = create_collate_fn(vocab, hop_length=256)
    batch = collate(
        [
            {"audio": torch.zeros(512), "audio_len": 512, "text": "ab"},
            {"audio": torch.zeros(256), "audio_len": 256, "text": "a"},
        ]
    )
    assert batch["lens"].dtype == torch.long
    assert batch["lens"].tolist() == [3, 2]
    assert batch["text"][1, -1].item() == -1
