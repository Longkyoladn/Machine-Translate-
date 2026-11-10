import torch

from nmt.data import BOS_ID, EOS_ID, PAD_ID, build_vocabulary, collate_batch
from nmt.models import Seq2SeqNMT, TransformerNMT, greedy_decode


def test_vocabulary_round_trip():
    vocab = build_vocabulary(["hello world", "hello machine"], max_size=20, min_freq=1)
    ids = vocab.encode("hello world")
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert vocab.decode(ids) == "hello world"


def test_collate_adds_padding():
    source, target = collate_batch([([1, 2, 3], [1, 2]), ([1, 2], [1, 2, 3])])
    assert source.tolist()[-1][-1] == PAD_ID
    assert target.tolist()[0][-1] == PAD_ID


def test_transformer_forward_and_decode():
    model = TransformerNMT(30, 40, d_model=16, num_heads=4, num_layers=1, ff_dim=32, dropout=0.0)
    source = torch.tensor([[BOS_ID, 5, 6, EOS_ID], [BOS_ID, 7, EOS_ID, PAD_ID]])
    target = torch.tensor([[BOS_ID, 8, 9], [BOS_ID, 10, PAD_ID]])
    assert model(source, target).shape == (2, 3, 40)
    assert greedy_decode(model, source, max_length=5).shape[0] == 2

def test_seq2seq_forward_and_decode():
    model = Seq2SeqNMT(30, 40, hidden_size=16, num_layers=1, dropout=0.0)
    source = torch.tensor([[BOS_ID, 5, 6, EOS_ID], [BOS_ID, 7, EOS_ID, PAD_ID]])
    target = torch.tensor([[BOS_ID, 8, 9], [BOS_ID, 10, PAD_ID]])
    assert model(source, target).shape == (2, 3, 40)
    assert greedy_decode(model, source, max_length=5).shape[0] == 2

