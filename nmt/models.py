from __future__ import annotations

import math

import torch
from torch import nn

from .data import BOS_ID, EOS_ID, PAD_ID


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 512) -> None:
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.encoding[:, : inputs.size(1)]


class TransformerNMT(nn.Module):
    def __init__(self, source_vocab_size: int, target_vocab_size: int, d_model: int = 256,
                 num_heads: int = 4, num_layers: int = 3, ff_dim: int = 1024,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.d_model = d_model
        self.source_embedding = nn.Embedding(source_vocab_size, d_model, padding_idx=PAD_ID)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model, padding_idx=PAD_ID)
        self.position = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.output = nn.Linear(d_model, target_vocab_size)

    def forward(self, source: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        source_padding = source.eq(PAD_ID)
        target_padding = target_input.eq(PAD_ID)
        target_mask = torch.triu(
            torch.ones(target_input.size(1), target_input.size(1), dtype=torch.bool,
                       device=target_input.device), diagonal=1
        )
        source_emb = self.position(self.source_embedding(source) * math.sqrt(self.d_model))
        target_emb = self.position(self.target_embedding(target_input) * math.sqrt(self.d_model))
        hidden = self.transformer(
            source_emb, target_emb, tgt_mask=target_mask,
            src_key_padding_mask=source_padding, tgt_key_padding_mask=target_padding,
            memory_key_padding_mask=source_padding,
        )
        return self.output(hidden)


class LuongAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, query: torch.Tensor, encoder_outputs: torch.Tensor,
                source_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self.key(encoder_outputs)
        scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(~source_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class Seq2SeqNMT(nn.Module):
    def __init__(self, source_vocab_size: int, target_vocab_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.source_embedding = nn.Embedding(source_vocab_size, hidden_size, padding_idx=PAD_ID)
        self.target_embedding = nn.Embedding(target_vocab_size, hidden_size, padding_idx=PAD_ID)
        self.encoder = nn.GRU(
            hidden_size, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True, bidirectional=True,
        )
        self.bridge = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder_cells = nn.ModuleList(
            [nn.GRUCell(hidden_size + hidden_size * 2, hidden_size)]
            + [nn.GRUCell(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )
        self.attention = LuongAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size + hidden_size * 2, target_vocab_size)

    def encode(self, source: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        lengths = source.ne(PAD_ID).sum(1).cpu()
        embedded = self.dropout(self.source_embedding(source))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.encoder(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=source.size(1)
        )
        layer_states: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            forward_state = hidden[layer * 2]
            backward_state = hidden[layer * 2 + 1]
            layer_states.append(torch.tanh(self.bridge(torch.cat([forward_state, backward_state], dim=1))))
        return outputs, layer_states, source.ne(PAD_ID)

    def decode_step(self, token: torch.Tensor, states: list[torch.Tensor], context: torch.Tensor,
                    encoder_outputs: torch.Tensor, source_mask: torch.Tensor
                    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        current = torch.cat([self.target_embedding(token), context], dim=1)
        new_states: list[torch.Tensor] = []
        for layer, cell in enumerate(self.decoder_cells):
            state = cell(current, states[layer])
            new_states.append(state)
            current = self.dropout(state)
        context, _ = self.attention(new_states[-1], encoder_outputs, source_mask)
        logits = self.output(torch.cat([new_states[-1], context], dim=1))
        return logits, new_states, context

    def forward(self, source: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        encoder_outputs, states, source_mask = self.encode(source)
        context = encoder_outputs.new_zeros(source.size(0), self.hidden_size * 2)
        logits = []
        for step in range(target_input.size(1)):
            step_logits, states, context = self.decode_step(
                target_input[:, step], states, context, encoder_outputs, source_mask
            )
            logits.append(step_logits)
        return torch.stack(logits, dim=1)


@torch.no_grad()
def greedy_decode(model: nn.Module, source: torch.Tensor, max_length: int = 100) -> torch.Tensor:
    model.eval()
    generated = torch.full((source.size(0), 1), BOS_ID, dtype=torch.long, device=source.device)
    finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)

    if isinstance(model, TransformerNMT):
        for _ in range(max_length - 1):
            next_token = model(source, generated)[:, -1].argmax(1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(EOS_ID)
            if finished.all():
                break
        return generated

    if isinstance(model, Seq2SeqNMT):
        encoder_outputs, states, source_mask = model.encode(source)
        context = encoder_outputs.new_zeros(source.size(0), model.hidden_size * 2)
        token = generated[:, 0]
        pieces = [token]
        for _ in range(max_length - 1):
            logits, states, context = model.decode_step(token, states, context, encoder_outputs, source_mask)
            token = logits.argmax(1)
            pieces.append(token)
            finished |= token.eq(EOS_ID)
            if finished.all():
                break
        return torch.stack(pieces, dim=1)

    raise TypeError(f"Không hỗ trợ model {type(model).__name__}")


def build_model(name: str, source_vocab_size: int, target_vocab_size: int,
                config: dict | None = None) -> nn.Module:
    config = config or {}
    if name == "transformer":
        return TransformerNMT(source_vocab_size, target_vocab_size, **config)
    if name == "seq2seq":
        return Seq2SeqNMT(source_vocab_size, target_vocab_size, **config)
    raise ValueError(f"Model không hợp lệ: {name}")

