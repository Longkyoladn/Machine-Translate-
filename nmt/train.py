from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from .data import PAD_ID, Vocabulary, make_loader, prepare_data
from .models import build_model, greedy_decode


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config(name: str, small: bool = False) -> dict:
    if name == "transformer":
        if small:
            return {"d_model": 128, "num_heads": 4, "num_layers": 2, "ff_dim": 512, "dropout": 0.2}
        return {"d_model": 256, "num_heads": 4, "num_layers": 3, "ff_dim": 1024, "dropout": 0.2}
    if small:
        return {"hidden_size": 128, "num_layers": 1, "dropout": 0.2}
    return {"hidden_size": 256, "num_layers": 2, "dropout": 0.2}


def run_epoch(model: nn.Module, loader, device: torch.device,
              optimizer: torch.optim.Optimizer | None, max_batches: int = 0) -> float:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(loader, desc="train" if training else "valid", leave=False)
    for batch_index, (source, target) in enumerate(iterator):
        if max_batches and batch_index >= max_batches:
            break
        source, target = source.to(device), target.to(device)
        target_input, expected = target[:, :-1], target[:, 1:]
        with torch.set_grad_enabled(training):
            logits = model(source, target_input)
            loss_sum = criterion(logits.reshape(-1, logits.size(-1)), expected.reshape(-1))
            token_count = expected.ne(PAD_ID).sum()
            loss = loss_sum / token_count.clamp_min(1)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss_sum.item()
        total_tokens += token_count.item()
        iterator.set_postfix(loss=f"{total_loss / max(total_tokens, 1):.3f}")
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_bleu(model, loader, target_vocab: Vocabulary, device: torch.device,
                  max_batches: int = 0) -> float:
    try:
        from sacrebleu import corpus_bleu
    except ImportError as error:
        raise RuntimeError("Hãy chạy: pip install -r requirements.txt") from error
    hypotheses: list[str] = []
    references: list[str] = []
    example_offset = 0
    for batch_index, (source, target) in enumerate(tqdm(loader, desc="BLEU", leave=False)):
        if max_batches and batch_index >= max_batches:
            break
        prediction = greedy_decode(model, source.to(device), max_length=min(target.size(1) + 20, 100)).cpu()
        hypotheses.extend(target_vocab.decode(row.tolist()) for row in prediction)
        batch_size = source.size(0)
        references.extend(
            target_text for _, target_text in loader.dataset.raw_examples[example_offset:example_offset + batch_size]
        )
        example_offset += batch_size
    return corpus_bleu(hypotheses, [references], tokenize="none").score if hypotheses else 0.0


def save_checkpoint(path: Path, model, model_name: str, config: dict,
                    source_vocab, target_vocab, args) -> None:
    torch.save({
        "model_name": model_name,
        "model_config": config,
        "model_state": model.state_dict(),
        "source_vocab": source_vocab.itos,
        "target_vocab": target_vocab.itos,
        "args": vars(args),
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NMT English → Vietnamese")
    parser.add_argument("--model", choices=["transformer", "seq2seq"], required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=15000)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--small", action="store_true", help="Model nhỏ để smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_vocab, target_vocab = prepare_data(args.data_dir, args.vocab_size)
    train_loader = make_loader(args.data_dir, "train", source_vocab, target_vocab,
                               args.batch_size, args.max_length, shuffle=True)
    dev_loader = make_loader(args.data_dir, "dev", source_vocab, target_vocab,
                             args.batch_size, args.max_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = model_config(args.model, args.small)
    model = build_model(args.model, len(source_vocab), len(target_vocab), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model={args.model} | device={device} | parameters={parameter_count:,}")

    best_loss = math.inf
    history_path = output_dir / "history.jsonl"
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer, args.max_train_batches)
        valid_loss = run_epoch(model, dev_loader, device, None, args.max_eval_batches)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_perplexity": math.exp(min(valid_loss, 20)),
            "seconds": round(time.time() - started, 1),
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))
        save_checkpoint(output_dir / "last.pt", model, args.model, config, source_vocab, target_vocab, args)
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_checkpoint(output_dir / "best.pt", model, args.model, config, source_vocab, target_vocab, args)

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    bleu = evaluate_bleu(model, dev_loader, target_vocab, device, args.max_eval_batches)
    summary = {"model": args.model, "dev_bleu": round(bleu, 2), "best_valid_loss": best_loss}
    (output_dir / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Validation result: BLEU={bleu:.2f}")


if __name__ == "__main__":
    main()
