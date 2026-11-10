from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import Vocabulary, make_loader
from .models import build_model, greedy_decode


def load_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    source_vocab = Vocabulary(checkpoint["source_vocab"])
    target_vocab = Vocabulary(checkpoint["target_vocab"])
    model = build_model(checkpoint["model_name"], len(source_vocab), len(target_vocab),
                        checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, source_vocab, target_vocab, checkpoint["model_name"]


@torch.no_grad()
def measure(model, loader, target_vocab, device, max_batches: int = 0):
    from sacrebleu import corpus_bleu

    hypotheses, references, examples = [], [], []
    started = time.perf_counter()
    sentence_count = 0
    example_offset = 0
    for batch_index, (source, target) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        prediction = greedy_decode(model, source.to(device), max_length=min(target.size(1) + 20, 100)).cpu()
        raw_batch = loader.dataset.raw_examples[example_offset:example_offset + source.size(0)]
        for (source_text, target_text), predicted_ids in zip(raw_batch, prediction):
            hypothesis = target_vocab.decode(predicted_ids.tolist())
            reference = target_text
            hypotheses.append(hypothesis)
            references.append(reference)
            if len(examples) < 10:
                examples.append({
                    "source": source_text,
                    "reference": reference,
                    "prediction": hypothesis,
                })
        sentence_count += source.size(0)
        example_offset += source.size(0)
    elapsed = time.perf_counter() - started
    return {
        "bleu": round(corpus_bleu(hypotheses, [references], tokenize="none").score, 2),
        "sentences": sentence_count,
        "sentences_per_second": round(sentence_count / max(elapsed, 1e-9), 2),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh hai checkpoint trên cùng tst2013")
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--seq2seq", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="outputs/comparison.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {"device": str(device), "split": "tst2013", "models": {}}
    expected_vocabs = None
    for path in (args.transformer, args.seq2seq):
        model, source_vocab, target_vocab, name = load_checkpoint(path, device)
        vocab_signature = (source_vocab.itos, target_vocab.itos)
        if expected_vocabs is not None and vocab_signature != expected_vocabs:
            raise ValueError("Hai checkpoint không dùng cùng vocabulary, không thể so sánh công bằng.")
        expected_vocabs = vocab_signature
        loader = make_loader(args.data_dir, "test", source_vocab, target_vocab,
                             args.batch_size, args.max_length)
        result["models"][name] = measure(
            model, loader, target_vocab, device, args.max_eval_batches
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Comparison saved to {output}.")


if __name__ == "__main__":
    main()
