from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
)


def load_model(model_path: str):
    print(f"[INFO] Loading model: {model_path}")

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation="eager",
        device_map="auto",      # Automatically use multiple GPUs if available
        trust_remote_code=True,
    )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    ctx_size=512,
    eval_batch_size=4,
):
    print("[INFO] Loading WikiText-2 test set...")

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )

    tokens = tokenizer(
        "\n\n".join(dataset["text"]),
        return_tensors="pt",
    ).input_ids

    total_chunks = tokens.numel() // ctx_size
    total_tokens = total_chunks * ctx_size

    tokens = tokens[:, :total_tokens]

    chunks = tokens.view(total_chunks, ctx_size)

    print(
        f"\nperplexity: calculating perplexity over "
        f"{total_chunks} chunks, "
        f"n_ctx={ctx_size}, "
        f"batch_size={ctx_size * eval_batch_size}, "
        f"n_seq={eval_batch_size}\n"
    )

    total_nll = 0.0
    total_pred_tokens = 0

    start_time = time.time()

    total_batches = math.ceil(total_chunks / eval_batch_size)

    for batch_idx, start in enumerate(
        range(0, total_chunks, eval_batch_size), start=1
    ):

        end = min(start + eval_batch_size, total_chunks)

        batch = chunks[start:end].to(model.device)

        outputs = model(batch)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )

        pred_tokens = shift_labels.numel()

        total_nll += loss.item() * pred_tokens
        total_pred_tokens += pred_tokens

        running_ppl = math.exp(total_nll / total_pred_tokens)

        elapsed = time.time() - start_time

        sec_per_batch = elapsed / batch_idx
        eta = sec_per_batch * (total_batches - batch_idx)

        print(
            f"[{end}/{total_chunks}] "
            f"PPL={running_ppl:.4f} "
            f"({elapsed:.1f}s, ETA {eta/60:.2f} min)"
        )

    total_time = time.time() - start_time

    final_ppl = math.exp(total_nll / total_pred_tokens)

    tokens_per_sec = total_pred_tokens / total_time

    print("\n========================================")
    print(
        f"Final estimate: PPL over "
        f"{total_chunks} chunks "
        f"for n_ctx={ctx_size} = {final_ppl:.4f}"
    )
    print("========================================")
    print()

    print("Timing")
    print("----------------------------------------")
    print(f"Elapsed time      : {total_time:.2f} sec")
    print(f"Tokens evaluated  : {total_pred_tokens}")
    print(f"Throughput        : {tokens_per_sec:.2f} tokens/sec")
    print()

    return final_ppl


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Model directory or HuggingFace model",
    )

    parser.add_argument(
        "--ctx-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of sequences evaluated simultaneously.",
    )

    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    ppl = evaluate_perplexity(
        model=model,
        tokenizer=tokenizer,
        ctx_size=args.ctx_size,
        eval_batch_size=args.batch_size,
    )

    print(f"\nPerplexity = {ppl:.6f}")


if __name__ == "__main__":
    main()