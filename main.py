import json
import numpy as np

FILES = {
    "Qwen/Qwen3.6-27B": 
        "results/Qwen__Qwen3.6-27B/samples_wikitext_2026-08-12T11-28-47.551818.jsonl",

    "Qwen/Qwen3.6-27B-FP8":
        "results/Qwen__Qwen3.6-27B-FP8/samples_wikitext_2026-08-12T11-45-30.614353.jsonl",

    "models/Qwen3.6-27B-FP8-KVFP8-ATFP8":
        "results/models__Qwen3.6-27B-FP8-KVFP8-ATFP8/samples_wikitext_2026-08-12T12-17-30.396761.jsonl",

    "models/Qwen3.6-27B-FP8-KVFP8":
        "results/models__Qwen3.6-27B-FP8-KVFP8/samples_wikitext_2026-08-12T13-12-16.208392.jsonl",

    "models/Qwen3.6-27B-FP8":
        "results/models__Qwen3.6-27B-FP8/samples_wikitext_2026-08-12T12-00-00.365934.jsonl",

    "models/Qwen3.6-27B-MXFP4-KVFP8":
        "results/models__Qwen3.6-27B-MXFP4-KVFP8/samples_wikitext_2026-08-12T12-47-06.770191.jsonl",

    "models/Qwen3.6-27B-MXFP4":
        "results/models__Qwen3.6-27B-MXFP4/samples_wikitext_2026-08-12T12-33-02.972536.jsonl",
}


N_BOOTSTRAPS = 1000


def calculate_ppl_ci(filepath, n_bootstraps=1000, seed=67):
    """Calculate PPL and percentile bootstrap 95% CI."""

    with open(filepath, "r") as f:
        samples = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    print(f"Loaded {len(samples)} samples")

    log_likelihoods = np.array([
        float(doc["word_perplexity"][0])
        for doc in samples
    ])

    word_counts = np.array([
        float(doc["word_perplexity"][1])
        for doc in samples
    ])

    total_log_likelihood = np.sum(log_likelihoods)
    total_words = np.sum(word_counts)

    ppl = np.exp(
        -total_log_likelihood / total_words
    )

    rng = np.random.default_rng(seed)

    n_samples = len(samples)

    bootstrap_ppl = np.empty(n_bootstraps)

    for i in range(n_bootstraps):

        indices = rng.integers(
            0,
            n_samples,
            size=n_samples
        )

        sampled_ll = log_likelihoods[indices]
        sampled_words = word_counts[indices]

        bootstrap_ppl[i] = np.exp(
            -np.sum(sampled_ll) / np.sum(sampled_words)
        )

    ci_lower = np.percentile(
        bootstrap_ppl,
        2.5
    )

    ci_upper = np.percentile(
        bootstrap_ppl,
        97.5
    )

    return {
        "n_samples": n_samples,
        "total_log_likelihood": total_log_likelihood,
        "total_words": total_words,
        "ppl": ppl,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_ppl": bootstrap_ppl,
    }

results = {}

for model_name, filepath in FILES.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    result = calculate_ppl_ci(
        filepath,
        n_bootstraps=N_BOOTSTRAPS,
        seed=67
    )

    results[model_name] = result

    print(f"Samples:                {result['n_samples']}")
    print(f"Total log-likelihood:   {result['total_log_likelihood']:.4f}")
    print(f"Total words:            {result['total_words']:.0f}")
    print(f"PPL:                    {result['ppl']:.6f}")
    print(
        f"95% bootstrap CI:       "
        f"[{result['ci_lower']:.6f}, "
        f"{result['ci_upper']:.6f}]"
    )
    print(f"Bootstrap samples:      {N_BOOTSTRAPS}")


print("\n\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(
    f"{'Model':<40}"
    f"{'PPL':>12}"
    f"{'95% CI':>30}"
)

print("-" * 90)

for model_name, result in results.items():

    ci = (
        f"[{result['ci_lower']:.4f}, "
        f"{result['ci_upper']:.4f}]"
    )

    print(
        f"{model_name:<40}"
        f"{result['ppl']:>12.4f}"
        f"{ci:>30}"
    )