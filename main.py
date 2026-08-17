import json
import numpy as np

FILES = {
    "Qwen/Qwen3.6-27B":
        "results/Qwen__Qwen3.6-27B/samples_wikitext_2026-08-13T15-44-04.085646.jsonl",

    "Qwen/Qwen3.6-27B-FP8":
        "results/Qwen__Qwen3.6-27B-FP8/samples_wikitext_2026-08-13T15-58-27.358353.jsonl",

    "models/Qwen3.6-27B-FP8-KVFP8-ATFP8":
        "results/models__Qwen3.6-27B-FP8-KVFP8-ATFP8/samples_wikitext_2026-08-13T16-33-30.380889.jsonl",

    "models/Qwen3.6-27B-FP8-KVFP8":
        "results/models__Qwen3.6-27B-FP8-KVFP8/samples_wikitext_2026-08-13T16-21-52.541479.jsonl",

    "models/Qwen3.6-27B-FP8":
        "results/models__Qwen3.6-27B-FP8/samples_wikitext_2026-08-13T16-10-24.355671.jsonl",

    "models/Qwen3.6-27B-MXFP4":
        "results/models__Qwen3.6-27B-MXFP4/samples_wikitext_2026-08-13T16-45-54.805180.jsonl",
}


N_BOOTSTRAPS = 1000
SEED = 67

ORIGINAL_MODEL = "Qwen/Qwen3.6-27B"


def load_samples(filepath):
    """Load WikiText lm_eval samples."""

    with open(filepath, "r") as f:
        samples = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    log_likelihoods = np.array([
        float(doc["word_perplexity"][0])
        for doc in samples
    ])

    word_counts = np.array([
        float(doc["word_perplexity"][1])
        for doc in samples
    ])

    return {
        "samples": samples,
        "log_likelihoods": log_likelihoods,
        "word_counts": word_counts,
    }


def calculate_ppl(log_likelihoods, word_counts):
    """Calculate corpus-level perplexity."""

    return np.expexp(-np.sum(log_likelihoods) / np.sum(word_counts))

data = {}

for model_name, filepath in FILES.items():

    model_data = load_samples(filepath)

    data[model_name] = model_data

    print(
        f"{model_name:<45} "
        f"Loaded {len(model_data['log_likelihoods'])} samples"
    )

original_n = len(data[ORIGINAL_MODEL]["log_likelihoods"])

for model_name, model_data in data.items():
    if len(model_data["log_likelihoods"]) != original_n:
        raise ValueError(
            f"Sample count mismatch:\n"
            f"Original: {original_n}\n"
            f"{model_name}: "
            f"{len(model_data['log_likelihoods'])}"
        )

rng = np.random.default_rng(SEED)

n_samples = original_n

# Store bootstrap PPL for every model
bootstrap_ppl = {
    model_name: np.empty(N_BOOTSTRAPS)
    for model_name in FILES
}

# Store bootstrap delta PPL
bootstrap_delta = {
    model_name: np.empty(N_BOOTSTRAPS)
    for model_name in FILES
    if model_name != ORIGINAL_MODEL
}

results = {}

for model_name, model_data in data.items():

    ll = model_data["log_likelihoods"]
    words = model_data["word_counts"]

    ppl = calculate_ppl(ll, words)

    results[model_name] = {
        "n_samples": n_samples,
        "total_log_likelihood": np.sum(ll),
        "total_words": np.sum(words),
        "ppl": ppl,
    }


# ------------------------------------------------------------
# Paired bootstrap
# ------------------------------------------------------------

original_ll = data[ORIGINAL_MODEL]["log_likelihoods"]
original_words = data[ORIGINAL_MODEL]["word_counts"]


for i in range(N_BOOTSTRAPS):

    # IMPORTANT:
    # Same indices are used for ORIGINAL and every quantized model.
    indices = rng.integers(
        0,
        n_samples,
        size=n_samples
    )

    # --------------------------------------------------------
    # Original model bootstrap PPL
    # --------------------------------------------------------

    original_sampled_ll = original_ll[indices]
    original_sampled_words = original_words[indices]

    original_bootstrap_ppl = calculate_ppl(
        original_sampled_ll,
        original_sampled_words
    )

    bootstrap_ppl[ORIGINAL_MODEL][i] = original_bootstrap_ppl

    # --------------------------------------------------------
    # Quantized models
    # --------------------------------------------------------

    for model_name, model_data in data.items():

        if model_name == ORIGINAL_MODEL:
            continue

        sampled_ll = model_data["log_likelihoods"][indices]
        sampled_words = model_data["word_counts"][indices]

        model_bootstrap_ppl = calculate_ppl(
            sampled_ll,
            sampled_words
        )

        bootstrap_ppl[model_name][i] = model_bootstrap_ppl

        # Paired bootstrap delta
        bootstrap_delta[model_name][i] = (
            model_bootstrap_ppl
            - original_bootstrap_ppl
        )


# ============================================================
# Calculate individual PPL confidence intervals
# ============================================================

for model_name in FILES:

    bootstrap_values = bootstrap_ppl[model_name]

    results[model_name]["ci_lower"] = np.percentile(
        bootstrap_values,
        2.5
    )

    results[model_name]["ci_upper"] = np.percentile(
        bootstrap_values,
        97.5
    )


# ============================================================
# Calculate Delta PPL confidence intervals
# ============================================================

original_ppl = results[ORIGINAL_MODEL]["ppl"]

for model_name in FILES:

    if model_name == ORIGINAL_MODEL:
        continue

    quantized_ppl = results[model_name]["ppl"]

    # Point estimate
    delta_ppl = quantized_ppl - original_ppl

    # 95% paired bootstrap CI
    delta_bootstrap = bootstrap_delta[model_name]

    delta_ci_lower = np.percentile(
        delta_bootstrap,
        2.5
    )

    delta_ci_upper = np.percentile(
        delta_bootstrap,
        97.5
    )

    results[model_name]["delta_ppl"] = delta_ppl
    results[model_name]["delta_ci_lower"] = delta_ci_lower
    results[model_name]["delta_ci_upper"] = delta_ci_upper


# ============================================================
# Print individual results
# ============================================================

print("\n\n" + "=" * 100)
print("PPL RESULTS")
print("=" * 100)

for model_name, result in results.items():

    print("\n" + "-" * 100)
    print(model_name)
    print("-" * 100)

    print(
        f"Samples:                "
        f"{result['n_samples']}"
    )

    print(
        f"Total log-likelihood:   "
        f"{result['total_log_likelihood']:.4f}"
    )

    print(
        f"Total words:            "
        f"{result['total_words']:.0f}"
    )

    print(
        f"PPL:                    "
        f"{result['ppl']:.6f}"
    )

    print(
        f"95% bootstrap CI:       "
        f"[{result['ci_lower']:.6f}, "
        f"{result['ci_upper']:.6f}]"
    )

    if model_name != ORIGINAL_MODEL:

        print(
            f"Delta PPL:              "
            f"{result['delta_ppl']:+.6f}"
        )

        print(
            f"95% Delta PPL CI:       "
            f"[{result['delta_ci_lower']:+.6f}, "
            f"{result['delta_ci_upper']:+.6f}]"
        )


# ============================================================
# Summary table
# ============================================================

print("\n\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)

print(
    f"{'Model':<45}"
    f"{'PPL':>12}"
    f"{'95% PPL CI':>28}"
    f"{'Delta PPL':>14}"
    f"{'95% Delta CI':>30}"
)

print("-" * 120)

for model_name, result in results.items():

    ppl_ci = (
        f"[{result['ci_lower']:.4f}, "
        f"{result['ci_upper']:.4f}]"
    )

    if model_name == ORIGINAL_MODEL:

        delta = "reference"
        delta_ci = "reference"

    else:

        delta = f"{result['delta_ppl']:+.4f}"

        delta_ci = (
            f"[{result['delta_ci_lower']:+.4f}, "
            f"{result['delta_ci_upper']:+.4f}]"
        )

    print(
        f"{model_name:<45}"
        f"{result['ppl']:>12.4f}"
        f"{ppl_ci:>28}"
        f"{delta:>14}"
        f"{delta_ci:>30}"
    )