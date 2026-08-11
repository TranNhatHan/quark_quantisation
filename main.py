import json
import numpy as np

# FILE = "results/Qwen3.6-27B-fp8/samples_wikitext_2026-08-11T05-16-22.395581.jsonl"
FILE = "results/Qwen__Qwen3.6-27B/samples_wikitext_2026-08-11T05-32-36.304517.jsonl"

with open(FILE, "r") as f:
    samples = [json.loads(line) for line in f if line.strip()]

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

ppl = np.exp(-total_log_likelihood / total_words)

print(f"Total log-likelihood: {total_log_likelihood:.4f}")
print(f"Total words:          {total_words:.0f}")
print(f"Perplexity:           {ppl:.6f}")

n_bootstraps = 1000

rng = np.random.default_rng(42)

bootstrap_ppl = np.empty(n_bootstraps)

n_samples = len(samples)

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

ci_lower = np.percentile(bootstrap_ppl, 2.5)
ci_upper = np.percentile(bootstrap_ppl, 97.5)


print()
print(f"PPL:                    {ppl:.6f}")
print(f"95% bootstrap CI:       [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"Bootstrap samples:      {n_bootstraps}")