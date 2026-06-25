"""
comparet_time.py
================

Compare the *inference time* of two defences on the **same inputs**, using the
**same Alpaca model**:

  1. ShapSelfDenoise (this repo / the paper)
        compute SHAP values  ->  mask the top-p% words (ONE masked copy)
        ->  denoise that copy  ->  classify it.
     The expensive part is the SHAP computation; everything after it is a
     single forward/denoise/classify.

  2. SelfDenoise (the original, github.com/UCSB-NLP-Chang/SelfDenoise)
        build `--num-copies` RANDOMLY masked copies of the input
        ->  denoise every copy  ->  classify every copy  ->  majority vote.
     There is no SHAP cost, but the number of denoise/classify passes is large
     (the original default `predict_ensemble` is 100).

So the comparison is essentially:
        (one SHAP computation + 1 denoise + 1 classify)
   vs   (num_copies denoises + num_copies classifies + a vote).

The random masking here mirrors the original `utils/mask.py::mask_sentence`
(round(length * rate) words masked, sampled uniformly without replacement,
keeping at least `min_keep` words), and the denoise / classify calls reuse the
exact generation settings already used elsewhere in this repo, so the two
pipelines differ only in *what this script is meant to measure*.

Example
-------
    python comparet_time.py --dataset sst2  --sample-size 20 --num-copies 100
    python comparet_time.py --dataset agnews --sample-size 20 --num-copies 100 --precision half

Notes
-----
* SHAP (captum's ShapleyValueSampling) is slow and scales with the sentence
  length, so keep --sample-size modest.
* `--mask-rate` mostly affects *accuracy*, not timing: the original's time is
  driven by --num-copies and ShapSelfDenoise's time is driven by the SHAP pass.
  The default 0.05 matches the mask rate used for Table 1 in the paper.
* Results are written to out/time_comparison/<dataset>/.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.model import Alpaca


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Compare inference time of ShapSelfDenoise vs the original "
        "SelfDenoise on the same inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default="config.yml",
                   help="Path to the YAML config (for the model id).")
    p.add_argument("--dataset", type=str, default="sst2",
                   choices=["agnews", "sst2"], help="Dataset to take inputs from.")
    p.add_argument("--data-file", type=str, default=None,
                   help="Override the input file. Defaults to "
                        "dataset/<dataset>/dataset_attack.json")
    p.add_argument("--precision", type=str, default="full",
                   choices=["full", "half"], help="full=float32, half=float16.")
    p.add_argument("--batchsize", type=int, default=16,
                   help="Batch size used to denoise/classify the ensemble copies.")
    p.add_argument("--mask_word", type=str, default="<mask>",
                   choices=["<mask>", "###"], help="Mask token.")
    p.add_argument("--sample-size", type=int, default=20,
                   help="Number of inputs to time both methods on.")
    p.add_argument("--num-copies", type=int, default=100,
                   help="Ensemble size for the ORIGINAL SelfDenoise "
                        "(the original default predict_ensemble is 100).")
    p.add_argument("--mask-rate", type=float, default=0.05,
                   help="ShapSelfDenoise mask rate: fraction of top-SHAP words "
                        "masked in the single copy (your paper's value).")
    p.add_argument("--selfdenoise-mask-rate", type=float, default=0.05,
                   help="SelfDenoise random mask rate per copy. The SelfDenoise "
                        "paper uses 5%% for ALL methods in the EMPIRICAL robustness "
                        "table (Table 1); the repo's 0.7 is a certification-only "
                        "default, not used here.")
    p.add_argument("--min-keep", type=int, default=2,
                   help="Minimum words to keep unmasked (matches the original).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--shap-n-samples", type=int, default=25,
                   help="captum ShapleyValueSampling permutations. Lower is "
                        "faster but noisier (25 = captum default).")
    return p.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_inputs(args):
    """Return (texts, labels) where labels are 0-based class indices."""
    data_file = args.data_file or f"dataset/{args.dataset}/dataset_attack.json"
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"{data_file} not found. Pass --data-file or check the dataset name."
        )
    df = pd.read_json(data_file, orient="records", lines=True)

    if args.dataset == "agnews":
        # labels are stored as strings "101".."104" -> 0..3
        labels = df["label"].astype(int) - 101
    else:  # sst2 labels are already 0/1
        labels = df["label"].astype(int)
    df = df.assign(label=labels)

    if args.sample_size < len(df):
        df = df.sample(args.sample_size, random_state=args.seed).reset_index(drop=True)
    return df["text"].tolist(), df["label"].tolist()


def now():
    """Wall-clock time, synchronising CUDA first so GPU work is actually done."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


# --------------------------------------------------------------------------- #
# Batched denoise / classify (reuse the repo's exact generation settings)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def denoise_batch(model, texts, progress=None):
    """Denoise a list of masked sentences, batched by model.batch_size.

    Mirrors Alpaca.denoise_sentence / denoise_instances exactly. If `progress`
    is a label string, show a transient per-batch bar (used for the big
    SelfDenoise ensemble); default None keeps the function silent."""
    tok = model.alpaca_tokenizer
    bs = max(1, model.batch_size)
    out = []
    chunks = range(0, len(texts), bs)
    if progress:
        chunks = tqdm(chunks, desc=progress, unit="batch", leave=False)
    for i in chunks:
        chunk = texts[i:i + bs]
        prompts = [
            model.template_without_input.format(model.denoise_instruction.format(t))
            for t in chunk
        ]
        inputs = tok(prompts, return_tensors="pt", padding=True)
        gen = model.ds_engine.generate(
            inputs.input_ids.to(model.alpaca_model.device),
            attention_mask=inputs.attention_mask.to(model.alpaca_model.device),
            bad_words_ids=[[529], [29966]],
            repetition_penalty=1.3,
            num_beams=2,
            max_new_tokens=80,
        )
        dec = tok.batch_decode(gen, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
        out.extend(o[len(p):] for o, p in zip(dec, prompts))
    return out


@torch.no_grad()
def classify_batch(model, texts, progress=None):
    """Classify a list of sentences, batched. Returns 0-based class indices.

    Mirrors Alpaca.classify_sentence (left padding -> last-token logits ->
    softmax over the label tokens -> argmax). `progress` works as in
    denoise_batch."""
    tok = model.alpaca_tokenizer
    bs = max(1, model.batch_size)
    preds = []
    chunks = range(0, len(texts), bs)
    if progress:
        chunks = tqdm(chunks, desc=progress, unit="batch", leave=False)
    for i in chunks:
        chunk = texts[i:i + bs]
        prompts = [model.template.format(model.instruction, t) for t in chunk]
        inputs = tok(prompts, return_tensors="pt", padding=True)
        outputs = model.ds_engine(
            inputs.input_ids.to(model.alpaca_model.device),
            attention_mask=inputs.attention_mask.to(model.alpaca_model.device),
        )
        dist = torch.softmax(outputs["logits"], dim=-1)[..., -1, :][:, model.label_token]
        preds.extend(torch.argmax(dist, dim=-1).cpu().tolist())
    return preds


# --------------------------------------------------------------------------- #
# The two pipelines
# --------------------------------------------------------------------------- #
def shapselfdenoise_predict(model, text, mask_rate):
    """SHAP-mask the top-p% words (one copy) -> denoise -> classify."""
    masked_text, _ = model.shap_masking(text, mask_rate, None)
    denoised = denoise_batch(model, [masked_text])[0]
    return classify_batch(model, [denoised])[0]


def random_masked_copies(text, rate, num_copies, mask_token, min_keep=2):
    """Build `num_copies` randomly masked versions of `text`.

    Faithful to the original SelfDenoise utils/mask.py::mask_sentence:
    mask round(length * rate) words, sampled uniformly without replacement,
    keeping at least `min_keep` words."""
    words = text.split()
    length = len(words)
    n_mask = round(length * rate)
    if length - n_mask < min_keep:
        n_mask = max(length - min_keep, 0)

    copies = []
    for _ in range(num_copies):
        idx = (np.random.choice(length, size=n_mask, replace=False)
               if n_mask > 0 else np.array([], dtype=int))
        masked = list(words)
        for j in idx:
            masked[j] = mask_token
        copies.append(" ".join(masked))
    return copies


def selfdenoise_predict(model, text, mask_rate, num_copies, min_keep=2, progress=False):
    """Original SelfDenoise: random-mask ensemble -> denoise each -> classify
    each -> majority vote. Set progress=True for a transient sub-bar over the
    (dominant) denoise step; tqdm overhead is sub-millisecond and does not
    meaningfully affect the timing."""
    copies = random_masked_copies(
        text, mask_rate, num_copies, mask_token=model.args.mask_word, min_keep=min_keep
    )
    denoised = denoise_batch(
        model, copies, progress=(f"  denoise x{num_copies}" if progress else None))
    preds = classify_batch(
        model, denoised, progress=("  classify" if progress else None))
    votes = np.bincount(np.asarray(preds, dtype=int), minlength=model.num_labels)
    return int(np.argmax(votes))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    np.random.seed(args.seed)

    config = load_config(args.config)
    texts, labels = load_inputs(args)
    if len(texts) == 0:
        raise RuntimeError("No inputs loaded.")

    # Build the model and configure it for the dataset.
    model = Alpaca(args, config["model"])
    if args.dataset == "agnews":
        model.as_agnews(args.mask_word)
    else:
        model.as_sst2()
    print(f"Model ready for {args.dataset} "
          f"({args.precision} precision, batch size {model.batch_size}).")
    print(f"Comparing on {len(texts)} inputs | "
          f"ShapSelfDenoise mask_rate={args.mask_rate} | "
          f"SelfDenoise mask_rate={args.selfdenoise_mask_rate} | "
          f"SelfDenoise num_copies={args.num_copies}\n")

    # ---- Warm-up (absorbs CUDA init / kernel compilation, not timed) ------- #
    # A tiny throwaway sentence triggers the one-time CUDA / cuBLAS / kernel
    # costs WITHOUT paying a full-length SHAP attribution (which can take
    # minutes). The first real input may re-autotune very slightly; negligible.
    print("Warming up ...")
    warm = "good movie"
    _ = shapselfdenoise_predict(model, warm, args.mask_rate)
    _ = selfdenoise_predict(model, warm, args.selfdenoise_mask_rate,
                            min(args.num_copies, model.batch_size), args.min_keep)

    # Re-seed so the timed random masks are reproducible regardless of warm-up.
    np.random.seed(args.seed)

    # ---- Timed loop -------------------------------------------------------- #
    rows = []
    header = (f"{'idx':>3} {'words':>5} {'shap(s)':>9} {'selfden(s)':>11} "
              f"{'speedup':>8} {'shap_pred':>9} {'self_pred':>9} {'label':>5}")
    tqdm.write("\n" + header)
    tqdm.write("-" * 70)
    bar = tqdm(enumerate(zip(texts, labels)), total=len(texts),
               desc="Comparing", unit="input")
    for i, (text, label) in bar:
        t0 = now()
        shap_pred = shapselfdenoise_predict(model, text, args.mask_rate)
        shap_t = now() - t0

        t0 = now()
        self_pred = selfdenoise_predict(model, text, args.selfdenoise_mask_rate,
                                        args.num_copies, args.min_keep, progress=True)
        self_t = now() - t0

        speedup = (self_t / shap_t) if shap_t > 0 else float("nan")
        rows.append(dict(idx=i, n_words=len(text.split()),
                         shap_time=shap_t, selfdenoise_time=self_t,
                         shap_pred=shap_pred, selfdenoise_pred=self_pred,
                         label=int(label)))
        bar.set_postfix({"shap": f"{shap_t:.2f}s", "self": f"{self_t:.2f}s",
                         "speedup": f"{speedup:.1f}x"})
        tqdm.write(f"{i:>3} {len(text.split()):>5} {shap_t:>9.3f} {self_t:>11.3f} "
                   f"{speedup:>7.2f}x {shap_pred:>9} {self_pred:>9} {int(label):>5}")
    bar.close()

    res = pd.DataFrame(rows)

    # ---- Summary ----------------------------------------------------------- #
    shap_total = res["shap_time"].sum()
    self_total = res["selfdenoise_time"].sum()
    shap_mean = res["shap_time"].mean()
    self_mean = res["selfdenoise_time"].mean()
    speedup_overall = self_total / shap_total if shap_total > 0 else float("nan")
    shap_acc = (res["shap_pred"] == res["label"]).mean()
    self_acc = (res["selfdenoise_pred"] == res["label"]).mean()

    print("\n" + "=" * 70)
    print(f"{'':37}{'ShapSelfDenoise':>17}{'SelfDenoise':>16}")
    print("-" * 70)
    print(f"{'total time (s)':37}{shap_total:>17.3f}{self_total:>16.3f}")
    print(f"{'mean time / input (s)':37}{shap_mean:>17.3f}{self_mean:>16.3f}")
    print(f"{'accuracy on these inputs':37}{shap_acc:>17.3f}{self_acc:>16.3f}")
    print("-" * 70)
    print(f"ShapSelfDenoise is {speedup_overall:.2f}x "
          f"{'faster' if speedup_overall >= 1 else 'slower'} than SelfDenoise "
          f"(num_copies={args.num_copies}).")
    print("=" * 70)

    # ---- Save -------------------------------------------------------------- #
    out_dir = f"out/time_comparison/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(os.path.join(out_dir, "per_input.csv"), index=False)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"dataset            : {args.dataset}\n")
        f.write(f"precision          : {args.precision}\n")
        f.write(f"batch size         : {model.batch_size}\n")
        f.write(f"ShapSelfDenoise mask rate : {args.mask_rate}\n")
        f.write(f"SelfDenoise     mask rate : {args.selfdenoise_mask_rate}\n")
        f.write(f"num inputs         : {len(res)}\n")
        f.write(f"SelfDenoise copies : {args.num_copies}\n\n")
        f.write(f"ShapSelfDenoise total time (s) : {shap_total:.3f}\n")
        f.write(f"SelfDenoise     total time (s) : {self_total:.3f}\n")
        f.write(f"ShapSelfDenoise mean/input (s) : {shap_mean:.3f}\n")
        f.write(f"SelfDenoise     mean/input (s) : {self_mean:.3f}\n")
        f.write(f"speedup (SelfDenoise / Shap)   : {speedup_overall:.2f}x\n")
        f.write(f"ShapSelfDenoise accuracy       : {shap_acc:.3f}\n")
        f.write(f"SelfDenoise     accuracy       : {self_acc:.3f}\n")
    print(f"\nSaved per-input timings to {out_dir}/per_input.csv")
    print(f"Saved summary to {out_dir}/summary.txt")


if __name__ == "__main__":
    main()