#!/usr/bin/env python3
"""
collect_figure_data.py
======================
Runs the experiments whose results `make_figures.py` turns into figures, and
writes them as tidy CSV/JSON under out/figures_data/. This must run where the
Alpaca model loads (GPU + model weights); it reuses the exact, already-tested
pipeline functions from comparet_time.py so the numbers here are consistent
with the timing script.

TASKS
-----
  ensemble      Sweep the ORIGINAL SelfDenoise ensemble size (--copies-grid) and,
                once, ShapSelfDenoise. Per (dataset, method, num_copies, seed):
                mean inference time/example + accuracy.
                -> out/figures_data/ensemble_sweep.csv
                Feeds figures: pareto, time_vs_ensemble, acc_vs_ensemble.
                COST ~ n_inputs * n_seeds * sum(copies_grid) denoise passes,
                plus n_inputs SHAP computations. Keep --sample-size modest.

  selection     Same budget (ONE masked copy), vary only HOW words are chosen:
                SHAP vs LIME vs random. SHAP/LIME are deterministic (computed
                once); random is repeated over seeds.
                -> out/figures_data/selection_ablation.csv   (feeds: ablation)

  shap_samples  Vary captum's ShapleyValueSampling n_samples (--samples-grid);
                record accuracy + time at each.
                -> out/figures_data/shap_samples_sweep.csv   (feeds: shap_samples)

  examples      Dump SHAP values + which words get masked + the prediction
                before/after the defence, for a few sentences.
                -> out/figures_data/shap_examples.json        (feeds: heatmap)

  all           Run every task above.

USAGE
-----
  python figures/collect_figure_data.py --task ensemble --dataset sst2  --sample-size 20 --seeds 3
  python figures/collect_figure_data.py --task ensemble --dataset agnews --precision half
  python figures/collect_figure_data.py --task selection --dataset sst2
  python figures/collect_figure_data.py --task examples  --dataset sst2 --num-examples 3
  python figures/collect_figure_data.py --task all --dataset sst2
"""

import argparse
import gc
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

# Reuse the model and the *exact* pipeline already validated in comparet_time.py.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from src.model import Alpaca, add_placeholders, mask_words  # noqa: E402
from comparet_time import (  # noqa: E402
    classify_batch, denoise_batch, load_inputs, now,
    random_masked_copies, selfdenoise_predict, shapselfdenoise_predict,
)
from captum.attr import (  # noqa: E402
    LLMAttribution, ShapleyValueSampling, TextTemplateInput,
)

OUT_DIR = os.path.join(REPO_ROOT, "out", "figures_data")

# Human-readable class names (idx is 0-based, matching comparet_time's labels).
CLASS_NAMES = {
    "sst2":   {0: "negative", 1: "positive"},
    "agnews": {0: "World", 1: "Sports", 2: "Business", 3: "Technology"},
}


def free_gpu():
    """Reclaim CUDA memory. Call right AFTER `del model` so the previous
    dataset's ~14 GB model is released before the next one loads; otherwise two
    models pile up on a single card and `--dataset both` hits CUDA OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def build_args(a, dataset):
    """A namespace with everything Alpaca + load_inputs read."""
    return SimpleNamespace(
        config=a.config, dataset=dataset, data_file=a.data_file,
        precision=a.precision, batchsize=a.batchsize, mask_word=a.mask_word,
        sample_size=a.sample_size, seed=a.seed,
    )


def setup(a, dataset):
    config = yaml.safe_load(open(a.config))
    args = build_args(a, dataset)
    model = Alpaca(args, config["model"])
    if dataset == "agnews":
        model.as_agnews(a.mask_word)
    else:
        model.as_sst2()
    texts, labels = load_inputs(args)
    print(f"[{dataset}] {len(texts)} inputs | precision={a.precision} | "
          f"batch={model.batch_size}")
    return model, texts, labels


def predict_one_copy(model, masked_text):
    """Denoise a single masked sentence then classify it (0-based class idx)."""
    return classify_batch(model, [denoise_batch(model, [masked_text])[0]])[0]


def save_csv(df, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  -> {path}  ({len(df)} rows)")


# --------------------------------------------------------------------------- #
# Task: ensemble sweep
# --------------------------------------------------------------------------- #
def task_ensemble(a):
    grid = [int(x) for x in a.copies_grid.split(",")]
    rows = []
    for ds in a.datasets:
        model, texts, labels = setup(a, ds)
        labels = np.asarray(labels)

        # Warm-up (absorbs CUDA init / kernel compile; not timed). Each method
        # warms up at the rate it will actually use.
        print("  warming up ...")
        _ = shapselfdenoise_predict(model, texts[0], a.mask_rate)
        _ = selfdenoise_predict(model, texts[0], a.selfdenoise_mask_rate,
                                min(grid), a.min_keep)

        # ShapSelfDenoise: deterministic, so measure once (its own mask rate).
        np.random.seed(a.seed)
        preds, times = [], []
        for t in texts:
            t0 = now(); p = shapselfdenoise_predict(model, t, a.mask_rate)
            times.append(now() - t0); preds.append(p)
        rows.append(dict(dataset=ds, method="ShapSelfDenoise", num_copies=1,
                         seed=a.seed, n_examples=len(texts), mask_rate=a.mask_rate,
                         mean_time_per_example=float(np.mean(times)),
                         accuracy=float((np.asarray(preds) == labels).mean())))
        print(f"  ShapSelfDenoise (rate={a.mask_rate}): acc={rows[-1]['accuracy']:.3f} "
              f"time={rows[-1]['mean_time_per_example']:.3f}s")

        # SelfDenoise: every ensemble size, every seed (its own random mask rate).
        for seed in range(1, a.seeds + 1):
            for nc in grid:
                np.random.seed(1000 * seed + nc)   # reproducible per (seed, nc)
                preds, times = [], []
                for t in texts:
                    t0 = now()
                    p = selfdenoise_predict(model, t, a.selfdenoise_mask_rate,
                                            nc, a.min_keep)
                    times.append(now() - t0); preds.append(p)
                rows.append(dict(dataset=ds, method="SelfDenoise", num_copies=nc,
                                 seed=seed, n_examples=len(texts),
                                 mask_rate=a.selfdenoise_mask_rate,
                                 mean_time_per_example=float(np.mean(times)),
                                 accuracy=float((np.asarray(preds) == labels).mean())))
                print(f"  SelfDenoise[seed={seed}, copies={nc}, rate={a.selfdenoise_mask_rate}]: "
                      f"acc={rows[-1]['accuracy']:.3f} "
                      f"time={rows[-1]['mean_time_per_example']:.3f}s")
        del model              # release before the next dataset loads (avoids OOM)
        free_gpu()
    save_csv(pd.DataFrame(rows), "ensemble_sweep.csv")


# --------------------------------------------------------------------------- #
# Task: selection-strategy ablation (equal budget = one masked copy)
# --------------------------------------------------------------------------- #
def task_selection(a):
    rows = []
    for ds in a.datasets:
        model, texts, labels = setup(a, ds)
        labels = np.asarray(labels)
        _ = predict_one_copy(model, texts[0])  # warm-up

        def run(masker):
            preds, times = [], []
            for t in texts:
                t0 = now()
                masked = masker(t)
                p = predict_one_copy(model, masked)
                times.append(now() - t0); preds.append(p)
            return float((np.asarray(preds) == labels).mean()), float(np.mean(times))

        # SHAP and LIME are deterministic -> measure once (seed = base).
        for strat, masker in [
            ("shap", lambda t: model.shap_masking(t, a.mask_rate, None)[0]),
            ("lime", lambda t: model.lime_masking(t, a.mask_rate, None)[0]),
        ]:
            acc, tm = run(masker)
            rows.append(dict(dataset=ds, strategy=strat, seed=a.seed,
                             n_examples=len(texts), mean_time_per_example=tm,
                             accuracy=acc))
            print(f"  {strat}: acc={acc:.3f} time={tm:.3f}s")

        # Random masking varies with the seed.
        for seed in range(1, a.seeds + 1):
            np.random.seed(seed)
            acc, tm = run(lambda t: random_masked_copies(
                t, a.mask_rate, 1, model.args.mask_word, a.min_keep)[0])
            rows.append(dict(dataset=ds, strategy="random", seed=seed,
                             n_examples=len(texts), mean_time_per_example=tm,
                             accuracy=acc))
            print(f"  random[seed={seed}]: acc={acc:.3f} time={tm:.3f}s")
        del model
        free_gpu()
    save_csv(pd.DataFrame(rows), "selection_ablation.csv")


# --------------------------------------------------------------------------- #
# Task: SHAP n_samples sweep
# --------------------------------------------------------------------------- #
def _shap_values_n(model, text, n_samples):
    """Replicate Alpaca.shap_masking's attribution with a configurable n_samples
    (the repo hard-codes captum's default), returning per-word SHAP values.

    SST-2 only: its prompt is short and reproduced verbatim below. AG News uses a
    long few-shot prompt, so sweep n_samples for it through the model directly
    rather than here. NOTE: if you edit the SST-2 prompt in model.py, mirror it
    here too, or this will drift from what shap_masking actually does."""
    if model.dataset != "sst2":
        raise NotImplementedError("n_samples sweep is wired for SST-2 only")
    import torch
    prompt = ("\nBelow is an instruction that describes a task, paired with an input "
              "that provides further context. Write a response that appropriately "
              "completes the request.\n\n### Instruction:\n\nGiven an English sentence "
              "input, determine its sentiment as positive or negative.\n\n### Input:\n")
    suffix = "\n\n### Response:\n"
    model_input = model.alpaca_tokenizer(prompt + text + suffix,
                                          return_tensors="pt").to("cuda")
    with torch.no_grad():
        out_ids = model.alpaca_model.generate(model_input["input_ids"], max_new_tokens=1)[0]
        category = model.alpaca_tokenizer.decode(out_ids, skip_special_tokens=True).split()[-1]
    fa = ShapleyValueSampling(model.alpaca_model)
    llm_attr = LLMAttribution(fa, model.alpaca_tokenizer)
    eval_prompt, values_to_add = add_placeholders(prompt, text, suffix)
    inp = TextTemplateInput(template=eval_prompt, values=values_to_add)
    attr_res = llm_attr.attribute(inp, target=category, n_samples=n_samples)
    return attr_res.token_attr.cpu().numpy().tolist()[0]


def task_shap_samples(a):
    grid = [int(x) for x in a.samples_grid.split(",")]
    rows = []
    for ds in a.datasets:
        if ds == "agnews":
            print("  [note] n_samples sweep is set up for SST-2; skipping agnews "
                  "(its few-shot prompt should be swept via the model directly).")
            continue
        model, texts, labels = setup(a, ds)
        labels = np.asarray(labels)
        for seed in range(1, a.seeds + 1):
            np.random.seed(seed)
            for ns in grid:
                preds, times = [], []
                for t in texts:
                    t0 = now()
                    try:
                        vals = _shap_values_n(model, t, ns)
                        masked = mask_words(t, vals, a.mask_rate, model.args.mask_word)
                        p = predict_one_copy(model, masked)
                    except Exception as e:
                        print(f"    [warn] n_samples={ns}: {type(e).__name__}: {e}")
                        p = -1
                    times.append(now() - t0); preds.append(p)
                rows.append(dict(dataset=ds, n_samples=ns, seed=seed,
                                 n_examples=len(texts),
                                 mean_time_per_example=float(np.mean(times)),
                                 accuracy=float((np.asarray(preds) == labels).mean())))
                print(f"  n_samples={ns}[seed={seed}]: acc={rows[-1]['accuracy']:.3f} "
                      f"time={rows[-1]['mean_time_per_example']:.3f}s")
        del model
        free_gpu()
    if rows:
        save_csv(pd.DataFrame(rows), "shap_samples_sweep.csv")


# --------------------------------------------------------------------------- #
# Task: SHAP heatmap examples
# --------------------------------------------------------------------------- #
def task_examples(a):
    examples = []
    for ds in a.datasets:
        model, texts, labels = setup(a, ds)
        names = CLASS_NAMES.get(ds, {})
        k = min(a.num_examples, len(texts))
        for t, lab in list(zip(texts, labels))[:k]:
            words = t.split()
            masked_text, shap_values = model.shap_masking(t, a.mask_rate, None)
            # Which indices get masked = top-(mask_rate) by SHAP value (same rule
            # as Alpaca.mask_words).
            num_to_replace = int(len(words) * a.mask_rate)
            order = sorted(range(len(shap_values)), key=lambda i: shap_values[i],
                           reverse=True)
            masked_indices = sorted(order[:num_to_replace])
            pred_before = classify_batch(model, [t])[0]              # undefended
            pred_after = shapselfdenoise_predict(model, t, a.mask_rate)  # defended
            examples.append(dict(
                dataset=ds, words=words,
                shap_values=[float(v) for v in shap_values],
                masked_indices=masked_indices,
                pred_before=names.get(int(pred_before), int(pred_before)),
                pred_after=names.get(int(pred_after), int(pred_after)),
                label=names.get(int(lab), int(lab)),
            ))
            print(f"  [{ds}] '{t[:50]}...' {examples[-1]['pred_before']} -> "
                  f"{examples[-1]['pred_after']} (label {examples[-1]['label']})")
        del model
        free_gpu()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "shap_examples.json")
    json.dump(examples, open(path, "w"), indent=2)
    print(f"  -> {path}  ({len(examples)} examples)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True,
                   choices=["ensemble", "selection", "shap_samples", "examples", "all"])
    p.add_argument("--dataset", default="sst2",
                   help="Dataset, or 'both' for sst2+agnews.")
    p.add_argument("--config", default=os.path.join(REPO_ROOT, "config.yml"))
    p.add_argument("--data-file", default=None,
                   help="Override input file (default dataset/<ds>/dataset_attack.json).")
    p.add_argument("--precision", default="full", choices=["full", "half"])
    p.add_argument("--batchsize", type=int, default=16)
    p.add_argument("--mask_word", default="<mask>", choices=["<mask>", "###"])
    p.add_argument("--sample-size", type=int, default=20)
    p.add_argument("--seeds", type=int, default=3,
                   help="Number of seeds for the stochastic (random-masking) parts.")
    p.add_argument("--seed", type=int, default=42, help="Base seed (input sampling).")
    p.add_argument("--mask-rate", type=float, default=0.05,
                   help="ShapSelfDenoise mask rate (top-SHAP fraction, single copy).")
    p.add_argument("--selfdenoise-mask-rate", type=float, default=0.05,
                   help="SelfDenoise random mask rate per copy. 5%% matches the "
                        "SelfDenoise paper's empirical experiments (Table 1; they "
                        "use 5%% for all methods). Used only by --task ensemble.")
    p.add_argument("--min-keep", type=int, default=2)
    p.add_argument("--copies-grid", default="1,5,10,25,50,100",
                   help="Ensemble sizes to sweep for SelfDenoise.")
    p.add_argument("--samples-grid", default="5,10,25,50,100",
                   help="captum n_samples values to sweep.")
    p.add_argument("--num-examples", type=int, default=3,
                   help="How many sentences to dump for the heatmap.")
    a = p.parse_args()
    a.datasets = ["sst2", "agnews"] if a.dataset == "both" else [a.dataset]
    return a


def main():
    a = parse_args()
    tasks = {"ensemble": task_ensemble, "selection": task_selection,
             "shap_samples": task_shap_samples, "examples": task_examples}
    to_run = list(tasks) if a.task == "all" else [a.task]
    for t in to_run:
        print(f"\n=== task: {t} ===")
        tasks[t](a)
    print("\nDone. Now run:  python figures/make_figures.py")


if __name__ == "__main__":
    main()