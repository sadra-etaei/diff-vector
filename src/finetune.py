"""Phase 1.1 — LoRA finetune Qwen2.5-1.5B-Instruct on recipes.

Deliberately a hand-written training loop rather than `Trainer`. The Trainer API
surface moves between transformers majors and we are on 5.x; a plain loop is ~60
lines, has no version risk, and we control exactly what is masked.

Loss is computed on the ASSISTANT TURN ONLY. Training on the user turn as well
would teach the model to generate recipe *questions*, which is a different
finetune from the one we mean to study, and would put question-template tokens
into the diff vector.

Run:  python src/finetune.py
      python src/finetune.py --rank 4 --seed 1      (Day 10: rank/seed runs)

--rank r sets lora_alpha = 2r, so the update scale alpha/r stays 2 as in the
r=16 finetunes (alpha 32) and in ADL's r=64 organisms (alpha 128). --seed
seeds LoRA initialisation and dropout; the data order is fixed (no shuffle)
in every run. The original r=16 finetunes were run without a seed.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Domain is a CLI argument so both finetunes run through IDENTICAL code and
# hyperparameters. If the two runs differed in anything but the data, a
# difference between their diff vectors would not be attributable to domain.
DOMAINS = {
    "recipe": ("ft_train.json", "ft_heldout.json", "recipe-lora"),
    "medical": ("ft_med_train.json", "ft_med_heldout.json", "medical-lora"),
}

# Local dir, not the hub name: cdn-lfs.huggingface.co does not resolve on this
# machine, so any hub fetch fails. Weights were pulled with scripts/fetch_model.sh.
BASE = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
DEVICE = "cuda"
DTYPE = torch.bfloat16

MAX_LEN = 512
EPOCHS = 2
LR = 1e-4
GRAD_ACCUM = 8
WARMUP = 10
LOG_EVERY = 20


def lora_config(r: int) -> LoraConfig:
    return LoraConfig(
    r=r,
    lora_alpha=2 * r,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    )


def encode(tok, ex: dict) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Tokenise one example; mask everything before the assistant turn."""
    msgs = ex["messages"]
    full = tok.apply_chat_template(msgs, tokenize=False)
    prompt = tok.apply_chat_template(msgs[:1], tokenize=False,
                                     add_generation_prompt=True)

    ids = tok(full, return_tensors="pt", truncation=True,
              max_length=MAX_LEN).input_ids[0]
    n_prompt = len(tok(prompt, truncation=True, max_length=MAX_LEN).input_ids)

    if n_prompt >= len(ids):          # response entirely truncated away
        return None

    labels = ids.clone()
    labels[:n_prompt] = -100
    return ids, labels


@torch.no_grad()
def eval_loss(model, tok, examples) -> float:
    """Mean loss on assistant tokens. Used for GATE A."""
    model.eval()
    total, n_tok = 0.0, 0
    for ex in examples:
        got = encode(tok, ex)
        if got is None:
            continue
        ids, labels = got
        ids, labels = ids[None].to(DEVICE), labels[None].to(DEVICE)
        out = model(input_ids=ids, labels=labels)
        n = int((labels != -100).sum())
        total += out.loss.item() * n
        n_tok += n
    model.train()
    return total / max(n_tok, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=sorted(DOMAINS), default="recipe")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    train_f, held_f, out_name = DOMAINS[args.domain]
    if args.rank != 16 or args.seed is not None:
        out_name += f"-r{args.rank}" + (f"-s{args.seed}" if args.seed is not None else "")
    LORA = lora_config(args.rank)
    OUT = ROOT / "models" / out_name

    tok = AutoTokenizer.from_pretrained(BASE)
    train = json.loads((DATA / train_f).read_text(encoding="utf-8"))
    heldout = json.loads((DATA / held_f).read_text(encoding="utf-8"))
    print(f"domain={args.domain}  train {len(train)}  heldout {len(heldout)}  -> {out_name}")

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=DTYPE).to(DEVICE)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()   # needed with checkpointing + LoRA
    model.config.use_cache = False

    # Baseline BEFORE any training. Half of GATE A is "did loss actually drop",
    # which is meaningless without this number.
    base_loss = eval_loss(model, tok, heldout[:50])
    print(f"heldout loss BEFORE finetuning: {base_loss:.4f}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    model = get_peft_model(model, LORA)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e6:.1f}M")

    steps = math.ceil(len(train) * EPOCHS / GRAD_ACCUM)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / WARMUP if s < WARMUP
        else 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / max(steps - WARMUP, 1))),
    )

    print(f"optimizer steps: {steps}")
    model.train()
    t0, seen, running, micro = time.time(), 0, 0.0, 0

    for epoch in range(EPOCHS):
        for ex in train:
            got = encode(tok, ex)
            if got is None:
                continue
            ids, labels = got
            out = model(input_ids=ids[None].to(DEVICE),
                        labels=labels[None].to(DEVICE))
            (out.loss / GRAD_ACCUM).backward()
            running += out.loss.item()
            micro += 1

            if micro % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                seen += 1
                if seen % LOG_EVERY == 0:
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  step {seen:>4}/{steps}  loss {running/(LOG_EVERY*GRAD_ACCUM):.4f}"
                          f"  lr {sched.get_last_lr()[0]:.2e}"
                          f"  {time.time()-t0:.0f}s  peak {mem:.1f}GB")
                    running = 0.0
        print(f"epoch {epoch+1} done ({time.time()-t0:.0f}s)")

    final_loss = eval_loss(model, tok, heldout[:50])
    print(f"\nheldout loss AFTER finetuning:  {final_loss:.4f}  "
          f"(before {base_loss:.4f}, delta {final_loss-base_loss:+.4f})")

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    (OUT / "train_summary.json").write_text(json.dumps({
        "base_model": BASE, "domain": args.domain, "n_train": len(train), "epochs": EPOCHS,
        "lr": LR, "grad_accum": GRAD_ACCUM, "max_len": MAX_LEN,
        "lora_r": LORA.r, "lora_alpha": LORA.lora_alpha, "seed": args.seed,
        "heldout_loss_before": base_loss, "heldout_loss_after": final_loss,
        "minutes": (time.time() - t0) / 60,
    }, indent=1))
    print(f"saved adapter -> {OUT}")


if __name__ == "__main__":
    main()
