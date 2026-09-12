"""Smoke test: does the harness reproduce a KNOWN diffing signal?

Base vs Instruct on Qwen2.5-0.5B. The instruct model was chat-finetuned, so the
mean activation difference should decode to chat/assistant-flavoured tokens even
though the probe text is generic prose. If this shows nothing, the harness is
broken and nothing downstream is trustworthy.
"""
import torch
from diffing import load, cache_acts, logit_lens

PROMPTS = [
    "The mitochondrion is an organelle found in most eukaryotic cells.",
    "Rainfall in the region peaks during the monsoon season each year.",
    "He walked down to the harbour and watched the fishing boats return.",
    "The bridge was completed in 1887 after six years of construction.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "Local elections were held across the province last November.",
    "She published her first novel at the age of twenty-nine.",
    "Iron ore deposits were discovered in the northern hills.",
    "The treaty established new borders between the two states.",
    "Migrating birds navigate using the earth's magnetic field.",
    "Concrete gains strength slowly over a period of many weeks.",
    "The orchestra performed three symphonies that evening.",
    "Glaciers in the range have retreated substantially since 1950.",
    "Wheat and barley were among the earliest domesticated crops.",
    "The museum houses a collection of medieval manuscripts.",
    "Tidal patterns along the coast follow a semidiurnal cycle.",
]

BASE, CHAT = "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"

acts = {}
for tag, name in [("base", BASE), ("chat", CHAT)]:
    model, tok = load(name)
    acts[tag] = cache_acts(model, tok, PROMPTS)
    print(f"{tag}: cached {tuple(acts[tag].shape)} (layers+1, d_model)")
    if tag == "chat":
        chat_model, chat_tok = model, tok   # keep for the lens
    else:
        del model
        torch.cuda.empty_cache()

diff = acts["chat"] - acts["base"]
n_layers = diff.shape[0] - 1
print(f"\nper-layer L2 norm of mean activation difference:")
for i in range(0, n_layers + 1, 4):
    print(f"  layer {i:2d}: {diff[i].norm().item():8.3f}")

for layer in [n_layers // 2, n_layers - 4, n_layers]:
    toks = logit_lens(diff[layer], chat_model, chat_tok, k=20)
    print(f"\n--- logit lens on mean diff, layer {layer} ---")
    print(", ".join(repr(t) for t, _ in toks))
