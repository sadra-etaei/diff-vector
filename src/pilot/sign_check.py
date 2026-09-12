"""Resolve the sign, and check the effect survives dropping position 0."""
import torch
from diffing import load, logit_lens, DEVICE, DTYPE
from smoke_test import PROMPTS, BASE, CHAT
from diagnose import acts_by_position

store = {}
for tag, name in [("base", BASE), ("chat", CHAT)]:
    m, t = load(name)
    store[tag] = acts_by_position(m, t, PROMPTS, n_pos=12)
    if tag == "chat":
        cm, ct = m, t
    else:
        del m; torch.cuda.empty_cache()

pd = store["chat"] - store["base"]
L = pd.shape[0] - 1

# average over positions 1..11, dropping the position-0 attention sink
avg_no0 = pd[L, 1:].mean(0)

@torch.no_grad()
def both_ends(vec, k=15):
    v = vec.to(DEVICE, DTYPE)
    logits = cm.lm_head(cm.model.norm(v.unsqueeze(0)).squeeze(0)).to(torch.float32)
    hi = logits.topk(k).indices.tolist()
    lo = logits.topk(k, largest=False).indices.tolist()
    return [ct.decode([i]) for i in hi], [ct.decode([i]) for i in lo]

hi, lo = both_ends(avg_no0)
print("=== mean diff over positions 1-11, final layer ===")
print("PROMOTED by instruct (top-k):")
print("  " + ", ".join(repr(t) for t in hi))
print("SUPPRESSED by instruct (bottom-k):")
print("  " + ", ".join(repr(t) for t in lo))

print("\n=== same, but for position 0 alone (the attention sink) ===")
hi0, lo0 = both_ends(pd[L, 0])
print("PROMOTED:  " + ", ".join(repr(t) for t in hi0))
print("SUPPRESSED:" + ", ".join(repr(t) for t in lo0))

print("\n=== control: bottom-k of random vectors (does junk cluster semantically?) ===")
torch.manual_seed(1)
d = pd.shape[-1]
for i in range(2):
    r = torch.randn(d) * (avg_no0.norm() / (d ** 0.5))
    rh, rl = both_ends(r, k=15)
    print(f"  rand{i} bottom: " + ", ".join(repr(t) for t in rl))
