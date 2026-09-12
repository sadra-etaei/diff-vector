"""Why is the logit lens on the mean diff returning junk?

Tests two hypotheses:
  (H-pos)  Signal lives at early token positions and is destroyed by averaging.
  (H-junk) Logit lens decodes ANY vector to rare-token junk, so we have no
           baseline to judge 'garbage' against.
"""
import torch
from diffing import load, logit_lens, DEVICE, DTYPE
from smoke_test import PROMPTS, BASE, CHAT


@torch.no_grad()
def acts_by_position(model, tok, prompts, n_pos=12, max_len=64):
    """Mean activation at each of the first n_pos token positions."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True,
              truncation=True, max_length=max_len).to(DEVICE)
    out = model(**enc, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, 0).to(torch.float32)  # (L, b, s, d)
    return hs[:, :, :n_pos, :].mean(dim=1).cpu()              # (L, n_pos, d)


store = {}
for tag, name in [("base", BASE), ("chat", CHAT)]:
    m, t = load(name)
    store[tag] = acts_by_position(m, t, PROMPTS)
    if tag == "chat":
        chat_model, chat_tok = m, t
    else:
        del m; torch.cuda.empty_cache()

pos_diff = store["chat"] - store["base"]          # (L, n_pos, d)
L = pos_diff.shape[0] - 1

print("=== H-junk control: logit lens on RANDOM vectors ===")
print("(if these look like the 'real' results, the lens is telling us nothing)")
torch.manual_seed(0)
d = pos_diff.shape[-1]
for trial in range(2):
    rand = torch.randn(d) * (pos_diff[L].norm() / (d ** 0.5))
    toks = logit_lens(rand, chat_model, chat_tok, k=12)
    print(f"  random {trial}: " + ", ".join(repr(t) for t, _ in toks))

print("\n=== H-pos: diff norm by token position (layer %d) ===" % L)
for p in range(pos_diff.shape[1]):
    print(f"  pos {p:2d}: norm {pos_diff[L, p].norm().item():8.3f}")

print("\n=== H-pos: logit lens per position, final layer ===")
for p in [0, 1, 2, 5, 11]:
    toks = logit_lens(pos_diff[L, p], chat_model, chat_tok, k=12)
    print(f"  pos {p:2d}: " + ", ".join(repr(t) for t, _ in toks))
