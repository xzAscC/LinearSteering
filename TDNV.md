# TDNV for Safety vs Unsafety

For layer \(\ell\), task \(t\), and sample \(i\), let the representation be
\(h_{t,i}^{(\ell)} \in \mathbb{R}^d\). The task mean is:

\[
\bar{h}_t^{(\ell)} = \frac{1}{N}\sum_{i=1}^{N} h_{t,i}^{(\ell)}
\]

Within-task variance (squared-distance form):

\[
\mathrm{var}_t^{(\ell)} = \frac{1}{N}\sum_{i=1}^{N}\left\|h_{t,i}^{(\ell)} - \bar{h}_t^{(\ell)}\right\|_2^2
\]

TDNV (Task-Distance Normalized Variance):

\[
\mathrm{TDNV}^{(\ell)}
= \sum_{t=1}^{T}\sum_{\substack{t'=1\\t'\neq t}}^{T}
\frac{\mathrm{var}_t^{(\ell)} + \mathrm{var}_{t'}^{(\ell)}}
{2\left\|\bar{h}_t^{(\ell)} - \bar{h}_{t'}^{(\ell)}\right\|_2^2}
\]

For our 2-task setting (Safety vs Unsafety), this simplifies to:

\[
\mathrm{TDNV}^{(\ell)}
= \frac{\mathrm{var}_{\text{safety}}^{(\ell)} + \mathrm{var}_{\text{unsafety}}^{(\ell)}}
{\left\|\bar{h}_{\text{safety}}^{(\ell)} - \bar{h}_{\text{unsafety}}^{(\ell)}\right\|_2^2}
\]

Lower TDNV means better compression/discrimination (smaller within-task spread,
larger between-task separation).

## Implementation in This Repo (Qwen3)

- Script: `src/pku_tdnv.py`
- Tasks: `label=pos` as safety, `label=neg` as unsafety
- Representation: last-token hidden state at each selected layer
- Model default: `Qwen/Qwen3-1.7B`

Run:

```bash
uv run python src/pku_tdnv.py \
  --model Qwen/Qwen3-1.7B \
  --dataset dataset/pku_saferlhf_pos_neg_100x2_minimal.jsonl \
  --text_key prompt \
  --layers all
```

Outputs:

- JSON: `assets/tdnv/Qwen3-1.7B/tdnv_pku_safety_unsafety_*.json`
- Plot: `plots/tdnv_pku_safety_unsafety_*.pdf`
