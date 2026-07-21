# ARCE Stage 1: Priority and Unit-Major Layout

This package contains only the first stage of the transport refactor.

## Included changes

- Separate the Where2Comm binary candidate mask from the continuous sender-side priority map.
- Use the binary mask only to choose candidate spatial positions.
- Use the confidence map only to sort positions inside that candidate set.
- Convert the compact payload from `[C,K,1]` to contiguous `[K,C]` before quantization.
- Quantize `[K,C]` with `channel_dim=1` and scatter received rows back with stable spatial IDs.
- Keep full priority and ID tensors out of long-lived communication records.
- Require native Where2Comm priority in the two main experiment configs.

## Explicitly unchanged

- C2MAB proposal and oracle objective
- physical budget allocation
- D-LinUCB implementation and seven-dimensional context
- reward and policy update
- cache semantics
- byte packetizer, packet loss and FEC
- action space

The package therefore does not yet solve quantization-dependent physical budget shrinkage or cache correctness. Those are Stage 2 and Stage 3.

## Install on the server

```bash
cd ~/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$PWD:$PYTHONPATH

tar -xzf /path/to/arce_stage1_priority_layout_20260721.tar.gz
PKG=/path/to/arce_stage1_priority_layout_20260721

REPO_DIR=$PWD bash "$PKG/install.sh" --check
REPO_DIR=$PWD bash "$PKG/install.sh" --apply
```

The installer refuses to overwrite a server file whose hash is neither the reviewed original nor the already-installed Stage 1 version.

## Server validation

First run both methods for 100 frames because the shared Fixed executor and payload preparation are changed:

```bash
METHOD=both RUN_AP=0 RUN_BW=1 MAX_FRAMES=100 \
TAG=stage1_priority_layout_bw100 \
bash scripts/run_arce_pair_eval.sh
```

Inspect these invariants before continuing:

- both methods have the same candidate `num_tokens` for the same frame/link;
- `compact_sparse.layout` is `KC`;
- `selected_tokens_outside_mask` is zero;
- `priority_source` is `where2comm_sender_confidence`;
- `quantization.channel_dim` is 1;
- no missing-priority fallback is used;
- there is no unexpected memory growth across the 100 frames.

Do not use this run to conclude that INT4 now transmits more source information under the same physical budget. That is the separate Stage 2 acceptance criterion.
