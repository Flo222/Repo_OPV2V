# GRACE RFC 6330 RaptorQ block transport

This patch replaces redundant GRACE actions with a real RFC 6330 RaptorQ
backend and a priority-preserving block scheduler. It does not change the
D-LinUCB implementation, Ego Oracle algorithm, context definition, reward
formula, or the Fixed baseline configuration.

The formal configuration requests `raptorq` explicitly. The legacy `raptor`
alias continues to mean `raptor_sim`, so old experiment configurations retain
their original behavior.

## Wire semantics

The existing GRACE preprocessing has already ordered complete spatial units
by priority. Quantization and the existing byte-stream packetizer then produce
1024-byte source symbols in that order.

For every admitted protection block `b`, the selected action ratio is first
written as its smallest integer packet group:

```text
rho=0.10 -> 10 source + 1 repair
rho=0.25 ->  4 source + 1 repair
rho=0.60 ->  5 source + 3 repair
```

The scheduler admits the largest complete integer group that fits, encodes its
source symbols as one RaptorQ block, and places that block's repair symbols
immediately after its source symbols:

```text
[protected block 1 sources][block 1 repairs]
[protected block 2 sources][block 2 repairs]
...
[best-effort lower-priority source tail]
```

If the remaining budget cannot hold one more complete ratio group, it is not
rounded upward and is not left unused. The remaining packet slots carry later,
lower-priority source symbols without repair. Thus the protected prefix keeps
the exact selected ratio, while the link continues to send useful content.
Repair packets and tail source packets both consume the same physical budget
and BW as any other wire packet.

Under a 12-packet Bad-state wire budget, the schedules are:

| action | protected source | repair | unprotected tail | wire total |
|---|---:|---:|---:|---:|
| rho=0.10 | 10 | 1 | 1 | 12 |
| rho=0.25 | 8 | 2 | 2 | 12 |
| rho=0.60 | 5 | 3 | 4 | 12 |

Runtime records deliberately distinguish two ratios:

```text
redundancy_ratio_target
protected_redundancy_ratio = repair / protected_source
overall_redundancy_ratio   = repair / (protected_source + tail_source)
```

`protected_redundancy_ratio` equals the selected action ratio exactly whenever
at least one complete group is protected. `overall_redundancy_ratio` is lower
when a best-effort tail exists; it is not the protection strength of the
protected prefix. A lost tail packet cannot be recovered by the preceding
block's repair packets and remains eligible for the existing cache/zero-fill
recovery path.

Each wire packet contains:

```text
2 B GRACE block id
2 B source-symbol count for this block
4 B RFC 6330 RaptorQ Payload ID
1024 B encoding symbol
= 1032 B on wire
```

The 8-byte metadata is included in `actual_transmitted_bytes`, BW, and budget
admission. A no-FEC action remains on the existing 1024-byte packet path.

## Install on the server

```bash
cd ~/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$PWD:$PYTHONPATH

python -m pip install --only-binary=:all: raptorq==1.6.3

tar -xzf grace_raptorq_blockwise_rfc6330_v1_20260801.tar.gz
PKG="$PWD/grace_raptorq_blockwise_rfc6330_v1_20260801"

REPO_DIR="$PWD" bash "$PKG/install.sh" --check
REPO_DIR="$PWD" bash "$PKG/install.sh" --apply
```

The installer creates a timestamped backup under `refactor_backups/` before
changing files.

## Runtime acceptance

The installer runs these checks automatically:

1. `raptorq==1.6.3` is importable.
2. The online action space has 25 arms: one no-send arm and 24 joint actions.
3. Redundant actions use `fec_type=raptorq`.
4. Source order remains systematic and byte-exact.
5. Lost source symbols are recovered byte-for-byte by RFC 6330 repair symbols.
6. Source and repair symbols jointly obey Good/Medium/Bad physical budgets.
7. Every protected block has the exact requested redundancy ratio.
8. Remaining wire slots transmit an unprotected lower-priority source tail.
9. The wire order is source-then-repair within every protected block.

## First runtime experiment

Run a 100-frame single-pass AP+BW audit before a full evaluation:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

METHOD=arce RUN_AP=1 RUN_BW=1 MAX_FRAMES=100 \
TAG=raptorq_blockwise_rfc6330_ap_bw100 \
bash scripts/run_arce_pair_eval.sh
```

Check the generated runtime records for:

```text
fec_type = raptorq
standard = RFC6330
scheduling = exact_ratio_protected_prefix_with_best_effort_tail
wire_packet_bytes = 1032
actual_transmitted_bytes <= physical_execution_budget_bytes
num_encoded_packets = num_admitted_source_packets + num_parity_packets
protected_redundancy_ratio = redundancy_ratio_target
overall_redundancy_ratio <= redundancy_ratio_target
```

Do not start the full 2170-frame run until these checks and the 100-frame AP/BW
result are both normal.
