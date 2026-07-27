# ARCE receiver temporal cache repair

This patch repairs the `cache1` execution path without changing UCB, reward,
action space, frame budget allocation, packet loss, or the fixed baseline.

## Runtime semantics

- The existing sender-side previous-frame delay cache remains unchanged.
- A separate receiver cache stores only spatial units that were actually
  recovered from transmitted packets.
- Cache lookup uses the stable original Where2Comm spatial `unit_id`.
- `cache1` may fill currently missing units from the immediately preceding
  receiver cache entry.
- `cache0` never consumes receiver-cache content.
- Cache-filled values are not marked as newly received when updating the next
  cache entry, preventing indefinite propagation of stale values.
- Entries older than `temporal_fusion.max_age_frames` are rejected.

The path is active only for ARCE's Stage-1 `KC` payload layout and a
`cache_enabled=1` action. The fixed baseline remains `cache0` with its legacy
layout, so its execution is unchanged.

## New audit fields

`partial_reconstruction.temporal_cache` records cache availability and unit
coverage. The main fields are:

- `cache_status`
- `cache_hit`
- `num_current_recovered_units`
- `num_temporal_filled_units`
- `num_temporal_filled_packets`
- `q_cache`
- `q_eff`

`quality.q_cache`, `quality.q_eff`, and
`partial_reconstruction.num_temporal_filled_packets` are no longer hard-coded
to zero.

## Install and test

```bash
REPO_DIR="$PWD" bash install.sh --check
REPO_DIR="$PWD" bash install.sh --apply
```

After installation, run the normal 100-frame single-pass ARCE AP+BW
evaluation and audit `cache1` records. At least one cache hit is expected only
when the same ego-sender link has a valid receiver cache from the immediately
preceding frame and the current payload has missing units.
