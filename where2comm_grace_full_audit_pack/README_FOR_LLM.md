# Where2Comm + GRACE/ARCE Implementation Audit Package

## Purpose

This package is prepared for auditing an OpenCOOD/OPV2V cooperative perception project.

The goal is to compare:

1. Where2Comm baseline under three-state Markov communication links;
2. GRACE/ARCE full method integrated into Where2Comm.

The main question is why AP drops after adding GRACE/ARCE, and whether the drop is due to:

- real method cost;
- unfair baseline;
- budget/statistics mismatch;
- mask/fusion implementation bug;
- quantization/FEC/cache side effects;
- reward design;
- evaluation inconsistency.

## Current Known Results

learned QRA:
AP@0.3 = 0.792
AP@0.5 = 0.778
AP@0.7 = 0.610

learned QRA + complementarity + FEC + cache:
AP@0.3 = 0.792
AP@0.5 = 0.777
AP@0.7 = 0.609

previous hand-crafted channel-aware heuristic:
AP@0.3 ≈ 0.799
AP@0.5 ≈ 0.787
AP@0.7 ≈ 0.623

The audit should determine whether these results are caused by implementation issues, fairness mismatch, or the real communication-quality tradeoff.

## Important Method Components

- Dataset: OPV2V
- Framework: OpenCOOD
- Backbone method: Where2Comm
- Added method: GRACE/ARCE
- Link model: three-state Markov good / medium / bad
- Budget: ego-side global shared communication budget
- Quantization: FP32 / FP16 / INT8 / INT4
- Redundancy: rho0 / rho0.25 / rho0.5
- Temporal cache: cache0 / cache1
- Complementarity: normalized with bounded transform
- Bandit: contextual LinUCB
- Reward: confidence, q_eff, cost, delay, quant quality, FEC gain, cache quality

## Requested Audit Tasks

Please inspect code and logs to determine:

1. Whether baseline Where2Comm under Markov link is correctly implemented.
2. Whether GRACE/ARCE uses the same checkpoint, split, fusion method, postprocess, and AP evaluation.
3. Whether both baseline and GRACE/ARCE are affected by the same communication/channel constraints.
4. Whether GRACE/ARCE actually modifies fused features, not only communication statistics.
5. Whether tx_bytes/rx_bytes/budget_bytes match actual feature transmission.
6. Whether quantization, FEC, cache, complementarity, reward, and bandit are implemented as designed.
7. Why AP decreases after adding GRACE/ARCE.
8. What minimal diagnostic experiments should be run next.
