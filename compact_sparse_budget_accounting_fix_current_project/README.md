# Compact-sparse source/parity budget-accounting fix

This package is tailored to the uploaded current project state.

The current repository already computes source/parity budget outcomes inside
`ARCEFixedComm.communicate_feature`, but its active
`compression_auditor.py` is the Experiment-1-only version. Consequently,
Experiment 2 transmits the correct number of bytes while audit fields such as
`bandwidth_budget_bytes`, `num_transmitted_source_packets`, and
`source_tx_ratio` are missing.

The installer:

1. backs up the two active source files;
2. installs the combined Experiment 1/2/2.5 auditor;
3. passes authoritative runtime packet accounting directly from
   `communicate_feature` to the auditor;
4. runs Experiment 1, Experiment 2, Experiment 2.5, and compact-sparse smoke
   tests.

It does not change packet selection, quantization, FEC, loss, reconstruction,
fusion, or detection.

Install:

```bash
bash compact_sparse_budget_accounting_fix_current_project/install.sh \
  /home/server/v2x_projects/OPV2V
```
