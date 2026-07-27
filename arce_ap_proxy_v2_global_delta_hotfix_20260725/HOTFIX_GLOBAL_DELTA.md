# AP-proxy v2 global-delta hotfix

Apply this only after the first `arce_ap_proxy_v2_20260725` package was
installed.

The first v2 smoke exposed a multi-collaborator definition mismatch:
target-sender no-send is not necessarily ego-only because other senders may
still transmit.

This hotfix exports both:

```text
global delta = Q(action output) - Q(ego-only)
action delta = Q(action output) - Q(target-sender no-send)
```

The paired reward proxy is trained on global delta. Action delta is used only
for centered marginal sign and action-ranking audits.

Apply from the OPV2V root:

```bash
BACKUP_DIR="refactor_backups/ap_proxy_v2_global_delta_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp opencood/tools/audit_arce_counterfactual.py "$BACKUP_DIR/"
cp opencood/tools/train_counterfactual_ap_proxies.py "$BACKUP_DIR/"

git apply --check ap_proxy_v2_global_delta_hotfix.patch
git apply ap_proxy_v2_global_delta_hotfix.patch

python -m py_compile \
  opencood/tools/audit_arce_counterfactual.py \
  opencood/tools/train_counterfactual_ap_proxies.py

echo "Backup: $BACKUP_DIR"
```

Do not use the CSV from the first smoke for training. Repeat the smoke into a
new output directory and verify that its header includes:

```text
true_ego_quality_mean_0357
label_true_global_delta_quality_mean_0357
label_true_delta_quality_mean_0357
proxy_action_delta
```
