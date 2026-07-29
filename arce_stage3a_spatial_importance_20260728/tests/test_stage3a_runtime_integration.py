from argparse import Namespace

import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils


def _load(model_dir):
    hypes = yaml_utils.load_yaml(
        None,
        Namespace(model_dir=model_dir),
    )
    return train_utils.create_model(hypes)


def main():
    arce_model = _load(
        "opencood/logs/main_opv2v_where2comm_grace_full"
    )
    fixed_model = _load(
        "opencood/logs/where2comm_markov_trueloss_fp32_rho0_cache0"
    )

    arce = arce_model.arce_comm
    fixed = fixed_model.arce_comm
    assert arce.uses_arce_spatial_importance is True
    assert arce.executor.uses_arce_spatial_importance is True
    assert fixed.uses_arce_spatial_importance is False

    feature = torch.zeros((4, 2, 3), dtype=torch.float32)
    feature[:, 0, 0] = 4.0
    feature[:, 0, 2] = 1.0
    feature[:, 1, 1] = 2.0

    compact, meta = arce.executor._compact_feature_by_message_mask(
        feature,
        message_mask=None,
        priority_map=None,
    )
    assert meta["layout"] == "KC"
    assert meta["candidate_source"] == (
        "arce_nonzero_spatial_support"
    )
    assert meta["priority_source"] == "arce_sender_feature_rms"
    assert tuple(compact.shape) == (3, 4)
    assert meta["indices"].tolist() == [0, 4, 2]
    assert meta["spatial_importance"]["num_candidate_units"] == 3

    native_mask = torch.zeros((1, 2, 3), dtype=torch.float32)
    native_mask[:, 0, 1] = 1.0
    native_mask[:, 1, 2] = 1.0
    fixed_compact, fixed_meta = (
        fixed._compact_feature_by_message_mask(
            feature,
            message_mask=native_mask,
            priority_map=None,
        )
    )
    assert fixed_meta["layout"] == "CK1"
    assert fixed_meta["priority_source"] == (
        "legacy_masked_confidence"
    )
    assert tuple(fixed_compact.shape) == (4, 2, 1)

    empty, empty_meta = (
        arce.executor._compact_feature_by_message_mask(
            torch.zeros_like(feature),
            message_mask=None,
            priority_map=None,
        )
    )
    assert tuple(empty.shape) == (0, 4)
    assert empty_meta["empty_candidate"] is True

    print("ARCE own importance:", arce.spatial_importance_method)
    print("ARCE unit order:", meta["indices"].tolist())
    print("Fixed layout:", fixed_meta["layout"])
    print("Empty shape:", tuple(empty.shape))
    print("Stage 3A runtime integration: PASS")


if __name__ == "__main__":
    main()
