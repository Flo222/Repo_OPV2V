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
    executor = arce.executor
    assert executor.zero_codec.enabled is True
    assert fixed.zero_codec.enabled is False

    q_tensor = torch.zeros((40, 64), dtype=torch.int8)
    q_tensor[:, 0] = torch.remainder(
        torch.arange(40, dtype=torch.int8),
        7,
    )
    unit_ids = [100 + i for i in range(40)]
    result = executor.zero_codec.packetize(
        q_tensor=q_tensor,
        quant_mode="int4",
        unit_ids=unit_ids,
    )
    assert result.encoding_mode == "adaptive_unit_bitmap"
    assert result.num_bitmap_units > 0
    assert result.metadata_bytes > 0
    assert result.num_packets * result.packet_size_bytes < (
        (result.dense_num_bytes + result.packet_size_bytes - 1)
        // result.packet_size_bytes
        * result.packet_size_bytes
    )

    receive_mask = torch.ones(
        result.num_packets,
        dtype=torch.bool,
    )
    recovered, valid = executor.zero_codec.unpacketize(
        result.packets,
        result,
        receive_mask,
    )
    assert bool(valid.all().item())
    assert torch.equal(recovered, q_tensor)

    if result.num_packets > 1:
        receive_mask[0] = False
        received_packets = result.packets.clone()
        received_packets[0] = 0
        partial, partial_valid = executor.zero_codec.unpacketize(
            received_packets,
            result,
            receive_mask,
        )
        assert int((~partial_valid).sum().item()) > 0
        assert torch.equal(
            partial[partial_valid],
            q_tensor[partial_valid],
        )
        assert int(
            torch.count_nonzero(partial[~partial_valid]).item()
        ) == 0

    print("ARCE zero codec:", executor.zero_codec.get_config())
    print("Fixed zero codec:", fixed.zero_codec.get_config())
    print("INT4 dense bytes:", result.dense_num_bytes)
    print("INT4 encoded valid bytes:", result.encoded_valid_bytes)
    print("INT4 source packets:", result.num_packets)
    print("Stage 3B runtime integration: PASS")


if __name__ == "__main__":
    main()
