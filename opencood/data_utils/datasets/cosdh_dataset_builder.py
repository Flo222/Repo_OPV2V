from opencood.data_utils.datasets.intermediate_late_fusion_dataset_cosdh import getIntermediatelateFusionDataset
from opencood.data_utils.datasets.opv2v_basedataset_cosdh import OPV2VBaseDataset


def build_dataset_cosdh(dataset_cfg, visualize=False, train=True):
    fusion_name = dataset_cfg["fusion"]["core_method"]

    if fusion_name == "intermediatelate":
        dataset_cls = getIntermediatelateFusionDataset(OPV2VBaseDataset)
        return dataset_cls(params=dataset_cfg, visualize=visualize, train=train)

    from opencood.data_utils.datasets import build_dataset
    return build_dataset(dataset_cfg, visualize=visualize, train=train)
