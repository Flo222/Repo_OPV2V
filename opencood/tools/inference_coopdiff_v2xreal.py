# -*- coding: utf-8 -*-
"""Inference for CoopDiff / CoopDiff-Markov on V2X-Real.

Compared with inference_v2xreal.py, this script calls model.update_epoch(epoch)
after loading the checkpoint.  CoopDiff uses the epoch to switch from the
student-backbone phase to the diffuser phase, so this is important for
checkpoint evaluation.
"""

import argparse
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import opencood.data_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils, inference_utils
from opencood.utils import eval_utils_v2xreal as eval_utils


def test_parser():
    parser = argparse.ArgumentParser(description='CoopDiff V2X-Real inference')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--fusion_method', required=True, type=str, default='intermediate')
    parser.add_argument('--dataset_mode', type=str, default='vc')
    parser.add_argument('--epoch', default=None, help='epoch number to load model')
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_epoch_override', type=int, default=None,
                        help='Override model.update_epoch() during inference. Use 14 to disable diffuser branch.')
    return parser.parse_args()


def load_model(saved_path, model, epoch):
    if epoch is None:
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        return init_epoch, model
    model_file = os.path.join(saved_path, 'net_epoch%d.pth' % int(epoch))
    if not os.path.exists(model_file):
        raise FileNotFoundError(model_file)
    print('loading epoch %d from %s' % (int(epoch), model_file))
    checkpoint = torch.load(model_file, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    del checkpoint
    return int(epoch), model


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion']

    hypes = yaml_utils.load_yaml(None, opt)
    if opt.dataset_mode:
        hypes['dataset_mode'] = opt.dataset_mode
    print(hypes.get('dataset_mode', ''))

    print('Dataset Building')
    dataset = build_dataset(hypes, visualize=True, train=False)
    print('%d samples found.' % len(dataset))
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=int(opt.num_workers),
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print('Loading Model from checkpoint')
    init_epoch, model = load_model(opt.model_dir, model, opt.epoch)
    eval_epoch = init_epoch if opt.eval_epoch_override is None else int(opt.eval_epoch_override)
    print('CoopDiff inference update_epoch = %s, loaded checkpoint epoch = %s' % (eval_epoch, init_epoch))
    if hasattr(model, 'update_epoch'):
        model.update_epoch(eval_epoch)
    model.eval()

    result_stat = {}
    for class_name in opencood.data_utils.SUPER_CLASS_MAP.keys():
        result_stat[class_name] = {}
        for iou_threshold in [0.3, 0.5, 0.7]:
            result_stat[class_name][iou_threshold] = {'tp': [], 'fp': [], 'gt': 0}

    for i, batch_data in tqdm(enumerate(data_loader)):
        if opt.max_samples > 0 and i >= opt.max_samples:
            break
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                ret = inference_utils.inference_late_fusion(batch_data, model, dataset)
            elif opt.fusion_method == 'nofusion':
                ret = inference_utils.inference_nofusion(batch_data, model, dataset)
            elif opt.fusion_method == 'early':
                ret = inference_utils.inference_early_fusion(batch_data, model, dataset)
            elif opt.fusion_method == 'intermediate':
                ret = inference_utils.inference_intermediate_fusion(batch_data, model, dataset)
            else:
                raise NotImplementedError

            pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = ret
            if pred_box_tensor is None or pred_score is None:
                continue

            for class_id, class_name in enumerate(result_stat.keys()):
                class_id += 1
                for iou_threshold in result_stat[class_name].keys():
                    keep_index_pred = pred_score[:, -1] == class_id
                    keep_index_gt = gt_label_tensor == class_id
                    eval_utils.caluclate_tp_fp(
                        pred_box_tensor[keep_index_pred, ...],
                        pred_score[keep_index_pred, 0],
                        gt_box_tensor[keep_index_gt, ...],
                        result_stat[class_name],
                        iou_threshold)

    eval_utils.eval_final_results(result_stat, opt.model_dir)


if __name__ == '__main__':
    main()
