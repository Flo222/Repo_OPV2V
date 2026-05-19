CUDA_VISIBLE_DEVICES=0 python -u opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/point_pillar_rocooper_opv2v.yaml \
  2>&1 | tee train_rocooper_opv2v.log

export PYTHONPATH=$PWD:$PYTHONPATH

(opencood) server@server-System-Product-Name:~/v2x_projects/OPV2V$ CUDA_VISIBLE_DEVICES=0 python -u opencood/tools/train.py   --hypes_yaml opencood/hypes_yaml/point_pillar_rocooper_opv2v.yaml   2>&1 | tee train_rocooper_opv2v.log
Not using distributed mode
-----------------Dataset Building------------------
too many cavs
too many cavs
too many cavs
---------------Creating Model------------------
optimizer method is: <class 'torch.optim.adam.Adam'>
Training start
/home/server/anaconda3/envs/opencood/lib/python3.7/site-packages/torch/optim/lr_scheduler.py:136: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
/home/server/anaconda3/envs/opencood/lib/python3.7/site-packages/torch/optim/lr_scheduler.py:156: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
learning rate 0.0010000
[epoch 0][36/3187], || Loss: 5.4886 || Conf Loss: 2.0242 || Loc Loss: 3.4644:   1%|          | 36/3187 [00:08<08:28,  6.19it/s]Traceback (most recent call last):
  File "opencood/tools/train.py", line 207, in <module>
    main()
  File "opencood/tools/train.py", line 155, in main
    ouput_dict = model(batch_data['ego'])
  File "/home/server/anaconda3/envs/opencood/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/server/v2x_projects/OPV2V/opencood/models/point_pillar_rocooper.py", line 539, in forward
    data_dict=data_dict,
  File "/home/server/anaconda3/envs/opencood/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/server/v2x_projects/OPV2V/opencood/models/fuse_modules/rocooper_fuse.py", line 522, in forward
    data_dict=data_dict,
  File "/home/server/v2x_projects/OPV2V/opencood/models/fuse_modules/rocooper_fuse.py", line 362, in _call_aggregator
    data_dict=data_dict,
  File "/home/server/anaconda3/envs/opencood/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/server/v2x_projects/OPV2V/opencood/models/sub_modules/rocooper_aggregator.py", line 1008, in forward
    round_idx=round_idx,
  File "/home/server/v2x_projects/OPV2V/opencood/models/sub_modules/rocooper_aggregator.py", line 912, in _process_one_round
    scale=scale,
  File "/home/server/v2x_projects/OPV2V/opencood/models/sub_modules/rocooper_aggregator.py", line 814, in _process_one_scale
    others_selected=selected_others,
  File "/home/server/anaconda3/envs/opencood/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/server/v2x_projects/OPV2V/opencood/models/sub_modules/rocooper_aggregator.py", line 340, in forward
    need_weights=False,
  File "/home/server/anaconda3/envs/opencood/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/server/anaconda3/envs/opencood/lib/python3.7/site-packages/torch/nn/modules/activation.py", line 1160, in forward
    attn_mask=attn_mask, average_attn_weights=average_attn_weights)
  File "/home/server/anaconda3/envs/opencood/lib/python3.7/site-packages/torch/nn/functional.py", line 5131, in multi_head_attention_forward
    v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
RuntimeError: CUDA out of memory. Tried to allocate 28.00 MiB (GPU 0; 23.52 GiB total capacity; 20.79 GiB already allocated; 2.81 MiB free; 21.32 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
[epoch 0][36/3187], || Loss: 5.4886 || Conf Loss: 2.0242 || Loc Loss: 3.4644:   1%|          | 36/3187 [00:09<13:39,  3.85it/s]
batch_size=2 显存爆炸

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CUDA_VISIBLE_DEVICES=0 python -u opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/point_pillar_rocooper_opv2v.yaml \
  2>&1 | tee train_rocooper_opv2v_bs1.log