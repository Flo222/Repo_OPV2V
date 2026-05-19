cd ~/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$PWD:$PYTHONPATH

python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/point_pillar_v2xvit_opv2v.yaml




