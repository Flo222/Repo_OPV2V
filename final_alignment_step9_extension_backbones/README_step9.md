# Step 9：扩展 backbone plug-in 实验

本包给出 V2X-ViT / RoCooper / CoopDiff + Ours 的扩展实验脚本和 YAML 生成工具框架。

## 实验定位

主实验仍然是 Where2comm。Step 9 只用于证明 ARCE/C²MAB 是通信层插件，不是 Where2comm 专属。

## 文件

```text
opencood/tools/generate_extension_arce_yamls.py
scripts/run_extension_backbones.sh
```

## 生成扩展 YAML

V2X-ViT 示例：

```bash
python opencood/tools/generate_extension_arce_yamls.py \
  --base opencood/hypes_yaml/point_pillar_v2xvit_opv2v_arce_dc2mab_full.yaml \
  --backbone v2xvit \
  --out opencood/hypes_yaml/extension_arce/v2xvit_ours.yaml
```

RoCooper / CoopDiff 需要你先提供对应已能正常 inference 的 base YAML：

```bash
python opencood/tools/generate_extension_arce_yamls.py --base <rocooper.yaml> --backbone rocooper --out <out.yaml>
python opencood/tools/generate_extension_arce_yamls.py --base <coopdiff.yaml> --backbone coopdiff --out <out.yaml>
```

## 运行

修改 `scripts/run_extension_backbones.sh` 里的 model_dir 后执行。
