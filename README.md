# Project 1：NumPy 复现 OpenCLIP ViT-B/32 图像编码器

## 一、项目功能

本项目在**纯 NumPy** 环境下（不使用 PyTorch、不联网、不做模型训练），用给定的训练好的模型权重，
复现 OpenCLIP **ViT-B/32** 图像编码器的前向计算：输入一批预处理后的图片，输出每张图一行
512 维、长度归一化的特征向量。该编码器将作为后续大作业（图文匹配 CLIP）的图像塔复用。

- 输入：预处理后的 float32 数组 `[B, 3, 224, 224]`
- 内部 token 序列（12 层 Block 后）：`[B, 50, 768]`（1 个 CLS + 49 个 patch）
- 输出：float32 `[B, 512]`，每张图一行单位长度特征向量

## 二、目录结构

```
project1-numpy-student/
├── image_encoder.py        # 【本人实现】图像编码器：6 个 TODO 全部补全
├── numpy_homework.py       # 【本人实现】12 道 NumPy 练习题（8 基础 + 4 中等）
├── helpers.py               # 课程提供：LayerNorm、精确 GELU、softmax、单位化等工具
├── preprocess_image.py     # 课程提供：缩放、中心裁剪、像素归一化
├── runtime.py              # 课程提供：权重 / 图片加载
├── run.py                   # 课程提供：命令行入口，图片 -> 输出特征向量
├── check.py                 # 课程提供：分阶段数值比对校验
├── transformer_steps.py     # 课程提供：3 行 4 列微型 Transformer 教学示范
├── config.json              # 模型结构与归一化常数
├── requirements.txt         # 依赖清单
├── weights/                 # 训练好的权重（大文件，不纳入 Git，由课程附件提供）
│   ├── image_encoder.npz
│   └── manifest.json
├── assets/
│   ├── images/              # 10 张测试图
│   ├── images.json          # 图片 ID 与来源记录
│   └── reference_outputs.npz # 各阶段参考答案（仅校验用，不纳入 Git）
├── lecture-demos/           # 课堂 NumPy 示例与示范输出
└── practice/                # 练习题 PDF（题目版）
```

## 三、环境与依赖

- Python 3.9 及以上（开发环境为 Python 3.13）
- 无需 GPU、无需 PyTorch / OpenCLIP 包、无需联网下载模型

安装依赖：

```bash
python -m pip install -r requirements.txt
```

依赖清单（见 `requirements.txt`）：

| 包 | 版本范围 | 用途 |
| --- | --- | --- |
| numpy | >=1.24,<3 | 全部数组与矩阵运算 |
| Pillow | >=9.2,<13 | 读图、缩放与裁剪 |
| imageio | >=2.31,<3 | 课堂图片示例 |

## 四、数据与权重获取

- 模型权重 `weights/image_encoder.npz`（152 个视觉张量，约 326 MB）与参考答案
  `assets/reference_outputs.npz` 由**课程实验附件**提供，按附件原始目录放置即可。
- 因体积较大，这两类文件**不纳入 Git 仓库**（见 `.gitignore`）；克隆后如需本地运行，
  请从课程附件重新获取并放回对应目录。
- 10 张测试图随附件提供于 `assets/images/`，其来源与校验值记录在 `assets/images.json`。
- 校验脚本只把参考答案用于比对，**不作为编码器输入**；编码器需对其他图片、其他 batch 大小同样成立。

## 五、运行方法

在项目根目录下执行：

```bash
# 1. 分阶段数值校验（assets → preprocess → patches → tokens → block0
#    → last_tokens → output → batch）
python check.py --stage all

# 2. 对默认 10 张图提取特征，保存为 npz
python run.py --output output_tokens.npz

# 3. 也可指定自己的图片
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

可选：运行微型 Transformer 教学示范，对照理解逐层公式：

```bash
python transformer_steps.py
```

## 六、本人实现的内容（`image_encoder.py`）

只修改了 `image_encoder.py`，共 6 组 TODO：

| 编号 | 方法 | 实现内容 | 形状变化 |
| --- | --- | --- | --- |
| TODO-01 | `patchify` | 切 7×7=49 个 32×32 patch 并按 C,H,W 展平 | `[B,3,224,224]→[B,49,3072]` |
| TODO-02 | `split_heads` | reshape + transpose 把特征拆成 12 个头 | `[B,N,768]→[B,12,N,64]` |
| TODO-03 | 注意力 `__call__` | QK 分数（÷√64）、softmax 加权 Value、合并头、输出投影 | `[B,12,N,64]→[B,N,768]` |
| TODO-04 | `TransformerBlock.__call__` | 第一条残差、MLP（GELU）、第二条残差 | `[B,N,768]→[B,N,768]` |
| TODO-05 | `embed` | patch 线性投影、前置拼接 CLS、加位置编码 | `[B,49,3072]→[B,50,768]` |
| TODO-06 | `__call__` | 选取 CLS、后归一化、投影到 512 维、单位化 | `[B,50,768]→[B,512]` |

关键权重方向约定：

1. patch 投影无 bias，使用 `patch_weight.T`；
2. 注意力 / MLP 线性权重存储为 `[out, in]`，矩阵乘时使用 `weight.T`；
3. 最终 `visual.proj` 已是 `[in, out]=[768,512]`，**不再转置**。

## 七、数值校验结果

`check.py` 与独立 FP32 参考逐阶段比对（形状、float32、有限值 + 误差容忍），全部 PASS：

| 阶段 | 形状 | 最大绝对误差 | 平均绝对误差 | 容忍 atol |
| --- | --- | --- | --- | --- |
| preprocess | (10, 3, 224, 224) | 0.000e+00 | 0.000e+00 | 2e-6 |
| patches | (2, 49, 3072) | 0.000e+00 | 0.000e+00 | 2e-4 |
| tokens | (2, 50, 768) | 4.005e-05 | 1.289e-06 | 2e-4 |
| block0 | (2, 50, 768) | 7.063e-06 | 3.250e-07 | 2e-4 |
| last_tokens | (2, 50, 768) | 4.768e-05 | 2.696e-06 | 2e-4 |
| output | (10, 512) | 5.439e-07 | 8.311e-08 | 2e-5 |
| batch | (3, 512) | 4.619e-07 | 7.632e-08 | 2e-5 |

`batch` 阶段故意打乱图片顺序并重复图片，输出仍与参考一致，说明结果依赖真实计算而非写死。

## 八、NumPy 练习题

`numpy_homework.py` 完成 `practice/project1_exercises.pdf` 中的 12 道题
（NP-E1~E8 基础，NP-M1~M4 中等），包含指定函数实现、给定数据上的运行结果与自检断言：

```bash
python numpy_homework.py
```

## 九、在后续大作业中的复用

完成后的 `image_encoder.py` 与课程提供的 `helpers.py` 可一并复制到最终 CLIP 项目目录，
与 `clip.py`、`text_encoder.py` 并列；`ImageEncoder(weights)`、`LayerNorm`、
`TransformerBlock(weights, prefix, heads)` 保持原有接口。最终项目使用完整的多模态权重，
不要用本项目仅含图像的权重替代。
