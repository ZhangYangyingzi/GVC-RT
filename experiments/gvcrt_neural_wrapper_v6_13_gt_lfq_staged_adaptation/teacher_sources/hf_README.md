---
license: apache-2.0
language:
- en
tags:
- Autoregressive
- Tokenizer
---

## Open-MAGVIT2: Democratizing Autoregressive Visual Generation

Code: https://github.com/TencentARC/SEED-Voken

Paper: https://arxiv.org/abs/2409.04410

## Introduction

Until now, VQGAN, the initial tokenizer is still acting an indispensible role in mainstream tasks, especially autoregressive visual generation. Limited by the bottleneck of the size of codebook and the utilization of code, the capability of AR generation with VQGAN is underestimated.

Therefore, [MAGVIT2](https://arxiv.org/abs/2310.05737) proposes a powerful tokenizer for visual generation task, which introduces a novel LookUpFree technique when quantization and extends the size of codebook to $2^{18}$, exhibiting promising performance in both image and video generation tasks. And it plays an important role in the recent state-of-the-art AR video generation model [VideoPoet](https://arxiv.org/abs/2312.14125). However, we have no access to this strong tokenizer so far. ☹️

In the codebase, we follow the significant insights of tokenizer design in MAGVIT-2 and re-implement it with Pytorch, achieving the closest results to the original so far. We hope that our effort can foster innovation, creativity within the field of Autoregressive Visual Generation. 😄
