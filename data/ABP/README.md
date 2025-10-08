---
license: cc-by-4.0
language:
- en
tags:
- proactive agent planning
pretty_name: Ask-before-Plan
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.csv
      - split: test
        path: test.csv
---
# Ask-before-Plan Dataset

<a href="https://arxiv.org/abs/2406.12639">Paper</a> •
<a href="https://github.com/magicgh/Ask-before-Plan">Code</a> •
<a href="https://drive.google.com/file/d/1vMIhs8mpMgk33pFDv2rWg6AJNyD70Sod">Environment</a> •
<a href="https://huggingface.co/magicgh/CEP">Checkpoints</a>

The Ask-before-Plan dataset is crafted for integrating uncertain user instructions that require clarifications into real-world travel planning scenarios, building upon the [TravelPlanner Dataset](https://huggingface.co/datasets/osunlp/TravelPlanner).




## Citation
If you find our research helpful for your work, please star [this repository](https://github.com/magicgh/Ask-before-Plan) and cite our paper:
```
@article{ask-before-plan,
    author = {Xuan Zhang and Yang Deng and Zifeng Ren and See-Kiong Ng and Tat-Seng Chua},
    journal = {ArXiv preprint},
    title = {Ask-before-Plan: Proactive Language Agents for Real-World Planning},
    url = {https://arxiv.org/abs/2406.12639},
    year = {2024}
}
```
