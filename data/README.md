# 数据目录

数据集被 `.gitignore` 排除，不随仓库发布。请从原发布方获取数据并遵循其适用条款，或使用自己的授权 AOI 数据。

## DAGM2007 演示结构

```text
data/dagm2007/
└── Class1/
    ├── Train/
    │   ├── 0001.PNG
    │   └── Label/
    └── Test/
        ├── 0001.PNG
        └── Label/
```

其他类别按同样结构放置，具体类别选择和样本规模以 `configs/train_dagm_demo.yaml` 为准。

原说明提供的归档与引用入口：[DAGM2007 数据记录](https://doi.org/10.5281/zenodo.12750201)。此链接不意味着本仓库授予数据再分发权。

## 自定义产品

参考 `configs/custom_train.example.yaml` 和 `configs/custom_test.example.yaml` 准备支持集、可选掩码及独立测试 manifest。训练、验证与测试边界需要保留，不能将测试数据混入支持集。

不要提交客户或生产图像、产品敏感图片、支持集、留出集和反馈采集内容。完整使用方法见[项目首页](../README.md)。
