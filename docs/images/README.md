# 异常检测可视化

准备获准使用的输入图像与兼容模型包后，从仓库根目录运行：

```bash
python -m scripts.generate_demo --input path/to/image.png --models outputs/dagm2007/active_models.yaml --category Class1 --device auto --output-dir docs/images
```

生成文件：

| 文件 | 内容 |
| --- | --- |
| `demo_original.png` | 输入图像副本 |
| `demo_anomaly_heatmap.png` | 真实模型输出的异常热力图 |
| `demo_result.png` | 检测叠加结果 |
| `demo_panel.png` | 三联对比面板 |
| `demo_result.yaml` | 判定、分数和运行记录 |

热力图不是标注真值、缺陷检测框或经过认证的分割掩码。生成图像与记录默认被 Git 忽略；发布前应确认图像再分发权限，并检查记录中的敏感路径。
