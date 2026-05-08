```markdown
# Endfield Credit Store OCR

从游戏截图（如明日方舟终末地信用商店）中自动提取商品信息、价格、折扣、剩余刷新次数和 UID。

本工具基于透视矫正 + PaddleOCR + 图标匹配，支持批量处理大量截图，输出结构化 JSON。

## 特性

- 自动检测商品卡片四边形，利用多卡片 RANSAC 整图矫正透视。
- 独立解析卡片主体（白色/灰色区域）和名称栏，避免 ROI 错位。
- 支持三种 OCR 模式：
  - `fast`：整图一次 OCR，只补充 UID/刷新次数。
  - `smart`：整图 OCR + 按需局部重识别（缺失名称栏/价格时）。
  - `full`：每张卡片独立裁剪 OCR（最准但较慢）。
- 对缺失名称栏的卡片，自动进行图标匹配（需提供参考图目录）。
- 流式批量处理（JSONL 输出），内存友好。
- 调试模式：保存每一步的可视化图像（矫正图、槽位框、OCR 结果），中文文字正常显示（需 Pillow）。

## 安装

```bash
uv sync
```

依赖项：
- opencv-python
- numpy
- paddleocr >=3.5.0
- paddlepaddle 3.2.0
- rapidfuzz
- pillow

## 快速开始

1. 将截图放入 `data/` 文件夹（支持 PNG/JPG/JPEG/BMP/WEBP）。
2. 修改 `main.py` 中的配置常量（可选）：
   - `INPUT_DIR`：图像目录
   - `OUTPUT_DIR`：输出目录
   - `REFS_DIR`：商品图标参考图目录（用于无名称栏的卡片匹配）
   - `MAX_INPUT_SIDE`：下采样边长
   - `OCR_MODE`：`fast` / `smart` / `full`
3. 运行：

```bash
uv run main.py
```

## 输出

- `output/results_stream.jsonl`：每行一个 JSON 结果（可中断恢复）。
- `output/results_final.json`：完整 JSON 数组。
- `output/failed_paths.txt`：处理失败的图像列表。
- `output/debug/`：若启用 `debug_save_dir`，保存可视化图像。

输出 JSON 结构示例：

```json
{
  "image_path": "data/shop.jpg",
  "uid": "1234567890",
  "refresh_remaining": 3,
  "refresh_total": 10,
  "items": [
    {
      "id": 0,
      "row": 0,
      "col": 0,
      "name": "折金票",
      "name_confidence": 0.96,
      "name_source": "ocr_namebar",
      "name_occluded": false,
      "price": 35,
      "original_price": 140,
      "price_panel_present": true,
      "discount_percent": 75,
      "quantity": 1000,
      "sold_out": false
    }
  ],
  "meta": { ... }
}
```

## 高级配置

可通过 `PipelineConfig` 调整 ROI、检测阈值、OCR 参数等。示例：

```python
from endfield_ocr.config import PipelineConfig, OCRConfig

config = PipelineConfig(
    ocr=OCRConfig(mode="smart", text_det_box_thresh=0.25),
    max_input_side=2000,
    debug_save_dir=Path("debug_out"),
)
```