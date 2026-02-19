# 作文答题卡图片批处理（拼接 + 去干扰 + OCR）

## 功能
- 按文件名 `原始得分_姓名_学号(第几张).jpg` 解析并分组。
- 同一学生 1-2 张图片按页码 `(1)->(2)` 纵向拼接（自动补白对齐宽度）。
- OCR 前干扰规避：
  - 红色/高饱和顶部标注遮罩（用于去除姓名与分数叠加信息）。
  - 右下角固定区域遮罩（用于去除模板后缀，如 `Yours, Li Hua`）。
- 图像轻度增强与二值化后执行 OCR。
- 导出 `results.xlsx`：`原始得分, 姓名, 学号, 提取文本`。

## 依赖安装
```bash
pip install opencv-python pytesseract openpyxl numpy
```

并确保系统安装 `tesseract` 以及所需语言包（默认 `chi_sim+eng`）。

## 使用方式
```bash
python process_cards.py <输入目录> --output results.xlsx
```

可选参数：
- `--ocr-lang`：OCR 语言包，默认 `chi_sim+eng`
- `--suffix-mask-width-ratio`：右下角遮罩宽度比例，默认 `0.25`
- `--suffix-mask-height-ratio`：右下角遮罩高度比例，默认 `0.25`
- `--debug-dir`：输出中间图（拼接图、遮罩图、OCR 输入图）

示例：
```bash
python process_cards.py ./images --output ./results.xlsx --debug-dir ./debug
```
