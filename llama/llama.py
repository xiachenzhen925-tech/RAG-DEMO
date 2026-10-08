from pathlib import Path
import sys

from llama_index.readers.file import PDFReader

# 让 Windows 终端能够显示 PDF 中的非 GBK 字符。
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# PDF 位于项目根目录；通过脚本路径定位，避免受当前命令行目录影响。
project_dir = Path(__file__).resolve().parent.parent
pdf_files = list(project_dir.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(f"未在 {project_dir} 找到 PDF 文件")

pdf_reader = PDFReader()
documents = []
for pdf_file in pdf_files:
    documents.extend(pdf_reader.load_data(file=pdf_file))

print(f"读取到 PDF: {', '.join(pdf_file.name for pdf_file in pdf_files)}")
print(f"文档片段数量: {len(documents)}")

pdf_text = "\n".join(document.text for document in documents).strip()
if not pdf_text:
    raise RuntimeError(
        "PDF 没有可提取的文本层，应该是扫描件。请先对 PDF 执行 OCR，"
        "或为脚本配置 OCR 读取器后再建立索引。"
    )

print(pdf_text[:500])  # 预览前 500 个字符
