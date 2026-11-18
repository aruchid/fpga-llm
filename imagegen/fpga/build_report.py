"""把 REPORT_32MATRIX.md 转成自包含的 HTML（图片内嵌），再用浏览器打印成 PDF。

用法: python build_report.py
"""

import base64
import re
import subprocess
from datetime import datetime
from pathlib import Path

import markdown

ROOT = Path(r"E:\FPGALLM\imagegen")
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--md", default="REPORT_32MATRIX.md")
_ap.add_argument("--name", default="report_32matrix")
_args = _ap.parse_args()
MD = ROOT / "fpga" / _args.md
HTML = ROOT / "out" / f"{_args.name}.html"
PDF = ROOT / "out" / f"{_args.name}.pdf"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
  font-size: 10.5pt; line-height: 1.65; color: #1a1a1a; margin: 0;
}
.hero {
  border-left: 6px solid #2f6feb; padding: 10px 0 10px 16px; margin-bottom: 22px;
}
.hero h1 { font-size: 21pt; margin: 0 0 6px 0; line-height: 1.25; }
.hero .meta { color: #666; font-size: 9.5pt; }
h2 {
  font-size: 14pt; margin: 22px 0 10px; padding-bottom: 4px;
  border-bottom: 1.5px solid #dfe3e8; page-break-after: avoid;
}
h3 { font-size: 11.5pt; margin: 16px 0 8px; page-break-after: avoid; }
p { margin: 7px 0; }
table {
  border-collapse: collapse; width: 100%; margin: 10px 0 14px;
  font-size: 9.5pt; page-break-inside: auto;
}
tr { page-break-inside: avoid; }
thead { display: table-header-group; }
th, td { border: 1px solid #d5dae0; padding: 5px 7px; text-align: left; vertical-align: top; }
th { background: #f2f5f9; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px;
       font-family: Consolas, "Cascadia Mono", monospace; font-size: 9.5pt; }
pre { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 5px;
      padding: 10px 12px; overflow-x: auto; page-break-inside: avoid; }
.hero { page-break-after: avoid; }
pre code { background: none; padding: 0; font-size: 9pt; line-height: 1.45; }
blockquote {
  margin: 10px 0; padding: 8px 14px; background: #eef4ff;
  border-left: 4px solid #2f6feb; color: #24303f;
}
blockquote p { margin: 3px 0; }
img { max-width: 100%; display: block; margin: 12px auto; border: 1px solid #e3e7ec; border-radius: 4px; }
hr { border: none; border-top: 1px solid #e5e8ec; margin: 18px 0; }
ul, ol { margin: 7px 0 7px 18px; padding: 0; }
li { margin: 3px 0; }
strong { color: #0f1b2d; }
"""


def inline_images(html: str) -> str:
    """把 <img src="E:\\...png"> 换成 base64，保证 HTML 自包含。"""

    def repl(m):
        path = Path(m.group(1))
        if not path.exists():
            print(f"  缺图: {path}")
            return m.group(0)
        data = base64.b64encode(path.read_bytes()).decode()
        ext = path.suffix.lstrip(".").lower()
        mime = "image/png" if ext == "png" else f"image/{ext}"
        print(f"  内嵌 {path.name} ({len(data)//1024} KB base64)")
        return f'src="data:{mime};base64,{data}"'

    return re.sub(r'src="([A-Za-z]:\\[^"]+\.(?:png|jpg|jpeg))"', repl, html)


def main() -> int:
    md = MD.read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    body = inline_images(body)

    # 标题抽出来做封面
    m = re.search(r"<h1>(.*?)</h1>", body, re.S)
    title = m.group(1) if m else "SD-Turbo 推理改造报告"
    if m:
        body = body.replace(m.group(0), "", 1)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head>
<body>
<div class="hero">
  <h1>{title}</h1>
  <div class="meta">生成时间：{stamp} ｜ 模型：SD-Turbo (SD 2.1) ｜ 平台：RTX 5060 / i5-12600KF</div>
</div>
{body}
</body></html>"""

    HTML.parent.mkdir(parents=True, exist_ok=True)
    HTML.write_text(html, encoding="utf-8")
    print(f"[html] -> {HTML}  ({HTML.stat().st_size/1024:.0f} KB)")

    if not Path(EDGE).exists():
        print("找不到 Edge")
        return 1
    cmd = [EDGE, "--headless=new", "--disable-gpu", "--no-sandbox",
           f"--print-to-pdf={PDF}", "--no-pdf-header-footer",
           "--virtual-time-budget=10000", HTML.as_uri()]
    print("[pdf] 打印中…")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not PDF.exists() or PDF.stat().st_size < 20000:
        print("PDF 生成失败", r.stdout[-800:], r.stderr[-800:])
        return 1
    print(f"[pdf] -> {PDF}  ({PDF.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
