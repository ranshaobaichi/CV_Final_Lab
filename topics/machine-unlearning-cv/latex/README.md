# LaTeX 编译说明（CVPR 格式）

目录：`topics/machine-unlearning-cv/latex/`

## 依赖文件

| 文件 | 说明 |
|------|------|
| `main.tex` | 英文主稿 |
| `references.bib` | 12 条参考文献 |
| `cvpr.sty` | CVPR 官方样式 |
| `ieeenat_fullname.bst` | 数字引用样式 |
| `taxonomy-tree.png` / `research-timeline.png` | 图1、图2 |

## 编译命令（推荐）

在 `latex/` 目录下执行：

```bash
cd topics/machine-unlearning-cv/latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

或使用 `latexmk`：

```bash
cd topics/machine-unlearning-cv/latex
latexmk -pdf -interaction=nonstopmode main.tex
```

成功后得到 `main.pdf`。

## Overleaf

1. 新建项目，上传整个 `latex/` 文件夹  
2. 将主文件设为 `main.tex`  
3. 编译器选 **pdfLaTeX**  
4. 点击 Recompile

## Windows（TeX Live / MiKTeX）

在 PowerShell 中：

```powershell
cd topics\machine-unlearning-cv\latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 常见问题

- **找不到 cvpr.sty**：确认当前目录为 `latex/`，且 `cvpr.sty` 与 `main.tex` 同目录。  
- **引用显示为 `[?]`**：需完整跑完 `pdflatex → bibtex → pdflatex ×2`。  
- **图片缺失**：确认两个 PNG 在 `latex/` 目录内。  
- **页数偏多**：课程综述可接受；投稿时可删 Discussion 细节或缩小图宽。
