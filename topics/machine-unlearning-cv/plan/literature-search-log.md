# 文献检索记录

## 检索信息

- 检索日期：2026-07-12
- 检索工具：`research-writing-skill/scripts/scholar_search.py`
- 数据源：CrossRef、arXiv
- 年份范围：2020–2026

## 检索式

| 序号 | 检索式 | 结果数 | 输出文件 |
|------|--------|--------|----------|
| 1 | machine unlearning computer vision | 35 | refs/search-results.json |
| 2 | concept erasure diffusion stable diffusion | 27 | refs/search-concept-erasure.json |
| 3 | machine unlearning computer vision (bibtex) | 43 | refs/refs.bib |

## 手工补充来源

- CVPRW 2026 MUV Workshop 系统综述（Safavigerdini 等）
- ICCV 2025 HUB 基准（Moon 等）
- arXiv:2605.26992 VLM 鲁棒性（Lin 等）
- arXiv:2602.20114 ViT 遗忘基准（Zhao 等）

## 纳入标准

1. 与计算机视觉机器遗忘直接相关
2. 发表于 CVPR、ICCV、NeurIPS、IEEE S&P、WACV 或 arXiv（2020 年后）
3. 提供可验证 DOI 或 arXiv ID
4. 对分类、方法、基准或鲁棒性有实质贡献

## 排除标准

1. 纯 NLP/LLM 遗忘且未涉及视觉
2. 公众号、知乎、中文博客
3. 无法追溯出处的二次引用
4. 与视觉遗忘关联过弱的通用 MU 综述（仅作背景引用）

## 去重说明

CrossRef 与 arXiv 结果按标题相似度去重；锚点论文与脚本结果合并后共保留 30 条核心文献进入 evidence-map。
