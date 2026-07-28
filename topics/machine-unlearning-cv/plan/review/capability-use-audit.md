# 能力使用审计（Capability-Use Audit）

- 任务：按 research-writing-skill 修复机器遗忘综述
- 日期：2026-07-12

## 应使用技能

| 技能 | 是否使用 | 说明 |
|------|----------|------|
| using-research-writing | 是 | 路由至文献综述与写作规范 |
| literature-review | 是 | 检索、evidence map、综合写作 |
| writing-core | 是 | 去列表化、去加粗、段落规范 |
| verification | 是 | style_check + 字数统计 |
| brainstorming-research | 部分 | 用户已确认选题，补录于 project-overview |
| paper-orchestration | 是 | 多章节修复任务编排 |
| latex-output | 否 | 留待下一阶段 |

## 已消费资料

- research-writing-skill/SKILL.md
- skills/literature-review/SKILL.md
- skills/writing-core/SKILL.md
- skills/verification/SKILL.md
- plan-template/*
- refs/search-results.json, refs/refs.bib
- CVPRW 2026 / ICCV 2025 锚点论文摘要

## 产物

- plan/（project-overview, outline, stage-gates, progress, literature-search-log）
- refs/evidence-map.md, refs/refs.bib
- survey_zh.md（v2 重写版）

## 验证命令

```bash
bash research-writing-skill/scripts/style_check.sh topics/machine-unlearning-cv/survey_zh.md
python3 -c "import re; t=open('topics/machine-unlearning-cv/survey_zh.md').read(); print(len(re.findall(r'[\u4e00-\u9fff]',t)))"
```

## 剩余风险

1. 部分 2026 arXiv 预印本未经同行评审
2. 图表尚未制作，分类树仅文字描述
3. 英文 LaTeX 版未开始
