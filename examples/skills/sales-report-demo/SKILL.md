---
name: sales-report-demo
description: >-
  分析销售订单 CSV，统计有效订单数、销量、销售额、按日汇总和商品排名。
  用户需要销售报表或希望用内置示例测试技能脚本时使用；不用于股票分析或任意格式的 Excel 文件。
metadata:
  version: "1.0"
  runtime: "Python 3.10+，只使用标准库，不需要 pip 安装"
---

# 销售报表演示

根据实际 CSV 生成销售小结，不修改输入文件、不联网。

## 按需使用资源

- 开始分析前读取 [references/data-rules.md](references/data-rules.md)，确认字段和订单状态口径。
- 使用 `scripts/analyze.py` 进行确定性计算。第一次使用或参数不明确时，先运行 `python <技能目录>/scripts/analyze.py --help`，通常不必读取脚本源码。
- 用户要求“测试一下”且没有提供文件时，使用 [assets/sample-sales.csv](assets/sample-sales.csv)。分析真实业务数据时不要用示例替代。

资源路径以本技能目录为基准。解释器使用环境中实际可用的 Python；RAgent.exe 本身不是 Python 解释器。缺少解释器时说明缺失项，不自动安装。

## 执行和回答

1. 确认输入文件；日期范围不明确时默认统计全部日期，不自行猜测筛选条件。
2. 通过已有命令工具执行脚本，遵守当前审批和沙箱；导入技能不代表授权执行。
3. 根据脚本实际返回的 JSON，说明订单数、销量、销售额、排除订单数量和商品排名。金额单位为人民币元，日期范围包含首尾两天。
4. 输入无效或执行失败时展示具体原因，不编造统计结果；只在已明确原因且仍符合用户任务时修正参数重试。

调用示例（将路径换成当前技能目录与用户输入）：

```text
python <技能目录>/scripts/analyze.py --input <CSV路径> --top 3
python <技能目录>/scripts/analyze.py --input <CSV路径> --start 2026-10-01 --end 2026-10-02
```

没有传 `--input` 时，脚本使用随包携带的示例 CSV；默认只输出到终端，不创建报告文件。

本包附带 `tests/smoke.json`。导入后请在技能详情展开“发布、灰度与冒烟测试”，确认沙箱测试，通过后点击“保存发布策略”；发布前不会向新任务提供此版本。
