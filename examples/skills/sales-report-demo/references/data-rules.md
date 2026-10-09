# 销售订单口径

输入为 UTF-8 CSV（可带 BOM），必须包含下列字段；允许其他列。

| 字段 | 含义与要求 |
| --- | --- |
| order_id | 非空订单编号，同一文件内唯一；一行对应一个商品订单 |
| date | 真实存在的日期，严格使用 YYYY-MM-DD |
| product | 非空商品名称 |
| quantity | 正整数，不超过 1000000 |
| unit_price | 人民币单价，非负，最多 9 位整数、2 位小数 |
| status | completed、cancelled 或 refunded |

## 统计规则

- 先校验整个文件；即使某行不在日期范围内，也不能掩盖无效输入。
- `--start`、`--end` 都是包含边界的日期筛选。
- 只统计 `completed`；筛选范围内的 `cancelled` 和 `refunded` 整单排除，不把退款写成负销售额。
- 销售额 = quantity × unit_price，使用十进制计算；不是利润，不包含税费、折扣或运费。
- 有效订单数为纳入统计的行数，销量为 quantity 合计。
- 商品按销售额降序排列；相同时按商品名称排序。按日汇总不受 `--top` 影响。
- 无有效订单时输出零值和空排名，不报成计算异常。

JSON 中 `orders_in_range` 是范围内全部订单数，`excluded_orders` 是其中被状态排除的数量；各 `revenue` 为保留两位小数的字符串，避免浮点误差。
