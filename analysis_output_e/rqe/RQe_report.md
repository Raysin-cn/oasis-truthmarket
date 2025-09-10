## RQe 报告（简版）

本报告汇总一次模拟的简化问题（RQe1–RQe4）统计与可视化结果，并结合卖家/买家 personas 进行简要解读。

### 实验设定（简述）
- 卖家 personas：
  - Honest Seller（seller_id=0，对应 `agent_id=0`）：始终 HQ 并如实广告，重视声誉与长期利润。
  - Deceptive Seller（seller_id=1，`agent_id=1`）：偏好短期利润，常将 LQ 广告为 HQ。
  - Strategic Seller（seller_id=2，`agent_id=2`）：在长期与短期间权衡，条件合适时可能欺骗。
- 买家 personas：
  - Pragmatic（`buyer_id`≈3）：看重效用，愿意在有 warrant 时尝试新卖家。
  - Analytical（`buyer_id`≈4）：以效用最大化为目标，理性购买。
  - Strategic（`buyer_id`≈5）：重视评分以提高市场信息质量。

注：以上 id 映射按本次数据写作时观测（见 CSV），不同仿真可能略有差异。

---

### RQe1 卖家不诚实与最终 Reputation 的关系
- 数据表：`analysis_output_e/rqe/rqe1_seller_dishonesty_vs_reputation.csv`
- 相关散点图：`rqe1_scatter_dishonesty_vs_reputation.png`

核心摘要：
- seller_id=1（Deceptive）不诚实广告率=1.0，对应较低/负面的声誉分（reputation_score=-5）。
- seller_id=0（Honest）不诚实广告率=0，对应较高声誉（4）。
- seller_id=2（Strategic）不诚实广告率=0，对应更高声誉（6）。

初步结论：不诚实程度与声誉呈显著负相关迹象（越不诚实声誉越差）。

---

### RQe2 四类商品的 rating 均值与标准差
- 表：`analysis_output_e/rqe/rqe2_post_type_rating_stats.csv`
- 图：`rqe2_rating_mean_by_type.png`

观测：
- HQ->HQ：平均评分约 1.0；
- LQ->HQ（欺骗广告）：平均评分约 -1.0；

当前样本中未出现 HQ->LQ 与 LQ->LQ（可能因为策略和市场机制设置）。

---

### RQe3 三个卖家的欺骗率与 warrant 概况
- 表：`analysis_output_e/rqe/rqe3_top3_seller_deception_and_warrant.csv`
- 图：`rqe3_sellers_dishonest_ad_rate.png`, `rqe3_sellers_warrant_rate.png`

结果摘录：

| seller_id | listings | dishonest_ads | warrant_rate | dishonest_ad_rate | sales | dishonest_sales | dishonest_sale_rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 0 | 1.0 | 0.0 | 3 | 0 | 0.0 |
| 1 | 5 | 5 | 0.0 | 1.0 | 5 | 5 | 1.0 |
| 2 | 5 | 0 | 0.0 | 0.0 | 5 | 0 | 0.0 |

列含义说明：
- seller_id：卖家用户ID。
- listings：该卖家上架的商品帖子数。
- dishonest_ads：上架中“LQ→HQ”欺骗广告的数量。
- warrant_rate：上架商品中带质保（has_warrant=1）的比例。
- dishonest_ad_rate：欺骗广告占比，dishonest_ads/listings。
- sales：成交的交易数量（以交易为单位）。
- dishonest_sales：成交中“LQ→HQ”的欺骗销售数量（基于交易×post判定）。
- dishonest_sale_rate：欺骗销售占比，dishonest_sales/sales。

解读：
- Honest（0）：全部带质保（warrant_rate=1），无欺骗，销售反馈良好；
- Deceptive（1）：全部欺骗且无质保，销售均为欺骗销售；
- Strategic（2）：未欺骗但也未使用质保（可能依赖声誉/价格等建立信任）。

---

### RQe4 三个买家的 rating 与其购买的 warrant 概况
- 表：`analysis_output_e/rqe/rqe4_top3_buyer_rating_and_warrant.csv`
- 图：`rqe4_buyers_rating_mean.png`, `rqe4_buyers_warrant_rate.png`

摘录：

| buyer_id | n_tx | rating_mean | rating_std | warrant_rate |
|---:|---:|---:|---:|---:|
| 3 | 4 | 0.5 | 1.0 | 0.5 |
| 4 | 5 | -0.2 | 1.095 | 0.2 |
| 5 | 4 | 0.5 | 1.0 | 0.0 |

列含义说明：
- buyer_id：买家用户ID。
- n_tx：该买家的交易次数。
- rating_mean：该买家给出的评分均值（交易层面）。
- rating_std：该买家评分的标准差。
- warrant_rate：该买家购买的商品中带质保的比例。

解读：
- Pragmatic/Strategic 买家（3、5）平均评分较高（0.5），其中 3 的 warrant_rate=0.5；
- Analytical 买家（4）平均评分略低（-0.2），其购买中质保比例也最低（0.2），可能更倾向于无质保的“高预期效用”尝试，遭遇更高风险。

---

### 综合小结与展望
- 卖家层面：人格设定与行为吻合。欺骗型卖家积累负面声誉；诚实/战略型卖家保持正面声誉与更稳定评分。
- 商品层面：LQ->HQ 的欺骗广告收到系统性负面评分；HQ->HQ 获得正面评分。
- 买家层面：不同 persona 与质保偏好影响其评分分布；购买质保比例较高的买家往往评分更稳定/更高。

建议：在后续“大 RQ”中，分情景（有/无金融惩罚）对比以上指标，并纳入程序回合维度的动态趋势，以检验机制设计对欺骗与不平等的影响。
