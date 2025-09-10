## RQe Report (Concise)

This report summarizes simplified questions (RQe1–RQe4) from one simulation run, with brief interpretations aligned to seller/buyer personas.

### Experiment Setup (Brief)
- Seller personas:
  - Honest Seller (seller_id=0, `agent_id=0`): always HQ and truthful advertising; prioritizes reputation and long-term profit.
  - Deceptive Seller (seller_id=1, `agent_id=1`): short-term profit oriented; often advertises LQ as HQ.
  - Strategic Seller (seller_id=2, `agent_id=2`): balances short vs long-term; may deceive under suitable conditions.
- Buyer personas:
  - Pragmatic (approx `buyer_id`=3): values utility; willing to try new sellers if warrant is present.
  - Analytical (approx `buyer_id`=4): aims to maximize utility; rational purchasing.
  - Strategic (approx `buyer_id`=5): values rating to improve market information quality.

Note: The id mapping reflects the present run (see CSV); it may vary across simulations.

---

### RQe1 Relationship Between Seller Dishonesty and Final Reputation
- Table: `analysis_output_e/rqe/rqe1_seller_dishonesty_vs_reputation.csv`
- Scatter: `rqe1_scatter_dishonesty_vs_reputation.png`

Key observations:
- seller_id=1 (Deceptive) dishonest ad rate = 1.0, with low/negative reputation (reputation_score = -5).
- seller_id=0 (Honest) dishonest ad rate = 0, higher reputation (4).
- seller_id=2 (Strategic) dishonest ad rate = 0, even higher reputation (6).

Preliminary conclusion: dishonesty appears negatively associated with reputation (more dishonesty → worse reputation).

---

### RQe2 Average and Std of Ratings for Four Post Types
- Table: `analysis_output_e/rqe/rqe2_post_type_rating_stats.csv`
- Figure: `rqe2_rating_mean_by_type.png`

Findings:
- HQ->HQ: mean rating ≈ 1.0
- LQ->HQ (deceptive advertising): mean rating ≈ -1.0

In this sample we did not observe HQ->LQ or LQ->LQ (likely due to strategies and market mechanism).

---

### RQe3 Deception and Warrant Profiles for Top 3 Sellers
- Table: `analysis_output_e/rqe/rqe3_top3_seller_deception_and_warrant.csv`
- Figures: `rqe3_sellers_dishonest_ad_rate.png`, `rqe3_sellers_warrant_rate.png`

Excerpt:

| seller_id | listings | dishonest_ads | warrant_rate | dishonest_ad_rate | sales | dishonest_sales | dishonest_sale_rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 0 | 1.0 | 0.0 | 3 | 0 | 0.0 |
| 1 | 5 | 5 | 0.0 | 1.0 | 5 | 5 | 1.0 |
| 2 | 5 | 0 | 0.0 | 0.0 | 5 | 0 | 0.0 |

Column notes:
- seller_id: seller user ID.
- listings: number of product posts listed by the seller.
- dishonest_ads: number of deceptive ads (LQ→HQ) among listings.
- warrant_rate: share of listed items that include a warranty (has_warrant=1).
- dishonest_ad_rate: dishonest_ads / listings.
- sales: number of completed transactions.
- dishonest_sales: number of deceptive sales (LQ→HQ) among transactions (via tx×post join).
- dishonest_sale_rate: dishonest_sales / sales.

Interpretation:
- Honest (0): full warranty usage (warrant_rate = 1), no deception, strong sales feedback.
- Deceptive (1): fully deceptive and no warranty; all sales are deceptive sales.
- Strategic (2): no deception but also no warranty (may rely on reputation/price to build trust).

---

### RQe4 Rating and Warranty Profiles for Top 3 Buyers
- Table: `analysis_output_e/rqe/rqe4_top3_buyer_rating_and_warrant.csv`
- Figures: `rqe4_buyers_rating_mean.png`, `rqe4_buyers_warrant_rate.png`

Excerpt:

| buyer_id | n_tx | rating_mean | rating_std | warrant_rate |
|---:|---:|---:|---:|---:|
| 3 | 4 | 0.5 | 1.0 | 0.5 |
| 4 | 5 | -0.2 | 1.095 | 0.2 |
| 5 | 4 | 0.5 | 1.0 | 0.0 |

Column notes:
- buyer_id: buyer user ID.
- n_tx: number of transactions made by the buyer.
- rating_mean: average rating given by the buyer (transaction-level).
- rating_std: standard deviation of that buyer’s ratings.
- warrant_rate: share of purchased items that include warranty.

Interpretation:
- Pragmatic/Strategic buyers (3, 5) have higher average ratings (0.5); buyer 3 has higher warranty rate (0.5).
- Analytical buyer (4) shows slightly lower average rating (-0.2) and the lowest warranty share (0.2), indicating more exposure to risk by choosing fewer warranty-backed purchases.

---

### Summary and Outlook
- Sellers: personas align with behaviors. Deceptive sellers accumulate negative reputation; honest/strategic sellers maintain positive reputation and more stable ratings.
- Products: deceptive ads (LQ->HQ) receive systematically negative ratings; HQ->HQ receives positive ratings.
- Buyers: persona and warranty preference shape rating distributions; higher warranty usage tends to correlate with more stable/higher ratings.

Recommendation: For the larger RQs, compare metrics across scenarios (with/without financial penalties) and add program-round dynamics to test whether mechanism design shifts deception and inequality over time.
