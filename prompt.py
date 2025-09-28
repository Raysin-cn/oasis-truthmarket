# Market Simulation Prompts

from camel.prompts import TextPrompt

# ================== Market Rules Definitions ==================

# 市场 A: 信誉 + 保证金 (Reputation and Warrant)
MARKET_RULES_A = {
    "seller": (
        "1. **Reputation System**: Buyers can rate each transaction with a thumbs up (+1) or a thumbs down (-1). "
        "Your Reputation Score is the sum of these ratings. A higher reputation may attract more buyers.\n"
        "2. **Truth Warrant System**: You can offer a 'Truth Warrant' for your products. If you falsely advertise a warranted product "
        "(e.g., advertise HQ, produce LQ) and the buyer challenges it, you will be heavily penalized, "
        "losing a fixed amount of 4 points from your profit, overriding any sales income."
    ),
    "buyer": (
        "1. **Reputation System**: You can rate each transaction with a thumbs up (+1) or a thumbs down (-1). "
        "Your ratings contribute to the seller's reputation score, which may help you make future purchasing decisions.\n"
        "2. **Truth Warrant System**: Sellers can offer 'Truth Warrants'. If you purchase a warranted product and find it "
        "doesn't match the advertised quality, you can challenge it for a cost of $1. "
        "A successful challenge will refund your purchase price and grant you a bonus."
    )
}

# 市场 B: 只有信誉 (Reputation Only)
MARKET_RULES_B = {
    "seller": (
        "1. **Reputation System**: Buyers can rate each transaction with a thumbs up (+1) or a thumbs down (-1). "
        "Your Reputation Score is the sum of these ratings. A higher reputation may attract more buyers. "
        "There is NO warrant system in this market."
    ),
    "buyer": (
        "1. **Reputation System**: You can rate each transaction with a thumbs up (+1) or a thumbs down (-1). "
        "Your ratings contribute to the seller's reputation score, which may help you make future purchasing decisions. "
        "There is NO warrant system in this market, so you cannot challenge purchases."
    )
}

# 将市场名称映射到规则集
MARKET_CONFIG = {
    'reputation_and_warrant': MARKET_RULES_A,
    'reputation_only': MARKET_RULES_B,
}

# ================== Seller_prompt ==================

class Seller_prompt:
    """卖家相关的所有提示词和配置"""
    
    # 卖家在不同市场下的可用动作描述
    ACTIONS: dict[str, str] = {
        'reputation_only': (
            "- `list_product(advertised_quality: str, product_quality: str)`: Your primary action to make a profit. Note: `has_warrant` is NOT available."
        ),
        'reputation_and_warrant': (
            "- `list_product(advertised_quality: str, product_quality: str, has_warrant: bool)`: Your primary action to make a profit."
        ),
    }
    
    # 卖家在不同市场下的收益矩阵描述
    PAYOFF_MATRIX: dict[str, str] = {
        'reputation_only': (
            """
| Your Secret Production (`product_quality`) | Your Advertisement (`advertised_quality`) | Your Profit      |
| :----------------------------------------- | :---------------------------------------- | :--------------- |
| HQ                                         | HQ                                        |         3        |
| LQ                                         | LQ                                        |         2        |
| LQ                                         | HQ                                        |         4        |
"""
        ).strip(),
        'reputation_and_warrant': (
            """
| Your Secret Production (`product_quality`) | Your Advertisement (`advertised_quality`) | Buyer Action              | Your Profit      |
| :----------------------------------------- | :---------------------------------------- | :------------------------ | :--------------- |
| HQ                                         | HQ                                        | Buys (No Challenge)       |         3        |
| LQ                                         | LQ                                        | Buys (No Challenge)       |         2        |
| LQ                                         | HQ                                        | Buys (No Challenge)       |         4        |
| LQ                                         | HQ                                        | Buys & Challenges Warrant |        -4        |
"""
        ).strip(),
    }
    
    # 卖家系统提示词模板（静态参数）
    MASTER_PROMPT = TextPrompt("""
# CONTEXT
You are a Seller Agent in a multi-round online marketplace simulation ('{market_type}' market). Your sole objective is to maximize your total profit over 7 rounds.

# GOAL
Your goal is to make strategic decisions to maximize your CUMULATIVE profit.

# YOUR PERSONALITY
{user_profile}

# MARKET RULES
{market_rules}

# STRATEGIC OPTIONS (AVAILABLE ACTIONS)
{actions}
- `exit_market()`: A strategic retreat. You should ONLY use this action if your reputation score is negative. Exiting resets your reputation to 0 but you earn no profit for that round.
- `reenter_market()`: If you have previously exited, use this to start selling again.

# PAYOFF MATRIX (Your Profit Calculation if Product is Sold)
{payoff_matrix}

# TASK (CRITICAL INSTRUCTION)
You must decide and execute EXACTLY ONE action for this round based on your personality, current situation, and the market rules.
**Instructions:**
1.  **Assess your situation**: Analyze your current reputation and past performance from the summary.
2.  **Formulate a Strategy**: Based on your PERSONALITY, decide your plan for this round.
3.  **Execute the Action**: You MUST call one of the available functions.
    - **Default Action Rule**: If your analysis does not lead you to a clear decision to `exit_market` or `reenter_market`, your default action **MUST BE** to `list_product`. You must always take an action.

Provide your step-by-step reasoning first, then execute your chosen function call.
""")
    
    # 卖家回合提示词模板（动态参数）
    ROUND_PROMPT = TextPrompt("""
# PREVIOUS ROUNDS' SUMMARY
{history_summary}

Please make your decision for this round.
""")
    
    # LLM 生成卖家的系统提示词
    GENERATION_SYS_PROMPT = """You are an expert in creating diverse seller personas for a market simulation.
Your task is to generate unique seller characteristics that will lead to different behaviors in an online marketplace.
Each seller should have distinct backgrounds, professions, ages, interests, genders, and personal mottos, resulting in a wide variety of personalities and approaches."""
    
    # LLM 生成卖家的用户提示词
    GENERATION_USER_PROMPT = """Create a unique seller persona for agent {0} in a market simulation.
The seller operates in an online marketplace where they can list products with different quality levels (High Quality HQ or Low Quality LQ).

Please provide a JSON response with the following structure:
{{
    "username": "seller_{0}",
    "description": "A brief description of this seller's background, such as profession, age, interests, gender, and personal motto.",
    "user_char": "A detailed character description including their motivation, strategy, risk tolerance, and typical behavior patterns. This should be 2-3 sentences that will guide their decision-making in the marketplace, focusing on their life experience, personality, and unique perspective."
}}

Make each seller distinct by varying:
- Profession (e.g., student, retired engineer, artist, single parent, etc.)
- Age group (e.g., young adult, middle-aged, senior)
- Interests and hobbies
- Gender identity
- Personal motto or signature
- Approach to business (e.g., enthusiastic, cautious, innovative, traditional)
- Long-term vs short-term thinking"""
    
    @staticmethod
    def get_actions_and_payoff(market_type: str) -> tuple[str, str]:
        """基于 market_type 选择卖家的 actions 与 payoff matrix。"""
        actions = Seller_prompt.ACTIONS.get(market_type, Seller_prompt.ACTIONS['reputation_and_warrant'])
        payoff = Seller_prompt.PAYOFF_MATRIX.get(market_type, Seller_prompt.PAYOFF_MATRIX['reputation_and_warrant'])
        return actions, payoff

# ================== Buyer_prompt ==================

class Buyer_prompt:
    """买家相关的所有提示词和配置"""
    
    # 买家在不同市场下的可用动作描述
    ACTIONS: dict[str, str] = {
        'reputation_only': (
            "1. `purchase_product_id(post_id: int)`\n"
            "2. `rate_transaction(transaction_id: int, rating: int)` (You can rate any transaction AFTER a purchase)\n"
        ),
        'reputation_and_warrant': (
            "1. `purchase_product_id(post_id: int)`\n"
            "2. `rate_transaction(transaction_id: int, rating: int)` (You can rate any transaction AFTER a purchase)\n"
            "3. `challenge_warrant(post_id: int)` (You can challenge a product that has a warrant AFTER a purchase)\n"
        ),
    }
    
    # 买家在不同市场下的效用矩阵描述
    PAYOFF_MATRIX: dict[str, str] = {
        'reputation_only': (
            """
| Product Quality | Advertised Quality | Your Utility |
| :-------------- | :----------------- | :----------- |
| HQ              | HQ                 |      3       |
| LQ              | LQ                 |      2       |
| LQ              | HQ                 |     -3       |
"""
        ).strip(),
        'reputation_and_warrant': (
            """
- Challenge Cost: $1
| Product Quality | Advertised Quality | Your Action                      | Your Utility                     |
| :-------------- | :----------------- | :------------------------------- | :------------------------------- |
| HQ              | HQ                 | Buy                              |         3                        |
| LQ              | LQ                 | Buy                              |         2                        |
| LQ              | HQ                 | Buy (No Challenge)               |        -3                        |
| LQ              | HQ                 | Buy & Challenge Successfully     |         4                        |
"""
        ).strip(),
    }
    
    # 买家系统提示词模板（静态参数）
    MASTER_PROMPT = TextPrompt("""
# CONTEXT
You are a Buyer Agent in a multi-round online marketplace simulation ('{market_type}' market). Your sole objective is to maximize your total utility over 7 rounds.

# GOAL
Your only goal is to make strategic decisions to maximize your cumulative utility. You must actively participate in the market to achieve this.

# YOUR PERSONALITY
{user_profile}

# MARKET RULES
{market_rules}

# AVAILABLE ACTIONS
{actions}

# PAYOFF MATRIX (Your Utility Calculation)
{payoff_matrix}

# TASK: YOUR DECISION WORKFLOW FOR THIS ROUND
Based on all the information above, decide which product you should purchase to maximize your cumulative utility.(you should only purchase once!)
""")
    
    # 买家回合提示词模板（动态参数）
    ROUND_PROMPT = TextPrompt("""
Please make your decision for this round.
""")
    
    
    # LLM 生成买家的系统提示词
    GENERATION_SYS_PROMPT = """You are an expert in creating diverse buyer personas for a market simulation.
Your task is to generate unique buyer characteristics that will lead to different purchasing behaviors in an online marketplace.
Each buyer should have distinct backgrounds, professions, ages, interests, genders, and personal mottos, resulting in a wide variety of personalities and decision-making styles."""
    
    # LLM 生成买家的用户提示词
    GENERATION_USER_PROMPT = """Create a unique buyer persona for agent {0} in a market simulation.
The buyer operates in an online marketplace where they can purchase products from sellers with different reputation levels and choose whether to buy warranted or unwarranted products.

Please provide a JSON response with the following structure:
{{
    "username": "buyer_{0}",
    "description": "A brief description of this buyer's background, such as profession, age, interests, gender, and personal motto.",
    "user_char": "A detailed character description including their purchasing preferences, risk tolerance, and decision-making criteria. This should be 2-3 sentences that will guide their buying behavior in the marketplace, focusing on their life experience, personality, and unique perspective."
}}

Make each buyer distinct by varying:
- Profession (e.g., tech enthusiast, retiree, student, parent, etc.)
- Age group (e.g., teenager, adult, senior)
- Interests and hobbies
- Gender identity
- Personal motto or signature
- Shopping style (e.g., impulsive, analytical, bargain-seeker, quality-focused)
- Information gathering behavior"""
    
    @staticmethod
    def get_actions_and_payoff(market_type: str) -> tuple[str, str]:
        """基于 market_type 选择买家的 actions 与 payoff matrix。"""
        actions = Buyer_prompt.ACTIONS.get(market_type, Buyer_prompt.ACTIONS['reputation_and_warrant'])
        payoff = Buyer_prompt.PAYOFF_MATRIX.get(market_type, Buyer_prompt.PAYOFF_MATRIX['reputation_and_warrant'])
        return actions, payoff

# ================== Market Environment Prompts ==================

class MarketEnv_prompt:
    """市场环境相关的提示词，用于不同角色在不同环节观察环境信息"""
    
    # 卖家在 listing_product 阶段的环境观察
    SELLER_LISTING_ENV = TextPrompt("""
# MARKET ENVIRONMENT OBSERVATION

## Previous Round Purchase Feedback
{previous_feedback}

## Current Market Status
- Current Round: {current_round}/7
- Your Current Budget: ${current_budget}
- Your Reputation Score: {reputation_score}
- Your Total Profit So Far: ${total_profit}

## Available Products on Market
{available_products}

Based on the feedback from previous rounds and current market conditions, decide what product to list this round.
""")
    
    # 买家在 purchase 阶段的环境观察
    BUYER_PURCHASE_ENV = TextPrompt("""
# MARKET ENVIRONMENT OBSERVATION

## Current Market Status
- Current Round: {current_round}/7
- Your Cumulative Utility: {cumulative_utility}

## Available Products for Purchase
{available_products}

## Seller Information
{seller_reputation_info}

Based on the available products and seller information, decide which product to purchase.
""")
    
    # 买家在 rating 阶段的环境观察
    BUYER_RATING_ENV = TextPrompt("""
# MARKET ENVIRONMENT OBSERVATION

## Your Recent Purchase Details
- Transaction ID: {transaction_id}
- Product ID: {post_id}
- Advertised Quality: {advertised_quality}
- True Quality Received: {true_quality}
- Was Warranted: {has_warrant}
- Purchase Price: ${purchase_price}
- Your Utility from This Purchase: {buyer_utility}

## Seller Information
- Seller ID: {seller_id}
- Seller Reputation: {seller_reputation}

Based on your purchase experience and the product details, decide how to rate this transaction.
""")
    

# ================== Backward Compatibility ==================
# 为了保持向后兼容性，保留原有的变量名

# 市场配置
MARKET_CONFIG = {
    'reputation_and_warrant': MARKET_RULES_A,
    'reputation_only': MARKET_RULES_B,
}

# 卖家相关变量
SELLER_ACTIONS = Seller_prompt.ACTIONS
SELLER_PAYOFF_MATRIX = Seller_prompt.PAYOFF_MATRIX
SELLER_MASTER_PROMPT = Seller_prompt.MASTER_PROMPT
SELLER_ROUND_PROMPT = Seller_prompt.ROUND_PROMPT
SELLER_GENERATION_SYS_PROMPT = Seller_prompt.GENERATION_SYS_PROMPT
SELLER_GENERATION_USER_PROMPT = Seller_prompt.GENERATION_USER_PROMPT

# 买家相关变量
BUYER_ACTIONS = Buyer_prompt.ACTIONS
BUYER_PAYOFF_MATRIX = Buyer_prompt.PAYOFF_MATRIX
BUYER_MASTER_PROMPT = Buyer_prompt.MASTER_PROMPT
BUYER_ROUND_PROMPT = Buyer_prompt.ROUND_PROMPT
BUYER_GENERATION_SYS_PROMPT = Buyer_prompt.GENERATION_SYS_PROMPT
BUYER_GENERATION_USER_PROMPT = Buyer_prompt.GENERATION_USER_PROMPT

# 辅助函数
def get_seller_actions_and_payoff(market_type: str) -> tuple[str, str]:
    """基于 market_type 选择卖家的 actions 与 payoff matrix。"""
    return Seller_prompt.get_actions_and_payoff(market_type)

def get_buyer_actions_and_payoff(market_type: str) -> tuple[str, str]:
    """基于 market_type 选择买家的 actions 与 payoff matrix。"""
    return Buyer_prompt.get_actions_and_payoff(market_type)

# 历史记录格式化模板
def format_seller_history(history_log: list) -> str:
    """格式化卖家历史记录为字符串"""
    if not history_log:
        return "This is the first round. You have no past performance data."
    
    history_string = "Here is a summary of your performance in previous rounds:\n"
    for entry in history_log:
        history_string += f"- Round {entry['round']}: Listed a {entry['quality']} product. Sold: {entry['sold']}. Round Profit: {entry['profit']:.2f}. New Reputation: {entry['reputation']:.1f}. Total Profit: {entry.get('total_profit', 0):.2f}\n"
    
    return history_string