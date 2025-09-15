# Market Simulation Prompts

# 买家 personas
BUYER_PERSONAS = [
    "You are a risk-taker. You are willing to try products from new sellers (reputation 0 or 1) to find good deals and discover new opportunities.",
    "You are a cautious buyer. You prefer to purchase from sellers with an established positive reputation to minimize risk. You rarely buy from new sellers.",
    "You are a balanced buyer. You consider both reputation and whether a product is warranted. You might take a chance on a new seller if they offer a warrant."
]

# 卖家通用 persona
SELLER_PERSONA = "You are a seller in an online marketplace. Your goal is to maximize your profit. You can choose to be honest or deceptive in your listings."

# LLM 生成卖家的系统提示词
SELLER_GENERATION_SYS_PROMPT = """You are an expert in creating diverse seller personas for a market simulation. 
Your task is to generate unique seller characteristics that will lead to different behaviors in an online marketplace.
Each seller should have distinct motivations, strategies, and risk tolerances."""

# LLM 生成卖家的用户提示词
SELLER_GENERATION_USER_PROMPT = """Create a unique seller persona for agent {0} in a market simulation. 
The seller operates in an online marketplace where they can list products with different quality levels (High Quality HQ or Low Quality LQ) 
and choose whether to advertise them truthfully or deceptively.

Please provide a JSON response with the following structure:
{{
    "username": "seller_{0}",
    "description": "A brief description of this seller's personality and approach",
    "user_char": "A detailed character description including their motivation, strategy, risk tolerance, and typical behavior patterns. This should be 2-3 sentences that will guide their decision-making in the marketplace."
}}

Make each seller distinct with different approaches to:
- Profit maximization vs reputation building
- Honesty vs deception
- Risk tolerance
- Long-term vs short-term thinking"""

# LLM 生成买家的系统提示词
BUYER_GENERATION_SYS_PROMPT = """You are an expert in creating diverse buyer personas for a market simulation.
Your task is to generate unique buyer characteristics that will lead to different purchasing behaviors in an online marketplace.
Each buyer should have distinct preferences, risk tolerances, and decision-making patterns."""

# LLM 生成买家的用户提示词
BUYER_GENERATION_USER_PROMPT = """Create a unique buyer persona for agent {0} in a market simulation.
The buyer operates in an online marketplace where they can purchase products from sellers with different reputation levels
and choose whether to buy warranted or unwarranted products.

Please provide a JSON response with the following structure:
{{
    "username": "buyer_{0}",
    "description": "A brief description of this buyer's personality and approach",
    "user_char": "A detailed character description including their purchasing preferences, risk tolerance, and decision-making criteria. This should be 2-3 sentences that will guide their buying behavior in the marketplace."
}}

Make each buyer distinct with different approaches to:
- Risk tolerance (willing to try new sellers vs preferring established ones)
- Preference for warranted vs unwarranted products
- Price sensitivity vs quality preference
- Information gathering behavior"""

# 历史记录格式化模板
def format_seller_history(history_log: list) -> str:
    """格式化卖家历史记录为字符串"""
    if not history_log:
        return "This is the first round. You have no past performance data."
    
    history_string = "Here is a summary of your performance in previous rounds:\n"
    for entry in history_log:
        history_string += f"- Round {entry['round']}: Listed a {entry['quality']} product. Sold: {entry['sold']}. Round Profit: {entry['profit']}. New Reputation: {entry['reputation']}\n"
    
    return history_string