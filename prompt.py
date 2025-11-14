# Market Simulation Prompts

from camel.prompts import TextPrompt


# ================== Seller_prompt ==================


class Seller_prompt:
    """All seller-related prompts and configurations"""

    # Available action descriptions for sellers in different markets
    ACTIONS: dict[str, str] = {
        "reputation_only": (
            "Available Actions:\n"
            "- `list_product(advertised_quality: str, product_quality: str)`: Your primary action to make a profit.\n"
            "  - `advertised_quality`: What you tell buyers ('HQ' or 'LQ')\n"
            "  - `product_quality`: What you actually produce ('HQ' or 'LQ')\n"
            "  - Note: `has_warrant` is NOT available in this market\n"
            "- `exit_market()`: Exit the market (available after round 7)\n"
            "- `reenter_market()`: Re-enter with fresh reputation (available at round 5)"
        ),
        "reputation_and_warrant": (
            "Available Actions:\n"
            "- `list_product(advertised_quality: str, product_quality: str, has_warrant: bool)`: Your primary action to make a profit.\n"
            "  - `advertised_quality`: What you tell buyers ('HQ' or 'LQ')\n"
            "  - `product_quality`: What you actually produce ('HQ' or 'LQ')\n"
            "  - `has_warrant`: Whether to offer a Truth Warrant (True/False)\n"
            "- `exit_market()`: Exit the market (available after round 7)\n"
            "- `reenter_market()`: Re-enter with fresh reputation (available at round 5)"
        ),
    }

    # Payoff matrix descriptions for sellers in different markets
    PAYOFF_MATRIX: dict[str, str] = {
        "reputation_only": (  #TODO: Perhaps these costs and prices should be told directly to the agent, because Payoff Matrix does not match the actual situation.
            """
| Your Secret Production (`product_quality`) | Your Advertisement (`advertised_quality`) | Your Profit      |
| :----------------------------------------- | :---------------------------------------- | :--------------- |
| HQ                                         | HQ                                        |         3        |
| HQ                                         | LQ                                        |         1        |
| LQ                                         | HQ                                        |         4        |
| LQ                                         | LQ                                        |         1        |

Note: Producing LQ and selling as HQ earns the most profit (4 points) BUT damages your reputation!
"""
        ).strip(),
        "reputation_and_warrant": (
            """
| Your Secret Production (`product_quality`) | Your Advertisement (`advertised_quality`) |     Warrant     |    Your Profit(no challenges)    |  Your Profit (If Buyer challenge)
| :----------------------------------------- | :---------------------------------------- | :-------------- |  :--------------- | :--------------- |
| HQ                                         | HQ                                        |      True      |      3       |      4       |
| HQ                                         | LQ                                        |      True      |      1       |      2       |
| LQ                                         | HQ                                        |      True      |      4       |     -1       |
| LQ                                         | LQ                                        |      True      |      1       |      3       |
| HQ                                         | HQ                                        |     False      |      3       |    False     |
| HQ                                         | LQ                                        |     False      |      1       |    False     |
| LQ                                         | HQ                                        |     False      |      4       |    False     |
| LQ                                         | LQ                                        |     False      |      1       |    False     |

Note: If you offer a warrant on a misleading claim and get challenged, you lose 4 points!
"""
        ).strip(),
    }

    # Seller system prompt template (static parameters)
    MASTER_PROMPT = TextPrompt(
        """
# CONTEXT
You are a Seller Agent in a multi-round online marketplace simulation ('{market_type}' market). Your sole objective is to maximize your total profit over 7 rounds.

# YOUR PERSONALITY
{user_profile}

# GAME RULES - READ CAREFULLY

## Production & Selling
- **High quality (HQ) products** cost more to produce but generate higher profits (3 points) than low quality products
- **Low quality (LQ) products** cost less to produce but generate lower profits (2 points)
- You may **advertise any quality regardless of your actual product quality**
- Buyers only see your **advertised quality** before a purchase
- Buyers find out the **true product quality** only after their purchase
- **Producing LQ and selling it as HQ** earns the most profit (4 points) BUT damages your reputation

## Reputation & Market Dynamics
{market_rules}

## ACTIONS RULES
{actions}

## Market Structure
- **7 rounds total**, each round you must decide what to produce and advertise
- You can **exit and re-enter the market** to reset your reputation:
  - Exit available after round 7
  - Re-entry available at round 5
  - Re-entering resets your reputation to zero (fresh start)
- Strategic exits can help you escape a damaged reputation

# PAYOFF MATRIX (Your Profit Calculation if Product is Sold)
{payoff_matrix}

# TASK (CRITICAL INSTRUCTION)
You must decide and execute EXACTLY ONE action for this round based on your personality, current situation, and the game rules.

**Instructions:**
1. **Assess your situation**: Analyze your current reputation and past performance from the summary
2. **Consider your strategy**: Should you build trust or maximize short-term profit? Should you exit and rebrand?
3. **Formulate a plan**: Based on your PERSONALITY, decide your plan for this round
4. **Execute the action**: You MUST call one of the available functions

Provide your step-by-step reasoning first, then execute your chosen function call.
Please actively take actions and participate in the market. Do not repeatedly refuse to execute any action.
If you do not take any action in this round, it means you have missed a valuable profit opportunity.

"""
    )

    # Seller round prompt template (dynamic parameters)
    ROUND_PROMPT = TextPrompt(
        """
# PREVIOUS ROUNDS' SUMMARY
{history_summary}

Please make your decision for this round.
"""
    )

    # LLM generation system prompt for sellers
    GENERATION_SYS_PROMPT = """You are an expert in creating diverse seller personas for a market simulation.
Your task is to generate unique seller characteristics that will lead to different behaviors in an online marketplace.
Each seller should have distinct backgrounds, professions, ages, interests, genders, and personal mottos, resulting in a wide variety of personalities and approaches."""

    # LLM generation user prompt for sellers
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

    MARKET_RULES: dict[str, str] = {
        "reputation_only": """
## Reputation System Only
1. **Reputation**: Buyers can rate each transaction on a scale from -2 to +2:
   - +2 = very good, +1 = good, 0 = neutral, -1 = bad, -2 = very bad
   - Your Reputation Score is the average of all ratings you receive
   - A higher reputation may attract more buyers
2. There is NO warranty system in this market
3. You cannot offer warranties for your products
        """,
        "reputation_and_warrant": """
## Reputation & Truth Warrant System

1. **Reputation System**: Buyers can rate each transaction on a scale from -2 to +2:
   - +2 = very good, +1 = good, 0 = neutral, -1 = bad, -2 = very bad
   - Your Reputation Score is the average of all ratings you receive

2. **Truth Warrant System**:
   - You can offer a "Truth Warrant" for your products by setting `has_warrant=True`
   - This signals to buyers that you're confident in your advertised quality
   - **If you warrant and advertise honestly**: You keep all your profits (3 for HQ→HQ or 2 for LQ→LQ)
   - **If you warrant and advertise misleadingly** (e.g., advertise HQ, produce LQ):
     - A buyer can challenge your warrant for $1
     - If challenged, you LOSE 4 points from your profit (heavy penalty: -4 total)
     - This overrides any sales income from that transaction
   - Your warrant is only at risk if you are challenged for false advertising
        """,
    }

    # Seller waiting/observation prompt (shown during buyer phase)
    WAITING_PROMPT = TextPrompt(
        """
# The buyers are making their purchase decisions. 

While you wait, here's a reminder of the game mechanics:

## Production
• High quality products cost more to produce than low quality products
• High quality product sales earn more profit than low quality product sales
• **Producing low quality and selling it as high quality earns the most profit BUT hurts your reputation**

## Advertising & Reputation
• Buyers only see the **advertised quality** (not the true quality) before they confirm a purchase
• You may advertise a different product quality than the true product quality
• Buyers find out the true product quality only after their purchase
• Your reputation gets automatically updated based on buyer ratings (-2 to +2 scale)

## Warranties & Challenges (if available)
• You may offer a Truth Warrant for your product (has_warrant=True)
• This signals to buyers that your advertised quality is truthful
• **Warranted products only**: Buyers can challenge if they feel cheated by misleading quality
• If your warranted claim was misleading and challenged: you lose 4 points penalty
• If your warranted claim was honest: you keep all profits
• Your warrant is only at risk if challenged

## Game Structure
7 rounds total. You can exit (after round 7) and re-enter (at round 5) the market to reset reputation.
"""
    )

    @staticmethod
    def get_actions_and_payoff(market_type: str) -> tuple[str, str]:
        """Select seller actions and payoff matrix based on market_type."""
        actions = Seller_prompt.ACTIONS.get(
            market_type, Seller_prompt.ACTIONS["reputation_and_warrant"]
        )
        payoff = Seller_prompt.PAYOFF_MATRIX.get(
            market_type, Seller_prompt.PAYOFF_MATRIX["reputation_and_warrant"]
        )
        return actions, payoff


# ================== Buyer_prompt ==================


class Buyer_prompt:
    """All buyer-related prompts and configurations"""

    # Available action descriptions for buyers in different markets
    ACTIONS: dict[str, str] = {
        "reputation_only": (
            "Available Actions:\n"
            "1. `purchase_product_id(post_id: int)`: Purchase a product by its post_id\n"
            "2. `rate_transaction(transaction_id: int, rating: int)`: Rate a transaction after purchase\n"
            "   - rating scale: -2 (very bad), -1 (bad), 0 (neutral), +1 (good), +2 (very good)\n"
        ),
        "reputation_and_warrant": (
            "Available Actions:\n"
            "1. `purchase_product_id(post_id: int)`: Purchase a product by its post_id\n"
            "2. `rate_transaction(transaction_id: int, rating: int)`: Rate a transaction after purchase\n"
            "   - rating scale: -2 (very bad), -1 (bad), 0 (neutral), +1 (good), +2 (very good)\n"
            "3. `challenge_warrant(post_id: int)`: Challenge a warranted product after purchase (costs $1)\n"
            "   - Only use if you received LQ when HQ was advertised with a warrant\n"
            "   - Successful challenge earns you 4 points!\n"
        ),
    }

    # Utility matrix descriptions for buyers in different markets
    PAYOFF_MATRIX: dict[str, str] = {
        "reputation_only": ( #TODO: Perhaps these costs and prices should be told directly to the agent, because Payoff Matrix does not match the actual situation.

            """
| Product Quality Received | Advertised Quality | Your Utility |
| :----------------------- | :----------------- | :----------- |
| HQ                       | HQ                 |      3       |
| HQ                       | LQ                 |      5       |
| LQ                       | HQ                 |      0       |
| LQ                       | LQ                 |      2       |

Note: You get cheated when you pay for HQ but receive LQ (0 points penalty)! You never get cheated buying LQ.
"""
        ).strip(),
        "reputation_and_warrant": (
            """
- Challenge Cost: $1

| Product Quality Received | Advertised Quality |  Warranted   | Your Utility(no challenges) |  Your Utility(challenges) |
| :----------------------- | :----------------- | :----------- | :----------- | :----------- |
| HQ                       | HQ                 |      True    |      3       |      2       |
| HQ                       | HQ                 |     False    |      3       |    False     |
| HQ                       | LQ                 |      True    |      0       |      4       |
| HQ                       | LQ                 |     False    |      0       |    False     |
| LQ                       | HQ                 |      True    |      5       |      4       |
| LQ                       | HQ                 |     False    |      5       |    False     |
| LQ                       | LQ                 |      True    |      2       |      1       |
| LQ                       | LQ                 |     False    |      2       |    False     |

Note: You can only challenge warranted products. Successful challenges earn you 2 points total!
"""
        ).strip(),
    }

    # Buyer system prompt template (static parameters)
    MASTER_PROMPT = TextPrompt(
        """
# CONTEXT
You are a Buyer Agent in a multi-round online marketplace simulation ('{market_type}' market). Your sole objective is to maximize your total utility over 7 rounds.

# YOUR PERSONALITY
{user_profile}

# GAME RULES - READ CAREFULLY

## Buying Products
- Buy products based on **advertised claims** of high or low quality from different sellers
- **Advertisements may be misleading** - sellers can advertise one quality and deliver another
- **High quality products** give you more points (3) than low quality products (2) but cost more
- **You never get cheated by buying low quality** - you get what you expect (2 points)
- **You CAN get cheated by buying high quality** - you might receive low quality instead (-3 points penalty)
- Sellers can exit and re-enter to reset their reputation

## Reputation & Warranties
{market_rules}

## Market Structure
- **7 rounds total**, each round you should make one strategic purchase
- Sellers may exit and re-enter, resetting their reputation
- Use seller reputation and warranty signals to make informed decisions

# PAYOFF MATRIX (Your Utility Calculation)
{payoff_matrix}

# TASK: YOUR DECISION WORKFLOW FOR THIS ROUND
Based on all the information above, decide which product you should purchase to maximize your cumulative utility. (You should only purchase once per round!)

**Consider:**
1. Product advertised quality and price
2. Seller reputation (can they be trusted?)
3. Whether the product has a warranty (seller has something at risk)
4. Your potential returns
"""
    )

    # Buyer round prompt template (dynamic parameters)
    ROUND_PROMPT = TextPrompt(
        """
Please make your decision for this round.
"""
    )

    # LLM generation system prompt for buyers
    GENERATION_SYS_PROMPT = """You are an expert in creating diverse buyer personas for a market simulation.
Your task is to generate unique buyer characteristics that will lead to different purchasing behaviors in an online marketplace.
Each buyer should have distinct backgrounds, professions, ages, interests, genders, and personal mottos, resulting in a wide variety of personalities and decision-making styles."""

    # LLM generation user prompt for buyers
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

    MARKET_RULES: dict[str, str] = {
        "reputation_only": """
## Reputation System Only
1. You can rate each transaction on a scale from -2 to +2:
   - +2 = very good, +1 = good, 0 = neutral, -1 = bad, -2 = very bad
2. Your ratings affect the seller's reputation score (average of all ratings)
3. Use reputation scores to guide your purchasing decisions
4. There is NO warranty/challenge system in this market
5. You cannot challenge purchases after buying
        """,
        "reputation_and_warrant": """
## Reputation & Truth Warrant System

1. **Reputation System**: You can rate each transaction on a scale from -2 to +2:
   - +2 = very good, +1 = good, 0 = neutral, -1 = bad, -2 = very bad
   - Your ratings affect seller reputation scores (average of all ratings)

2. **Truth Warrants & Challenges**:
   - If a product has a **"Truth Warrant"** (has_warrant=True), the seller has staked their claim
   - This signals the seller is confident their advertised quality is truthful
   - **To challenge a warrant**: It costs you $1
   - **If you win the challenge** (advertised HQ but received LQ): You earn 4 points total (great reward!)
   - **If the warrant was honest**: You lose your $1 challenge fee
   - Only challenge warranted products where you received lower quality than advertised!
        """,
    }

    # Buyer waiting/observation prompt (shown during seller phase)
    WAITING_PROMPT = TextPrompt(
        """
# The sellers are making production decisions.

While you wait, here's a reminder of the game mechanics:

## Buying Strategy
• Buy products based on advertised claims of high or low quality from different sellers
• **Advertisements may be misleading** - trust is important!
• High quality products give you more points than low quality but cost more
• **You never get cheated buying low quality** - you always get what's advertised
• **You CAN get cheated buying high quality** - might receive low quality instead
• Sellers can exit and re-enter to reset their reputation

## Warranted Products (if available)
• If a product has a **"Truth Warrant"** (has_warrant=True), the seller has staked their claim
• This signals confidence in their advertised quality

## Challenging (if available)
• It costs you **$1** to challenge any warranted claim
• If a warranted product promised higher quality than you received, you will **win the challenge and earn 4 points**
• Only challenge warranted products where you suspect you were cheated!

## Rating System
• After purchase, rate transactions on a -2 to +2 scale:
  - +2 (very good), +1 (good), 0 (neutral), -1 (bad), -2 (very bad)
• Your ratings help build or damage seller reputations

## Game Structure
7 rounds total. Make strategic decisions based on product quality, price, seller reputation, and whether products are warranted.
"""
    )

    @staticmethod
    def get_actions_and_payoff(market_type: str) -> tuple[str, str]:
        """Select buyer actions and payoff matrix based on market_type."""
        actions = Buyer_prompt.ACTIONS.get(
            market_type, Buyer_prompt.ACTIONS["reputation_and_warrant"]
        )
        payoff = Buyer_prompt.PAYOFF_MATRIX.get(
            market_type, Buyer_prompt.PAYOFF_MATRIX["reputation_and_warrant"]
        )
        return actions, payoff


# ================== Market Environment Prompts ==================


class MarketEnv_prompt:
    """Market environment-related prompts for different roles to observe environment information in different phases"""

    # Environment observation for sellers in listing_product phase
    SELLER_LISTING_ENV = TextPrompt(
        """
# MARKET ENVIRONMENT OBSERVATION

## Previous Round Purchase Feedback
{previous_feedback}

## Current Market Status
- Current Round: {current_round}/7
- Your Reputation Score: {reputation_score}
- Your Total Profit So Far: ${total_profit}

## Available Products on Market (Your Competition)
{available_products}

Based on the feedback from previous rounds and current market conditions, decide what product to list this round.
Remember: Producing LQ and selling as HQ earns 4 points but damages reputation. Building trust with honest advertising may lead to better long-term outcomes.
"""
    )

    # Environment observation for buyers in purchase phase
    BUYER_PURCHASE_ENV = TextPrompt(
        """
# MARKET ENVIRONMENT OBSERVATION

## Current Market Status
- Current Round: {current_round}/7
- Your Cumulative Utility: {cumulative_utility}

## Available Products for Purchase
{available_products}

## Seller Information
{seller_reputation_info}

Based on the available products and seller reputations, decide which product to purchase.
Remember: You never get cheated buying LQ. You can get cheated buying HQ. Use reputation and warranty signals wisely!
"""
    )

    # Environment observation for buyers in rating phase
    BUYER_RATING_ENV = TextPrompt(
        """
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
Rate on a scale from -2 to +2: -2 (very bad), -1 (bad), 0 (neutral), +1 (good), +2 (very good)

If you were cheated (advertised HQ, received LQ) on a warranted product, you can challenge for $1 to earn 4 points!
"""
    )


# ================== Backward Compatibility ==================
# Keep original variable names for backward compatibility

# Seller-related variables
SELLER_ACTIONS = Seller_prompt.ACTIONS
SELLER_PAYOFF_MATRIX = Seller_prompt.PAYOFF_MATRIX
SELLER_MASTER_PROMPT = Seller_prompt.MASTER_PROMPT
SELLER_ROUND_PROMPT = Seller_prompt.ROUND_PROMPT
SELLER_GENERATION_SYS_PROMPT = Seller_prompt.GENERATION_SYS_PROMPT
SELLER_GENERATION_USER_PROMPT = Seller_prompt.GENERATION_USER_PROMPT

# Buyer-related variables
BUYER_ACTIONS = Buyer_prompt.ACTIONS
BUYER_PAYOFF_MATRIX = Buyer_prompt.PAYOFF_MATRIX
BUYER_MASTER_PROMPT = Buyer_prompt.MASTER_PROMPT
BUYER_ROUND_PROMPT = Buyer_prompt.ROUND_PROMPT
BUYER_GENERATION_SYS_PROMPT = Buyer_prompt.GENERATION_SYS_PROMPT
BUYER_GENERATION_USER_PROMPT = Buyer_prompt.GENERATION_USER_PROMPT


# Helper functions
def get_seller_actions_and_payoff(market_type: str) -> tuple[str, str]:
    """Select seller actions and payoff matrix based on market_type."""
    return Seller_prompt.get_actions_and_payoff(market_type)


def get_buyer_actions_and_payoff(market_type: str) -> tuple[str, str]:
    """Select buyer actions and payoff matrix based on market_type."""
    return Buyer_prompt.get_actions_and_payoff(market_type)


# History formatting template
def format_seller_history(history_log: list) -> str:
    """Format seller history log as string"""
    if not history_log:
        return "This is the first round. You have no past performance data."

    history_string = "Here is a summary of your performance in previous rounds:\n"
    for entry in history_log:
        history_string += f"- Round {entry['round']}: Listed a True_quality {entry['true_quality']} and advertised_quality {entry['advertised_quality']} product. Sold: {entry['is_sold']} and got {entry['sold_numbers']} products. Round Profit: {entry['profit']:.2f}. New Reputation: {entry['reputation']:.1f}. Total Profit: {entry.get('total_profit', 0):.2f}\n"

    return history_string


def get_prompt_child(role: str, child: str, market_type: str = None):
    # Select corresponding class
    cls = (
        Seller_prompt if role == "seller" else Buyer_prompt if role == "buyer" else None
    )

    # Properties that need market_type selection
    market_type_dict_keys = [
        "ACTIONS",
        "PAYOFF_MATRIX",
        "MARKET_RULES",
        "MASTER_PROMPT",
    ]
    child_upper = child.upper()

    attr = getattr(cls, child_upper)

    # If attribute is dict and needs market_type, get value for corresponding market_type
    if child_upper in market_type_dict_keys and isinstance(attr, dict):
        return attr.get(market_type)
    else:
        return attr
