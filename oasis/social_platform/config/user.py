# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# flake8: noqa: E501
import warnings
from dataclasses import dataclass
from typing import Any

from camel.prompts import TextPrompt


@dataclass
class UserInfo:
    user_name: str | None = None
    name: str | None = None
    description: str | None = None
    profile: dict[str, Any] | None = None
    recsys_type: str = "twitter"
    is_controllable: bool = False

    def to_custom_system_message(self, user_info_template: TextPrompt) -> str:
        required_keys = user_info_template.key_words
        info_keys = set(self.profile.keys())
        missing = required_keys - info_keys
        extra = info_keys - required_keys
        if missing:
            raise ValueError(
                f"Missing required keys in UserInfo.profile: {missing}")
        if extra:
            warnings.warn(f"Extra keys not used in UserInfo.profile: {extra}")

        return user_info_template.format(**self.profile)

    def to_system_message(self) -> str:
        # Router to generate prompt based on the agent's role
        role = self.profile.get("other_info", {}).get("role")
        if role == 'seller':
            return self.to_seller_master_prompt()
        elif role == 'buyer':
            return self.to_buyer_master_prompt()


    def to_seller_master_prompt(self) -> str:
        """Generates the master prompt for a Seller Agent in the combined market."""
        
        market_rules = (
            "1. Reputation Lag: Your public `reputation_score` updates with a lag of {reputation_lag} round(s). Do not expect same-round ratings to take effect immediately; plan for long-term profit.\n"
            "2. Exit (Round Settlement): `exit_market()` exits the market for this round and settles for the round (no further sales this round).\n"
            "3. Re-entry: From round {reentry_round}, you can use `reenter_market()` to return. If your reputation is negative, re-entry resets it to 0 so you can restart.\n"
            "4. Unsold Clearance: At the end of each round, any unsold listings are cleared and will not carry over automatically.\n"
            "5. Truth Warrant: If you offer a warrant, ensure advertised and true quality match; successful challenges incur severe penalties and hurt long-term profit."
        )
        
        persona = self.profile.get("other_info", {}).get("user_profile", "You are a seller.")

        prompt = f"""
    # CONTEXT
    You are a Seller Agent in a multi-round online marketplace simulation. Your sole objective is to maximize your total profit over 7 rounds.
    - Current Round: {{current_round}} / 7
    - Your Current Budget: ${{current_budget}}
    - Your Current Reputation Score: {{reputation_score}}

    # PREVIOUS ROUNDS' SUMMARY
    {{history_summary}}

    # GOAL
    Your goal is to make strategic decisions to maximize your CUMULATIVE profit.

    # YOUR PERSONALITY
    {persona}

    # MARKET RULES
    {market_rules}

    # STRATEGIC OPTIONS (AVAILABLE ACTIONS)
    - `list_product(advertised_quality: str, product_quality: str, has_warrant: bool, price: float)`: Your primary action to make a profit. You can set your own price for the product.
    - `exit_market()`: Exit the market and settle for this round (no further sales this round).
    - `reenter_market()`: If you previously exited and your reputation is negative, re-entry resets it to 0 (no immediate profit).

    # PROFIT CALCULATION (Your Profit if Product is Sold)
    **Your Profit = Your Set Price - Production Cost - Warrant Cost (if applicable)**
    
    - **Production Costs**: HQ products cost $2 to produce, LQ products cost $1 to produce
    - **Warrant Cost**: If you offer a warrant, you must pay $3 as escrow
    - **Your Price**: You decide the selling price (must be positive)
    
    **Example**: If you set price=$5, produce HQ (cost=$2), and offer warrant (cost=$3):
    - Your profit = $5 - $2 - $3 = $0

    # TASK (CRITICAL INSTRUCTION)
    You must decide and execute EXACTLY ONE action for this round based on your personality, current situation, and the following instructions.

    **Special Instruction for Round 1:**
    - To start the market, **you MUST call the `list_product` function**. 
    - Based on your personality, decide on the parameters. 

    **Instructions for Subsequent Rounds (Round 2 onwards):**
    1.  **Assess your situation**: Analyze your current reputation and past performance from the summary.
    2.  **Formulate a Strategy**: Based on your PERSONALITY, decide your plan for this round.
    3.  **Execute the Action**: You MUST call one of the available functions.
        - **Default Action Rule**: If you have no clear reason to exit or re-enter, your default action should be `list_product` to actively participate in the market.
        - **Warrant Honesty Rule**: If you offer a warrant, keep advertised and true quality aligned; successful challenges cause severe penalties and hurt long-term profit.
        - **Reputation Awareness**: Public reputation updates with lag (not real-time); plan strategies toward long-term cumulative profit.

    Provide your step-by-step reasoning first, then execute your chosen function call.
    """
        # Return template; caller will fill placeholders (current_round, reputation_lag, reentry_round, etc.)
        return prompt.strip()

    def to_buyer_master_prompt(self) -> str:
        """Generates the master prompt for a Buyer Agent in the combined market."""
        market_rules = (
            "1. **Reputation System**: You can rate each transaction with a thumbs up (+1) or a thumbs down (-1). "
            "Your ratings contribute to the seller's reputation score, which may help you make future purchasing decisions.\n"
            "2. **Truth Warrant System**: Sellers can offer 'Truth Warrants'. If you purchase a warranted product and find it "
            "doesn't match the advertised quality, you can challenge it for a cost of $1. "
            "A successful challenge will refund your purchase price and grant you a bonus."
        )
        available_actions = (
            "1. `purchase_product_id(post_id: int)`\n"
            "2. `rate_transaction(transaction_id: int, rating: int)` (You can rate any transaction AFTER a purchase)\n"
            "3. `challenge_warrant(post_id: int)` (You can challenge a product that has a warrant AFTER a purchase)\n"
        )

        persona = self.profile.get("other_info", {}).get("user_profile", "You are a buyer.")
        
        prompt = f"""
    # CONTEXT
    You are a Buyer Agent in a multi-round online marketplace simulation. Your sole objective is to maximize your total utility over 7 rounds.
    - Current Round: {{current_round}} / 7
    - Your Cumulative Utility: {{cumulative_utility}}

    # GOAL
    Your only goal is to make strategic decisions to maximize your cumulative utility. You must actively participate in the market to achieve this.

    # YOUR PERSONALITY
    {persona}

    # MARKET RULES
    {market_rules}

    # AVAILABLE ACTIONS
    {available_actions}

    # PAYOFF MATRIX (Your Utility Calculation)
    - Challenge Cost: $1
    | Product Quality | Advertised Quality | Your Action                      | Your Utility                     |
    | :-------------- | :----------------- | :------------------------------- | :------------------------------- |
    | HQ              | HQ                 | Buy                              |         3                        |
    | LQ              | LQ                 | Buy                              |         2                        |
    | LQ              | HQ                 | Buy (No Challenge)               |        -3                        |
    | LQ              | HQ                 | Buy & Challenge Successfully     |         4                        |

    # AVAILABLE PRODUCTS
    {{product_listings}}

    # TASK: YOUR DECISION WORKFLOW FOR THIS ROUND
    Based on all the information above, decide which product you should purchase to maximize your cumulative utility.(you should only purchase once!)
    """
        return prompt.strip()
    
    def to_buyer_post_purchase_prompt(self) -> str:
        """购买后的决策Promp"""
        
        prompt = f"""
    # CONTEXT
    You are a Buyer Agent. You have just completed a purchase. Now you must decide on your post-purchase actions based on the outcome.
    - Transaction ID: {{transaction_id}}
    - Product ID: {{post_id}}
    - Advertised Quality: {{advertised_quality}}
    - True Quality You Received: {{true_quality}}
    - Was Warranted: {{has_warrant}}
    - Your Utility from This Purchase: {{buyer_utility}}

    # TASK: YOUR POST-PURCHASE WORKFLOW
    You MUST now consider two actions: challenging and rating.

    **Step 1: Challenge Decision**
    - **Rule**: If the product was advertised as 'HQ' but the true quality was 'LQ', AND it was warranted (`Was Warranted: True`), you SHOULD challenge it to maximize your utility.
    - **Action**: If you decide to challenge, call `challenge_warrant(post_id={{post_id}})`. Otherwise, do not call this function.

    **Step 2: Rating Decision**
    - **Rule**: You MUST always rate the transaction.
    - If the `advertised_quality` matches the `true_quality`, give a positive rating (`rating=1`).
    - If they do not match, give a negative rating (`rating=-1`).
    - **Action**: Call `rate_transaction(transaction_id={{transaction_id}}, rating=<Your Chosen 1 or -1>)`.

    **Your Response:**
    First, provide a step-by-step reasoning for your decisions. Then, call the chosen function(s). You can call both functions if necessary.
    """
        return prompt.strip()

    def to_reddit_system_message(self) -> str:
        name_string = ""
        description_string = ""
        if self.name is not None:
            name_string = f"Your name is {self.name}."
        if self.profile is None:
            description = name_string
        elif "other_info" not in self.profile:
            description = name_string
        elif "user_profile" in self.profile["other_info"]:
            if self.profile["other_info"]["user_profile"] is not None:
                user_profile = self.profile["other_info"]["user_profile"]
                description_string = f"Your have profile: {user_profile}."
                description = f"{name_string}\n{description_string}"
                print(self.profile['other_info'])
                description += (
                    f"You are a {self.profile['other_info']['gender']}, "
                    f"{self.profile['other_info']['age']} years old, with an MBTI "
                    f"personality type of {self.profile['other_info']['mbti']} from "
                    f"{self.profile['other_info']['country']}.")

        system_content = f"""
# OBJECTIVE
You're a Reddit user, and I'll present you with some tweets. After you see the tweets, choose some actions from the following functions.

# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.
{description}

# RESPONSE METHOD
Please perform actions by tool calling.
"""
        return system_content
