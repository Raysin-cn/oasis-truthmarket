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
        
        # Market rules now combine both reputation and warrant mechanisms
        market_rules = (
            "1. **Reputation System**: Buyers can rate each transaction with a thumbs up (+1) or a thumbs down (-1). "
            "Your Reputation Score is the sum of these ratings. A higher reputation may attract more buyers.\n"
            "2. **Truth Warrant System**: You can offer a 'Truth Warrant' for your products. If you falsely advertise a warranted product "
            "(e.g., advertise HQ, produce LQ) and the buyer challenges it, you will be heavily penalized, "
            "losing a fixed amount of 4 points from your profit, overriding any sales income."
        )
        
        prompt = f"""
# CONTEXT
You are a Seller Agent in a multi-round online marketplace simulation. Your sole objective is to maximize your total profit over 7 rounds.
- Current Round: {{current_round}} / 7
- Your Current Budget: ${{current_budget}}
- Your Current Reputation Score: {{reputation_score}}

# PREVIOUS ROUNDS' SUMMARY
{{history_summary}}

# GOAL
Your goal is to make strategic decisions to maximize your CUMULATIVE profit. Short-term gains from deception might lead to a low reputation, hurting long-term profits.

# MARKET RULES
{market_rules}

# STRATEGIC OPTIONS (AVAILABLE ACTIONS)
- `list_product(advertised_quality: str, product_quality: str, has_warrant: bool)`: 
    - Your primary action to make a profit.
    - `product_quality`: This is your SECRET choice, deciding whether to produce a High-Quality (HQ) or Low-Quality (LQ) product.
    - `advertised_quality`: This is what you PUBLICLY claim your product's quality is.
    - `has_warrant`: A strategic choice to offer a quality guarantee.
- `exit_market()`: A strategic retreat. If your reputation score is very low (e.g., negative), you can exit the market for a round to RESET YOUR REPUTATION TO 0. This means you earn no profit for the round you exit.
- `reenter_market()`: If you have previously exited, use this to start selling again.

# PAYOFF MATRIX (Your Profit Calculation if Product is Sold)
| Your Secret Production (`product_quality`) | Your Advertisement (`advertised_quality`) | Buyer Action              | Your Profit      |
| :----------------------------------------- | :---------------------------------------- | :------------------------ | :--------------- |
| HQ                                         | HQ                                        | Buys (No Challenge)       | price(5) - 2 = 3 |
| LQ                                         | LQ                                        | Buys (No Challenge)       | price(3) - 1 = 2 |
| LQ                                         | HQ                                        | Buys (No Challenge)       | price(5) - 1 = 4 |
| LQ                                         | HQ                                        | Buys & Challenges Warrant | **-4 (Penalty)** |

# TASK (CRITICAL INSTRUCTION)
Your task is to make a strategic decision and execute ONE of your available actions. You MUST choose either `list_product` or `exit_market` (or `reenter_market` if applicable).
1.  **Assess your situation**: Analyze your current reputation and past performance from the summary.
2.  **Formulate a Strategy**: Based on your assessment, decide your plan for this round. For example: "My reputation is good, so I will secretly produce HQ and advertise it as HQ," or "My reputation is negative, so I must exit the market to reset it."
3.  **Execute the Action**: Provide your step-by-step reasoning and then call the function that matches your strategy. Inaction is not a valid option.
"""
        return prompt.strip()

    def to_buyer_master_prompt(self) -> str:
        """Generates the final, integrated master prompt for a Buyer Agent."""
        
        market_rules = (
            "1. **Reputation System**: You can rate each transaction with a thumbs up (+1) or a thumbs down (-1). "
            "Your ratings contribute to the seller's reputation score, which may help you make future purchasing decisions.\n"
            "2. **Truth Warrant System**: Sellers can offer 'Truth Warrants'. If you purchase a warranted product and find it "
            "doesn't match the advertised quality, you can challenge it for a cost of $1. "
            "A successful challenge will refund your purchase price and grant you a bonus."
        )
        available_actions = (
            "1. `purchase_product(post_id: int)`\n"
            "2. `rate_transaction(transaction_id: int, rating: int)` (You can rate any transaction after purchase)\n"
            "3. `challenge_warrant(post_id: int)` (You can challenge a product that has a warrant)\n"
        )

        persona = self.profile.get("other_info", {}).get("user_profile", "You are a buyer.")
        
        prompt = f"""
# CONTEXT
You are a Buyer Agent in a multi-round online marketplace simulation. Your sole objective is to maximize your total utility.
- Current Round: {{current_round}} / 7
- Your Cumulative Utility: {{cumulative_utility}}

# GOAL
Your only goal is to make strategic decisions to maximize your cumulative utility by the end of the simulation.

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
| HQ              | HQ                 | Buy                              | 8 - price                        |
| LQ              | LQ                 | Buy                              | 5 - price                        |
| LQ              | HQ                 | Buy (No Challenge)               | 5 - price                        |
| LQ              | HQ                 | Buy & Challenge Successfully     | 5 - price + refund + bonus       |

# AVAILABLE PRODUCTS
{{product_listings}}

# TASK
Based on all the information above, decide your action for this round. Consider the available products, their prices, warranties, and seller reputations. Provide your step-by-step reasoning, and then state your chosen action by calling the appropriate function.
"""
        return prompt.strip()
    
    def to_twitter_system_message(self) -> str:
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

        system_content = f"""
# OBJECTIVE
You're a Twitter user, and I'll present you with some posts. After you see the posts, choose some actions from the following functions.

# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.
{description}

# RESPONSE METHOD
Please perform actions by tool calling.
        """

        return system_content

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
