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
        if self.recsys_type != "reddit":
            return self.to_twitter_system_message()
        else:
            return self.to_reddit_system_message()

    def to_market_system_message(self) -> str:
        # Router to generate prompt based on the agent's role
        role = self.profile.get("other_info", {}).get("role")
        if role == 'seller':
            return self.to_seller_master_prompt()
        elif role == 'buyer':
            return self.to_buyer_master_prompt()

    def to_seller_master_prompt(self) -> str:
        """为卖家动态生成Master Prompt，包含历史记录。"""
        
        market_rules = (
            "1. **Reputation System**: You can rate each transaction with a thumbs up (+1) or a thumbs down (-1). "
            "Your ratings contribute to the seller's reputation score, which may help you make future purchasing decisions.\n"
            "2. **Truth Warrant System**: Sellers can offer 'Truth Warrants'. If you purchase a warranted product and find it "
            "doesn't match the advertised quality, you can challenge it for a cost of $1. "
            "A successful challenge will refund your purchase price and grant you a bonus."
        )
        
        prompt = f"""
# CONTEXT
You are a Seller Agent in a multi-round online marketplace simulation. Your sole objective is to maximize your total profit. This market features both a Reputation and a Truth Warrant system.
- Current Round: {{current_round}} / 7
- Your Current Budget: ${{current_budget}}
- Your Current Reputation Score: {{reputation_score}}

# PREVIOUS ROUNDS' SUMMARY
{{history_summary}}

# GOAL
Your only goal is to make strategic decisions to maximize your cumulative profit by the end of the simulation.

# MARKET RULES
{market_rules}

# AVAILABLE ACTIONS
- `list_product(advertised_quality: str, product_quality: str, has_warrant: bool)`
- `exit_market()`

# PAYOFF MATRIX
- Cost to Produce HQ: 2, Price: 5, Profit: 3
- Cost to Produce LQ: 1, Price: 3, Profit: 2
- Profit from deceptive LQ (advertised as HQ): 4
- Penalty for a successfully challenged warrant: -4

# TASK
You MUST list one product this round. Based on your performance in previous rounds (summarized above) and your overall goal, decide your strategy. Choose whether to produce HQ or LQ, whether to advertise honestly, and whether to use a warrant. Provide your step-by-step reasoning, then state your chosen action.
"""
        return prompt.strip()
    
    def to_buyer_master_prompt(self) -> str:
        """Generates the master prompt for a Buyer Agent in the combined market."""
        
        # Market rules and actions for the combined market
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

        prompt = f"""
# CONTEXT
You are a Buyer Agent in a multi-round online marketplace simulation. Your sole objective is to maximize your total utility. This market features both a Reputation and a Truth Warrant system.
- Current Round: {{current_round}} / 7
- Your Cumulative Utility: {{cumulative_utility}}

# GOAL
Your only goal is to make strategic decisions to maximize your cumulative utility by the end of the simulation.

# MARKET RULES
{market_rules}

# AVAILABLE ACTIONS
{available_actions}

# PAYOFF MATRIX
- Utility from High-Quality (HQ) product: 8 - price
- Utility from Low-Quality (LQ) product: 5 - price
- Cost to Challenge: 1
- Reward for successful challenge: Purchase price refund + bonus

# AVAILABLE PRODUCTS
{{product_listings}}

# TASK
Based on all the information above, especially the list of available products, decide your action for this round. Consider seller reputations and whether a product is warranted. Provide your step-by-step reasoning, then state your chosen action.
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
