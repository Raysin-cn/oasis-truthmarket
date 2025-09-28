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
from prompt import MARKET_CONFIG, get_seller_actions_and_payoff, get_buyer_actions_and_payoff, SELLER_MASTER_PROMPT, BUYER_MASTER_PROMPT

@dataclass
class UserInfo:
    user_name: str | None = None
    name: str | None = None
    description: str | None = None
    profile: dict[str, Any] | None = None
    recsys_type: str = "twitter"
    is_controllable: bool = False
    market_type: str = 'reputation_and_warrant'

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
        """
        根据角色和市场类型路由到正确的master prompt生成函数。
        """
        role = self.profile.get("other_info", {}).get("role")
        
        if role == 'seller':
            return self.to_seller_master_prompt()
        elif role == 'buyer':
            return self.to_buyer_master_prompt()
        return ""

    def to_seller_master_prompt(self) -> str:
        """为卖家生成通用的主提示词，使用TextPrompt模板。"""
        # 准备模板所需的参数
        market_rules = MARKET_CONFIG.get(self.market_type, {}).get('seller', "No rules defined for this role.")
        actions, payoff_matrix = get_seller_actions_and_payoff(self.market_type)
        
        # 使用TextPrompt模板
        return SELLER_MASTER_PROMPT.format(
            market_type=self.market_type,
            market_rules=market_rules,
            actions=actions,
            payoff_matrix=payoff_matrix,
            user_profile=self.profile.get("other_info", {}).get("user_profile", "You are a seller.")
        )

    def to_buyer_master_prompt(self) -> str:
        """为买家生成通用的主提示词，使用TextPrompt模板。"""
        # 准备模板所需的参数
        market_rules = MARKET_CONFIG.get(self.market_type, {}).get('buyer', "No rules defined for this role.")
        available_actions, payoff_matrix = get_buyer_actions_and_payoff(self.market_type)
        
        # 使用TextPrompt模板
        return BUYER_MASTER_PROMPT.format(
            market_type=self.market_type,
            market_rules=market_rules,
            actions=available_actions,
            payoff_matrix=payoff_matrix,
            user_profile=self.profile.get("other_info", {}).get("user_profile", "You are a buyer.")
        )
    

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
