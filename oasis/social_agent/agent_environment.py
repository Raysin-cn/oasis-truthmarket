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
from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from string import Template

import os
from oasis.social_agent.agent_action import SocialAction
from oasis.social_platform.database import get_db_path


class Environment(ABC):
    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


class SocialEnvironment(Environment):
    followers_env_template = Template("I have $num_followers followers.")
    follows_env_template = Template("I have $num_follows follows.")

    posts_env_template = Template(
        "After refreshing, you see some posts $posts")

    groups_env_template = Template(
        "And there are many group chat channels $all_groups\n"
        "And You are already in some groups $joined_groups\n"
        "You receive some messages from them $messages\n"
        "You can join the groups you are interested, "
        "leave the groups you already in, send messages to the group "
        "you already in.\n"
        "You must make sure you can only send messages to the group you "
        "are already in")
    env_template = Template(
        "$groups_env\n"
        "$posts_env\npick one you want to perform action that best "
        "reflects your current inclination based on your profile and "
        "posts content. Do not limit your action in just `like` to like posts")
    
    market_env_template = Template(
        "Current Round: $current_round / 7\n"
        "Current Budget: $$current_budget\n"
        "Reputation Score: $reputation_score\n"
        "Cumulative Utility: $cumulative_utility\n\n"
        "Current available products:\n$products")

    def __init__(self, action: SocialAction, db_path: str = ""):
        self.action = action
        self.db_path = db_path if db_path else get_db_path()
        # === market-sim: add a flag to distinguish modes ===
        self.is_market_sim = False 

    def get_product_listings_for_env(self) -> str:
        """Query all products on sale from database and format as string."""
        if not os.path.exists(self.db_path):
            return "No products are currently on sale."
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        listings = "No products are currently on sale."
        try:
            cursor.execute(
                """
                SELECT p.product_id, p.user_id, p.advertised_quality, p.price, p.has_warrant,
                    COALESCE(u.reputation_score, 0) AS reputation_score
                FROM product p
                LEFT JOIN user u ON u.user_id = p.user_id
                WHERE p.status = 'on_sale'
                """
            )
            products = cursor.fetchall()
            if products:
                listings = "Here is the list of products currently on sale:\n"
                for p in products:
                    warrant_info = " (Warranted)" if p[4] else ""
                    listings += (
                        f"- Product ID: {p[0]}, Seller ID: {p[1]}, Seller Reputation: {p[5]}, "
                        f"Advertised Quality: {p[2]}, Price: ${p[3]:.2f}{warrant_info}\n"
                    )
        except sqlite3.Error as e:
            print(f"Database query error (get_product_listings_for_env): {e}")
        finally:
            conn.close()
        return listings

    def get_posts_communication_data(self) -> list:
        """Query all posts from database and return raw data as list of dicts."""
        if not os.path.exists(self.db_path):
            return []
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        posts_data = []
        try:
            # Query all posts (excluding reposts, only original posts and quotes)
            cursor.execute(
                "SELECT post_id, user_id, content, useful_info, created_at, num_likes, num_dislikes "
                "FROM post WHERE original_post_id IS NULL ORDER BY created_at DESC LIMIT 50"
            )
            posts = cursor.fetchall()
            for p in posts:
                posts_data.append({
                    "post_id": p[0],
                    "user_id": p[1],
                    "content": p[2],
                    "useful_info": p[3],
                    "created_at": p[4],
                    "num_likes": p[5],
                    "num_dislikes": p[6]
                })
        except sqlite3.Error as e:
            print(f"Database query error (get_posts_communication_data): {e}")
        finally:
            conn.close()
        return posts_data

    def get_posts_communication_for_env(self) -> str:
        """Query all posts from database and format as string."""
        posts_data = self.get_posts_communication_data()
        if not posts_data:
            return "No posts are currently available."
        if self.conditions == "siloed": # siloed mode: only show posts from itself
            posts_data = [p for p in posts_data if p['user_id'] == self.action.agent_id]
        elif self.conditions == "shared": # shared mode: show all posts from all buyers
            posts_data = posts_data
        posts_env = "Here is the list of recent posts from other buyers:\n"
        for p in posts_data:
            useful_info_str = f" (Useful Info: {p['useful_info']})" if p['useful_info'] else ""
            posts_env += (
                f"- Post ID: {p['post_id']}, User ID: {p['user_id']}, Content: {p['content']}{useful_info_str}, "
                f"Likes: {p['num_likes']}, Dislikes: {p['num_dislikes']}\n"
            )
        return posts_env
    

    async def get_posts_env(self) -> str:
        posts = await self.action.refresh()
        # TODO: Replace posts json format string to other formats
        if posts["success"]:
            posts_env = json.dumps(posts["posts"], indent=4)
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
        return posts_env

    async def get_followers_env(self) -> str:
        # TODO: Implement followers env
        agent_id = self.action.agent_id
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT num_followers FROM user WHERE agent_id = ?",
                           (agent_id, ))
            result = cursor.fetchone()
            num_followers = result[0] if result else 0
            conn.close()
        except Exception:
            num_followers = 0
        return self.followers_env_template.substitute(
            {"num_followers": num_followers})

    async def get_follows_env(self) -> str:
        # TODO: Implement follows env
        agent_id = self.action.agent_id
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT num_followings FROM user WHERE agent_id = ?",
                (agent_id, ))
            result = cursor.fetchone()
            num_followings = result[0] if result else 0
            conn.close()
        except Exception:
            num_followings = 0
        return self.follows_env_template.substitute(
            {"num_follows": num_followings})

    async def get_group_env(self) -> str:
        groups = await self.action.listen_from_group()
        if groups["success"]:
            all_groups = json.dumps(groups["all_groups"])
            joined_groups = json.dumps(groups["joined_groups"])
            messages = json.dumps(groups["messages"])
            groups_env = self.groups_env_template.substitute(
                all_groups=all_groups,
                joined_groups=joined_groups,
                messages=messages,
            )
        else:
            groups_env = "No groups."
        return groups_env

    async def get_market_environment(self, agent=None, current_round=1, market_phase="general") -> str:
        """Get environment information for market simulation, providing different observation content based on different market phases."""
        from prompt import MarketEnv_prompt
        
        if not agent:
            # If no agent information, return basic environment information
            return f"Current Round: {current_round}/7\nNo agent information available."
        
        role = agent.user_info.profile.get("role")
        
        if market_phase == "communication":
            # Use direct database query instead of refresh() for communication phase
            posts_env = self.get_posts_communication_for_env()
            return posts_env
        
        elif market_phase == "listing" and role == "seller":
            # Seller in listing phase: observe previous phase purchase feedback and current market status
            previous_feedback = self._get_previous_feedback(agent)
            total_profit = getattr(agent, 'total_profit', 0)
            
            return MarketEnv_prompt.SELLER_LISTING_ENV.format(
                previous_feedback=previous_feedback,
                current_round=current_round,
                reputation_score=agent.reputation_score,
                total_profit=total_profit,
            )
            
        elif market_phase == "purchase" and role == "buyer":
            # Buyer in purchase phase: observe currently purchasable products and seller information
            available_products = self.get_product_listings_for_env()
            seller_reputation_info = self._get_seller_reputation_info()
            
            return MarketEnv_prompt.BUYER_PURCHASE_ENV.format(
                current_round=current_round,
                cumulative_utility=agent.cumulative_utility,
                available_products=available_products,
                seller_reputation_info=seller_reputation_info
            )
            
        elif market_phase == "rating" and role == "buyer":
            # Buyer in rating phase: observe specific product information after purchase
            purchase_info = getattr(agent, 'last_purchase_info', {})
            
            return MarketEnv_prompt.BUYER_RATING_ENV.format(
                transaction_id=purchase_info.get('transaction_id', 'N/A'),
                product_id=purchase_info.get('product_id', 'N/A'),
                advertised_quality=purchase_info.get('advertised_quality', 'N/A'),
                true_quality=purchase_info.get('true_quality', 'N/A'),
                has_warrant=purchase_info.get('has_warrant', False),
                purchase_price=purchase_info.get('purchase_price', 0),
                buyer_utility=purchase_info.get('buyer_utility', 0),
                seller_id=purchase_info.get('seller_id', 'N/A'),
                seller_reputation=purchase_info.get('seller_reputation', 0)
            )
            
        else:
            # Other cases return basic environment information
            return f"Current Round: {current_round}/7\nRole: {role}, Phase: {market_phase}\nNo specific environment template available."
    
    def _get_previous_feedback(self, agent) -> str:
        """Get previous phase historical performance feedback information"""
        # Get historical feedback information from agent's history_summary attribute
        # This attribute is updated through sellers_history in run_single_simulation.py
        if hasattr(agent, 'history_summary') and agent.history_summary:
            # Check if it's a hidden message during initial window period
            if agent.history_summary == "History hidden in initial window.":
                return "This is within the initial window period. Historical performance data is not available yet."
            return agent.history_summary
        else:
            return "No previous feedback available. This is your first round."
    
    def _get_seller_reputation_info(self) -> str:
        """Get seller reputation information"""
        # Should get all sellers' reputation information from database or platform here
        # Return simulated data for now
        return "Seller reputation information can't be view."

    async def to_text_prompt(
        self,
        include_posts: bool = True,
        include_followers: bool = True,
        include_follows: bool = True,
        agent=None,
        current_round=1,
        market_phase="general",
    ) -> str:

        if self.is_market_sim:
            return await self.get_market_environment(agent, current_round, market_phase)
        else:
            followers_env = (await self.get_followers_env()
                            if include_follows else "No followers.")
            follows_env = (await self.get_follows_env()
                        if include_followers else "No follows.")
            posts_env = await self.get_posts_env() if include_posts else ""
        
            return self.env_template.substitute(
                followers_env=followers_env,
                follows_env=follows_env,
                posts_env=posts_env,
                groups_env=await self.get_group_env(),
            )
