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
from enum import Enum


class ActionType(Enum):
    EXIT = "exit"
    REFRESH = "refresh"
    SEARCH_USER = "search_user"
    SEARCH_POSTS = "search_posts"
    CREATE_POST = "create_post"
    LIKE_POST = "like_post"
    UNLIKE_POST = "unlike_post"
    DISLIKE_POST = "dislike_post"
    UNDO_DISLIKE_POST = "undo_dislike_post"
    REPORT_POST = "report_post"
    FOLLOW = "follow"
    UNFOLLOW = "unfollow"
    MUTE = "mute"
    UNMUTE = "unmute"
    TREND = "trend"
    SIGNUP = "sign_up"
    REPOST = "repost"
    QUOTE_POST = "quote_post"
    UPDATE_REC_TABLE = "update_rec_table"
    CREATE_COMMENT = "create_comment"
    LIKE_COMMENT = "like_comment"
    UNLIKE_COMMENT = "unlike_comment"
    DISLIKE_COMMENT = "dislike_comment"
    UNDO_DISLIKE_COMMENT = "undo_dislike_comment"
    DO_NOTHING = "do_nothing"
    PURCHASE_PRODUCT = "purchase_product"
    INTERVIEW = "interview"
    JOIN_GROUP = "join_group"
    LEAVE_GROUP = "leave_group"
    SEND_TO_GROUP = "send_to_group"
    CREATE_GROUP = "create_group"
    LISTEN_FROM_GROUP = "listen_from_group"
    LIST_PRODUCT = "list_product"            # 卖家上架商品
    EXIT_MARKET = "exit_market"              # 卖家退出市场
    REENTER_MARKET = "reenter_market"        # 卖家重新进入市场
    PURCHASE_PRODUCT_ID = "purchase_product_id"    # 买家购买商品
    CHALLENGE_WARRANT = "challenge_warrant"  # 买家挑战保证金 
    RATE_TRANSACTION = "rate_transaction"    # 买家评价交易 

    @classmethod
    def get_default_twitter_actions(cls):
        return [
            cls.CREATE_POST,
            cls.LIKE_POST,
            cls.REPOST,
            cls.FOLLOW,
            cls.DO_NOTHING,
            cls.QUOTE_POST,
        ]

    @classmethod
    def get_default_reddit_actions(cls):
        return [
            cls.LIKE_POST,
            cls.DISLIKE_POST,
            cls.CREATE_POST,
            cls.CREATE_COMMENT,
            cls.LIKE_COMMENT,
            cls.DISLIKE_COMMENT,
            cls.SEARCH_POSTS,
            cls.SEARCH_USER,
            cls.TREND,
            cls.REFRESH,
            cls.DO_NOTHING,
            cls.FOLLOW,
            cls.MUTE,
        ]
        
    @classmethod
    def get_seller_actions(cls):
        """返回一个只包含卖家专属动作的列表。"""
        return [
            cls.LIST_PRODUCT,
            cls.EXIT_MARKET,
            cls.REENTER_MARKET,
        ]

    @classmethod
    def get_buyer_actions(cls):
        """返回一个只包含买家专属动作的列表。"""
        return [
            cls.PURCHASE_PRODUCT_ID,
            cls.CHALLENGE_WARRANT,
            cls.RATE_TRANSACTION,
        ]
    
    @classmethod
    def get_market_actions(cls, market_type: str = 'reputation_and_warrant'):
        """根据市场类型返回所有相关的动作列表。"""
        if market_type == 'reputation_only':
            return [
                cls.LIST_PRODUCT,
                cls.EXIT_MARKET,
                cls.REENTER_MARKET,
                cls.PURCHASE_PRODUCT_ID,
                cls.RATE_TRANSACTION,
            ]
        else:  # 默认 'reputation_and_warrant'
            return [
                cls.LIST_PRODUCT,
                cls.EXIT_MARKET,
                cls.REENTER_MARKET,
                cls.PURCHASE_PRODUCT_ID,
                cls.CHALLENGE_WARRANT,
                cls.RATE_TRANSACTION,
            ]    


class RecsysType(Enum):
    TWITTER = "twitter"
    TWHIN = "twhin-bert"
    REDDIT = "reddit"
    RANDOM = "random"


class DefaultPlatformType(Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
