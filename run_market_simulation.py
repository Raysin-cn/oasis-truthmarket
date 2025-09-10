import asyncio
import sqlite3
import os
import oasis
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from oasis import SocialAgent, AgentGraph, UserInfo, make
from oasis.environment.env_action import LLMAction
from oasis.social_platform.typing import ActionType


# 实验配置

TOTAL_AGENTS = 6
NUM_SELLERS = 3
NUM_BUYERS = 3
SIMULATION_ROUNDS = 7
DATABASE_PATH = 'market_sim.db'


def get_agent_state(agent_id: int, role: str, round_num: int = -1) -> dict:
    """获取智能体在特定回合或当前的状态。"""
    if not os.path.exists(DATABASE_PATH):
        # 如果是卖家，返回初始预算和声誉
        if role == 'seller':
            return {'current_budget': 100.0, 'reputation_score': 1}
        # 如果是买家，返回初始效用
        else:
            return {'cumulative_utility': 0}

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    state = {}
    try:
        if role == 'seller':
            # 如果提供了有效的 round_num，则查询该回合的 trace 记录来回溯状态
            cursor.execute(
                "SELECT budget, reputation_score FROM user WHERE user_id = ?",
                (agent_id,)
            )
            result = cursor.fetchone()
            state['current_budget'] = result[0] if result else 100.0
            state['reputation_score'] = result[1] if result else 1
        
        elif role == 'buyer':
            cursor.execute(
                "SELECT profit_utility_score FROM user WHERE user_id = ?",
                (agent_id,)
            )
            result = cursor.fetchone()
            state['cumulative_utility'] = result[0] if result and result[0] is not None else 0
    
    except sqlite3.Error as e:
        print(f"数据库查询错误 (get_agent_state): {e}")
        # 出错时返回默认值
        if role == 'seller':
            state = {'current_budget': 100.0, 'reputation_score': 1}
        else:
            state = {'cumulative_utility': 0}
    finally:
        conn.close()
        
    return state

def get_product_listings() -> str:
    if not os.path.exists(DATABASE_PATH):
        return "No products are currently on sale."
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    listings = "No products are currently on sale."
    try:
        cursor.execute("PRAGMA table_info(post)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'status' not in columns:
            return "Database schema is incorrect: 'post' table is missing the 'status' column."
        cursor.execute("SELECT post_id, user_id, advertised_quality, price, has_warrant FROM post WHERE status = 'on_sale'")
        products = cursor.fetchall()
        if products:
            listings = "Here is the list of products currently on sale:\n"
            for p in products:
                warrant_info = " (Warranted)" if p[4] else ""
                listings += f"- Product ID: {p[0]}, Seller ID: {p[1]}, Advertised Quality: {p[2]}, Price: ${p[3]:.2f}{warrant_info}\n"
    except sqlite3.Error as e:
        print(f"数据库查询错误 (get_product_listings): {e}")
    finally:
        conn.close()
    return listings

def get_seller_round_summary(seller_id: int, round_num: int) -> dict:
    """获取卖家在特定回合的上架商品信息和销售状态。"""
    summary = {"quality": "Did not list", "sold": "No", "price": 0}
    if not os.path.exists(DATABASE_PATH):
        return summary
        
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        # 在 post 表中按 user_id 和 round_number 查询
        cursor.execute(
            "SELECT advertised_quality, is_sold, price FROM post WHERE user_id = ? AND round_number = ? ORDER BY post_id DESC LIMIT 1",
            (seller_id, round_num) # 查询当前回合 (round_num)
        )
        result = cursor.fetchone()
        if result:
            summary["quality"] = result[0]
            summary["sold"] = "Yes" if result[1] else "No"
            summary["price"] = result[2] if result[1] else 0
    except sqlite3.Error as e:
        print(f"数据库查询错误 (get_seller_round_summary): {e}")
    finally:
        conn.close()
    return summary

def clear_market():
    """将所有在售商品的状态更新为'expired'，实现市场清空。"""
    if not os.path.exists(DATABASE_PATH):
        return
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        # 将所有状态为 'on_sale' 的商品更新为 'expired'
        cursor.execute("UPDATE post SET status = 'expired' WHERE status = 'on_sale'")
        conn.commit()
        # 获取被影响的行数，用于调试
        changes = conn.total_changes
        print(f"Market cleared: {changes} unsold products have been removed from sale.")
    except sqlite3.Error as e:
        print(f"数据库错误 (clear_market): {e}")
    finally:
        conn.close()

def initialize_market_roles(agent_graph: AgentGraph):
    """在数据库中为所有智能体设置角色和初始状态。"""
    print("Initializing market roles in the database...")
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        for agent_id, agent in agent_graph.get_agents():
            role = agent.user_info.profile.get("other_info", {}).get("role")
            if role == 'seller':
                # 为卖家设置初始预算和声誉
                cursor.execute(
                    "UPDATE user SET role = ?, budget = ?, reputation_score = ?, profit_utility_score = ? WHERE agent_id = ?",
                    ('seller', 100.0, 1, 0.0, agent_id) 
                )
            elif role == 'buyer':
                # 为买家设置角色和初始分数
                cursor.execute(
                    "UPDATE user SET role = ?, profit_utility_score = ? WHERE agent_id = ?",
                    ('buyer', 0.0, agent_id)
                )
        conn.commit()
        print("Market roles initialized successfully.")
    except sqlite3.Error as e:
        print(f"数据库错误 (initialize_market_roles): {e}")
    finally:
        conn.close()
                
async def main():
    print("Starting market simulation initialization...")
    
    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)
    
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        api_key="sk-I9Af4lxw1jSCHbOBtV7BikArScsQoTlCRliJo5pQjeHlwpuc", 
        url="https://api.mnapi.com/v1", 
    )

    agent_graph = AgentGraph()
    
    seller_personas = [
        "You are an honest seller. Your primary goal is to build a long-term, trustworthy business. You believe that selling High-Quality (HQ) products is the best way to achieve a high reputation and sustainable profit, even if it means lower margins in the short term. You will always produce HQ products and advertise them truthfully.",
        "You are a deceptive seller. Your main goal is to maximize immediate profit, even at the risk of your reputation. You see advertising LQ products as HQ as a highly profitable, albeit risky, strategy. You will try to deceive buyers whenever you think you can get away with it, especially if your reputation is not yet negative.",
        "You are a strategic seller. You balance the pursuit of profit with the need to maintain a decent reputation. You might honestly sell HQ products to build up your reputation, but you are not against selling a deceptive LQ product if you feel the market conditions are right (e.g., your reputation is high enough to absorb a potential hit)."
    ]
    
    buyer_personas = [
        "You are a pragmatic buyer focused on maximizing utility. You are willing to purchase from sellers with low or neutral reputation, especially if the product has a warrant, as you understand this is how a new market starts.",
        "You are an analytical buyer. Your primary goal is to increase your utility score. You analyze the available products and are willing to make a purchase if the potential utility gain is positive, even from new sellers.",
        "You are a strategic buyer. You understand that rating sellers helps build a healthy market. You are willing to be the first to buy a product to rate the seller and gain information for future rounds."
    ]
    
    for i in range(TOTAL_AGENTS):
        if i < NUM_SELLERS:
            role = 'seller'
            persona = seller_personas[i % len(seller_personas)]
            available_actions = ActionType.get_seller_actions()
        else:
            role = 'buyer'
            persona = buyer_personas[(i - NUM_SELLERS) % len(buyer_personas)]
            available_actions = ActionType.get_buyer_actions()
        
        user_info = UserInfo(
            user_name=f"agent_{i}",
            name=f"Agent {i} ({role.capitalize()})",
            description=f"A {role} in the market.",
            profile={
                "other_info": {
                    "user_profile": persona,
                    "role": role,
                }
            }
        )
        
        agent = SocialAgent(
            agent_id=i,
            user_info=user_info,
            model=model,
            agent_graph=agent_graph,
            available_actions=available_actions, 
        )
        agent.env.is_market_sim =True
        agent_graph.add_agent(agent)

    env = make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT, 
        database_path=DATABASE_PATH
    )
    await env.reset()
    print(f"Environment initialized. Database at '{DATABASE_PATH}'.")
    initialize_market_roles(agent_graph)
    sellers_history = {i: [] for i in range(NUM_SELLERS)}
    
    
    for round_num in range(1, SIMULATION_ROUNDS + 1):
        print(f"\n{'='*20} Starting Round {round_num}/{SIMULATION_ROUNDS} {'='*20}")
        
        # 卖家行动
        print(f"\n--- [Round {round_num}] Seller Action Phase ---")
        seller_actions = {}
        for agent_id, agent in agent_graph.get_agents():
            if agent.user_info.profile.get("other_info", {}).get("role") == 'seller':
                state = get_agent_state(agent_id, 'seller')
                
                # 格式化历史记录为字符串
                history_log = sellers_history.get(agent_id, [])
                if not history_log:
                    history_string = "This is the first round. You have no past performance data."
                else:
                    history_string = "Here is a summary of your performance in previous rounds:\n"
                    for entry in history_log:
                        history_string += f"- Round {entry['round']}: Listed a {entry['quality']} product. Sold: {entry['sold']}. Round Profit: {entry['profit']}. New Reputation: {entry['reputation']}\n"
                
                # 更新系统Prompt
                prompt_template = agent.user_info.to_seller_master_prompt()
                agent.system_message.content = prompt_template.format(
                    current_round=round_num,
                    current_budget=state['current_budget'],
                    reputation_score=state['reputation_score'],
                    history_summary=history_string 
                )
                seller_actions[agent] = LLMAction()
        
        if seller_actions:
            await env.step(seller_actions)
        print("All seller actions are complete.")

        print(f"\n--- [Round {round_num}] Buyer Action Phase 1: Purchase ---")
        buyer_actions = {}
        product_listings_for_this_round = get_product_listings()
        

        for agent_id, agent in agent_graph.get_agents():
             if agent.user_info.profile.get("other_info", {}).get("role") == 'buyer':
                state = get_agent_state(agent_id, 'buyer')
                prompt_template = agent.user_info.to_buyer_master_prompt()
                
                agent.system_message.content = prompt_template.format(
                    current_round=round_num,
                    cumulative_utility=state['cumulative_utility'],
                    product_listings=product_listings_for_this_round
                )
                buyer_actions[agent] = LLMAction()
        
        purchase_results = []
        if buyer_actions:
            purchase_results = await env.step(buyer_actions)
        print("All purchase actions are attempted.")

        # 挑战与评价 
        print(f"\n--- [Round {round_num}] Buyer Action Phase 2: Challenge & Rate ---")
        post_purchase_actions = {}
        
        successful_purchases = [res for res in purchase_results if res and res.get("success")]

        if successful_purchases:
            for purchase_info in successful_purchases:
                agent_id = purchase_info.get("agent_id")
                if agent_id is None: continue
                
                agent = agent_graph.get_agent(agent_id)

                prompt_template = agent.user_info.to_buyer_post_purchase_prompt()
                agent.system_message.content = prompt_template.format(
                    transaction_id=purchase_info.get("transaction_id"),
                    post_id=purchase_info.get("post_id"),
                    advertised_quality=purchase_info.get("advertised_quality"),
                    true_quality=purchase_info.get("true_quality"),
                    has_warrant=purchase_info.get("has_warrant")
                )
                post_purchase_actions[agent] = LLMAction()
        
        if post_purchase_actions:
            await env.step(post_purchase_actions)
        print("All post-purchase actions are complete.")

        clear_market()
        
        #  更新历史记录 
        for seller_id in range(NUM_SELLERS):
            round_summary = get_seller_round_summary(seller_id, round_num)
            new_state = get_agent_state(seller_id, 'seller') 

            profit = 0
            if round_summary["sold"] == "Yes":
                profit = round_summary["price"] 

            sellers_history[seller_id].append({
                "round": round_num,
                "quality": round_summary["quality"], 
                "sold": round_summary["sold"], 
                "profit": profit,
                "reputation": new_state['reputation_score']
            })

        print(f"\n{'='*20} End of Round {round_num} {'='*20}")

    
    await env.close()
    print("\nSimulation finished")

if __name__ == "__main__":
    asyncio.run(main())