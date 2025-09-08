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


def get_agent_state(agent_id: int, role: str) -> dict:
    if not os.path.exists(DATABASE_PATH):
        return {'current_budget': 100.0, 'reputation_score': 0, 'cumulative_utility': 0.0}
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    state = {'current_budget': 100.0, 'reputation_score': 0, 'cumulative_utility': 0.0}
    try:
        if role == 'seller':
            cursor.execute("SELECT budget, reputation_score FROM user WHERE agent_id = ?", (agent_id,))
            result = cursor.fetchone()
            if result:
                state['current_budget'] = result[0] if result[0] is not None else 100.0
                state['reputation_score'] = result[1] if result[1] is not None else 0
        elif role == 'buyer':
            cursor.execute("SELECT profit_utility_score FROM user WHERE agent_id = ?", (agent_id,))
            result = cursor.fetchone()
            if result:
                state['cumulative_utility'] = result[0] if result[0] is not None else 0.0
    except sqlite3.Error as e:
        print(f"数据库查询错误 (get_agent_state): {e}")
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
    """获取卖家在特定回合的上架商品信息。"""
    summary = {"quality": "Did not list", "sold": "N/A"}
    if not os.path.exists(DATABASE_PATH):
        return summary
        
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT advertised_quality, is_sold FROM post WHERE user_id = ? AND created_at = ? ORDER BY post_id DESC LIMIT 1",
            (seller_id, str(round_num - 1)) 
        )
        result = cursor.fetchone()
        if result:
            summary["quality"] = result[0]
            summary["sold"] = "Yes" if result[1] else "No"
    except sqlite3.Error as e:
        print(f"数据库查询错误 (get_seller_round_summary): {e}")
    finally:
        conn.close()
    return summary

async def main():
    print("Starting market simulation initialization...")

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        
    )

    agent_graph = AgentGraph()

    for i in range(TOTAL_AGENTS):
        if i < NUM_SELLERS:
            role = 'seller'
            persona = "You are a seller in an online marketplace. Your goal is to maximize your profit. You can choose to be honest or deceptive in your listings."
        else:
            role = 'buyer'
            persona = "You are a buyer in an online marketplace. Your goal is to maximize your utility by purchasing valuable products while avoiding scams."
        
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
            available_actions=ActionType.get_warrant_market_actions(),
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

        # 买家行动
        print(f"\n--- [Round {round_num}] Buyer Action Phase ---")
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
        if buyer_actions:
            await env.step(buyer_actions)
        print("All buyer actions are complete.")
        
        for seller_id in range(NUM_SELLERS):
            
            new_state = get_agent_state(seller_id, 'seller')
            round_summary = get_seller_round_summary(seller_id, round_num)

            profit = new_state['current_budget'] - 100 
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