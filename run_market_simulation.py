import asyncio
import sqlite3
import os
import oasis
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from oasis import SocialAgent, AgentGraph, UserInfo, make
from oasis.environment.env_action import LLMAction
from oasis.social_platform.typing import ActionType
from oasis.social_agent.agents_generator import generate_agent_from_LLM
from prompt import format_seller_history, SELLER_GENERATION_SYS_PROMPT, SELLER_GENERATION_USER_PROMPT, BUYER_GENERATION_SYS_PROMPT, BUYER_GENERATION_USER_PROMPT
from utils import print_round_statistics, clear_market, print_simulation_summary
from oasis.environment.processing.reputation import compute_and_update_reputation
from oasis.environment.processing.valunerability import run_detection

from dotenv import load_dotenv
load_dotenv(override=True)


# 实验配置

TOTAL_AGENTS = 6
NUM_SELLERS = 10
NUM_BUYERS = 10
SIMULATION_ROUNDS = 7
DATABASE_PATH = 'market_sim.db'
os.environ.setdefault('MARKET_DB_PATH', DATABASE_PATH)

# 超参数（根据 proposals ）
REPUTATION_LAG = 2  # ratings from round t available to public at t+2
REENTRY_ALLOWED_ROUND = 5
INITIAL_WINDOW_ROUNDS = [1, 2]  # rounds to hide full history
EXIT_ROUND = 7  # after round 7 sellers may exit


def get_agent_state(agent_id: int, role: str, round_num: int = -1) -> dict:
    """获取智能体在特定回合或当前的状态。"""
    if not os.path.exists(DATABASE_PATH):
        # 如果是卖家，返回初始预算和声誉
        if role == 'seller':
            return {'current_budget': 100.0, 'reputation_score': 1, 'total_profit': 0}
        # 如果是买家，返回初始效用
        else:
            return {'cumulative_utility': 0, 'total_utility': 0}

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    state = {}
    try:
        if role == 'seller':
            # 获取卖家基本信息
            cursor.execute(
                "SELECT budget, reputation_score, profit_utility_score FROM user WHERE user_id = ?",
                (agent_id,)
            )
            result = cursor.fetchone()
            state['current_budget'] = result[0] if result else 100.0
            state['reputation_score'] = result[1] if result else 1
            state['total_profit'] = result[2] if result and result[2] is not None else 0
            
            # 获取该回合的销售情况
            if round_num > 0:
                cursor.execute(
                    "SELECT COUNT(*), SUM(seller_profit) FROM transactions WHERE seller_id = ? AND round_number = ?",
                    (agent_id, round_num)
                )
                sales_result = cursor.fetchone()
                state['round_sales'] = sales_result[0] if sales_result else 0
                state['round_profit'] = sales_result[1] if sales_result and sales_result[1] is not None else 0
        
        elif role == 'buyer':
            # 获取买家基本信息
            cursor.execute(
                "SELECT profit_utility_score FROM user WHERE user_id = ?",
                (agent_id,)
            )
            result = cursor.fetchone()
            state['cumulative_utility'] = result[0] if result and result[0] is not None else 0
            state['total_utility'] = result[0] if result and result[0] is not None else 0
            
            # 获取该回合的购买情况
            if round_num > 0:
                cursor.execute(
                    "SELECT COUNT(*), SUM(buyer_utility) FROM transactions WHERE buyer_id = ? AND round_number = ?",
                    (agent_id, round_num)
                )
                purchase_result = cursor.fetchone()
                state['round_purchases'] = purchase_result[0] if purchase_result else 0
                state['round_utility'] = purchase_result[1] if purchase_result and purchase_result[1] is not None else 0
    
    except sqlite3.Error as e:
        print(f"数据库查询错误 (get_agent_state): {e}")
        # 出错时返回默认值
        if role == 'seller':
            state = {'current_budget': 100.0, 'reputation_score': 1, 'total_profit': 0}
        else:
            state = {'cumulative_utility': 0, 'total_utility': 0}
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
        api_key=os.getenv("OPENAI_API_KEY"), 
        url=os.getenv("OPENAI_BASE_URL"), 
    )
    agent_graph = AgentGraph()


    # 生成卖家 agents
    print("Generating seller agents...")
    seller_agent_graph = await generate_agent_from_LLM(
        agents_num=NUM_SELLERS,
        sys_prompt=SELLER_GENERATION_SYS_PROMPT,
        user_prompt=SELLER_GENERATION_USER_PROMPT,
        available_actions=ActionType.get_seller_actions(),
        role="seller",
    )
    
    # 生成买家 agents
    print("Generating buyer agents...")
    buyer_agent_graph = await generate_agent_from_LLM(
        agents_num=NUM_BUYERS,
        sys_prompt=BUYER_GENERATION_SYS_PROMPT,
        user_prompt=BUYER_GENERATION_USER_PROMPT,
        available_actions=ActionType.get_buyer_actions(),
        role="buyer"
    )
    
    # 合并 agent graphs
    
    # 添加卖家 agents
    for agent_id, agent in seller_agent_graph.get_agents():
        agent.user_info.profile["other_info"]["role"] = "seller"
        agent.env.is_market_sim = True
        agent_graph.add_agent(agent)
    
    # 添加买家 agents (重新分配 agent_id)
    for agent_id, agent in buyer_agent_graph.get_agents():
        agent.user_info.profile["other_info"]["role"] = "buyer"
        agent.env.is_market_sim = True
        agent_graph.add_agent(agent)

    env = make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT, 
        database_path=DATABASE_PATH
    )
    await env.reset()
    available_tools = agent.env.action.get_openai_function_list()
    name_to_tool = {t.func.__name__: t for t in available_tools}
    
    print(f"Environment initialized. Database at '{DATABASE_PATH}'.")
    initialize_market_roles(agent_graph)
    sellers_history = {i: [] for i in range(NUM_SELLERS)}
    for round_num in range(1, SIMULATION_ROUNDS + 1):
        print(f"\n{'='*20} Starting Round {round_num}/{SIMULATION_ROUNDS} {'='*20}")

        # 卖家退场机制：超过 EXIT_ROUND 的回合后允许退出（此处仅标注，具体执行可在平台侧实现）
        if round_num > EXIT_ROUND:
            print("Sellers may exit market. (soft flag)")
        
        # 卖家行动
        print(f"\n--- [Round {round_num}] Seller Action Phase ---")
        seller_actions = {}
        for agent_id, agent in agent_graph.get_agents():
            if agent.user_info.profile.get("other_info", {}).get("role") == 'seller':
                state = get_agent_state(agent_id, 'seller')

                # 隐藏早期窗口内的完整历史（仅展示聚合/摘要或空）
                history_log = sellers_history.get(agent_id, [])
                if round_num in INITIAL_WINDOW_ROUNDS:
                    visible_history_string = "History hidden in initial window."
                else:
                    visible_history_string = format_seller_history(history_log)
                
                # 更新系统Prompt
                prompt_template = agent.user_info.to_seller_master_prompt()
                agent.system_message.content = prompt_template.format(
                    current_round=round_num,
                    current_budget=state['current_budget'],
                    reputation_score=state['reputation_score'],
                    history_summary=visible_history_string 
                )
                # 条件注入额外工具：在允许时机暴露退出/重返市场动作
                extra_tools = []
                # 退出：在达到或超过 EXIT_ROUND 时允许
                if round_num >= EXIT_ROUND and 'exit_market' in name_to_tool:
                    extra_tools.append(name_to_tool['exit_market'])
                # 重返：在达到或超过 REENTRY_ALLOWED_ROUND 且低声誉时允许
                if round_num >= REENTRY_ALLOWED_ROUND and state.get('reputation_score', 0) <= 0 and 'reenter_market' in name_to_tool:
                    extra_tools.append(name_to_tool['reenter_market'])

                seller_actions[agent] = LLMAction(extra_action=extra_tools if extra_tools else None)
        
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

        # 卖家重返机制：REENTRY_ALLOWED_ROUND 之后，允许被标记操纵/低声誉者重新进入（此处留接口，实际控制可依赖 platform 层）
        if round_num >= REENTRY_ALLOWED_ROUND:
            print("Re-entry policy active for low-reputation/manipulators. (soft flag)")

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
        
        # 打印回合统计信息
        print_round_statistics(round_num)
        
        # 统计与记录：更新声誉历史（应用滞后显示：在回合 r 只公开到 r-REPUTATION_LAG 的评分）
        ratings_cutoff_round = max(0, round_num - REPUTATION_LAG)
        with sqlite3.connect(DATABASE_PATH) as conn:
            compute_and_update_reputation(conn, round_num, ratings_up_to_round=ratings_cutoff_round if ratings_cutoff_round > 0 else None)

        #  更新历史记录 
        for seller_id in range(NUM_SELLERS):
            round_summary = get_seller_round_summary(seller_id, round_num)
            new_state = get_agent_state(seller_id, 'seller', round_num) 

            # 获取该回合的实际收益
            round_profit = new_state.get('round_profit', 0)
            total_profit = new_state.get('total_profit', 0)

            sellers_history[seller_id].append({
                "round": round_num,
                "quality": round_summary["quality"], 
                "sold": round_summary["sold"], 
                "profit": round_profit,
                "reputation": new_state['reputation_score'],
                "total_profit": total_profit
            })

        print(f"\n{'='*20} End of Round {round_num} {'='*20}")

    
    # 漏洞/操纵检测（整个模拟结束后一次性运行）
    with sqlite3.connect(DATABASE_PATH) as conn:
        run_detection(SIMULATION_ROUNDS)
    print("Manipulation analysis completed.")
    
    await env.close()
    print_simulation_summary()
    print("\nSimulation finished")

if __name__ == "__main__":
    asyncio.run(main())