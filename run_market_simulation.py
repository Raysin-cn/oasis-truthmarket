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
from prompt import format_seller_history, SELLER_GENERATION_SYS_PROMPT, SELLER_GENERATION_USER_PROMPT, BUYER_GENERATION_SYS_PROMPT, BUYER_GENERATION_USER_PROMPT, SELLER_ROUND_PROMPT, BUYER_ROUND_PROMPT
from utils import print_round_statistics, clear_market, print_simulation_summary
from oasis.environment.processing.reputation import compute_and_update_reputation
from oasis.environment.processing.valunerability import run_detection

from dotenv import load_dotenv
load_dotenv(override=True)


# Experiment configuration

TOTAL_AGENTS = 6
NUM_SELLERS = 10
NUM_BUYERS = 10
SIMULATION_ROUNDS = 7
DATABASE_PATH = 'market_sim.db'
os.environ.setdefault('MARKET_DB_PATH', DATABASE_PATH)

# Hyperparameters (based on proposals)
REPUTATION_LAG = 1  # ratings from round t available to public at t+REPUTATION_LAG
REENTRY_ALLOWED_ROUND = 5
INITIAL_WINDOW_ROUNDS = [1, 2]  # rounds to hide full history
EXIT_ROUND = 7  # after round 7 sellers may exit
MARKET_TYPE = 'reputation_only'   #reputation_and_warrant or reputation_only

def get_agent_state(agent_id: int, role: str, round_num: int = -1) -> dict:
    """Get the state of an agent at a specific round or current state."""
    if not os.path.exists(DATABASE_PATH):
        # If seller, return initial budget and reputation
        if role == 'seller':
            return {'current_budget': 100.0, 'reputation_score': 0, 'total_profit': 0}
        # If buyer, return initial utility
        else:
            return {'cumulative_utility': 0, 'total_utility': 0}

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    state = {}
    try:
        if role == 'seller':
            # Get seller basic information
            cursor.execute(
                "SELECT budget, reputation_score, profit_utility_score FROM user WHERE user_id = ?",
                (agent_id,)
            )
            result = cursor.fetchone()
            state['current_budget'] = result[0] if result else 100.0
            state['reputation_score'] = result[1] if result else 0
            state['total_profit'] = result[2] if result and result[2] is not None else 0
            
            # Get sales information for this round
            if round_num > 0:
                cursor.execute(
                    "SELECT COUNT(*), SUM(seller_profit) FROM transactions WHERE seller_id = ? AND round_number = ?",
                    (agent_id, round_num)
                )
                sales_result = cursor.fetchone()
                state['round_sales'] = sales_result[0] if sales_result else 0
                state['round_profit'] = sales_result[1] if sales_result and sales_result[1] is not None else 0
        
        elif role == 'buyer':
            # Get buyer basic information
            cursor.execute(
                "SELECT profit_utility_score FROM user WHERE user_id = ?",
                (agent_id,)
            )
            result = cursor.fetchone()
            state['cumulative_utility'] = result[0] if result and result[0] is not None else 0
            state['total_utility'] = result[0] if result and result[0] is not None else 0
            
            # Get purchase information for this round
            if round_num > 0:
                cursor.execute(
                    "SELECT COUNT(*), SUM(buyer_utility) FROM transactions WHERE buyer_id = ? AND round_number = ?",
                    (agent_id, round_num)
                )
                purchase_result = cursor.fetchone()
                state['round_purchases'] = purchase_result[0] if purchase_result else 0
                state['round_utility'] = purchase_result[1] if purchase_result and purchase_result[1] is not None else 0
    
    except sqlite3.Error as e:
        print(f"Database query error (get_agent_state): {e}")
        # Return default values on error
        if role == 'seller':
            state = {'current_budget': 100.0, 'reputation_score': 0, 'total_profit': 0}
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
        cursor.execute(
            """
            SELECT p.post_id, p.user_id, p.advertised_quality, p.price, p.has_warrant, 
                   COALESCE(u.reputation_score, 0) AS reputation_score
            FROM post p
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
        print(f"Database query error (get_product_listings): {e}")
    finally:
        conn.close()
    return listings

def get_seller_round_summary(seller_id: int, round_num: int) -> dict:
    """Get seller's product listing information and sales status for a specific round."""
    summary = {"quality": "Did not list", "sold": "No", "price": 0}
    if not os.path.exists(DATABASE_PATH):
        return summary
        
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        # Query in post table by user_id and round_number
        cursor.execute(
            "SELECT advertised_quality, is_sold, price FROM post WHERE user_id = ? AND round_number = ? ORDER BY post_id DESC LIMIT 1",
            (seller_id, round_num) # Query current round (round_num)
        )
        result = cursor.fetchone()
        if result:
            summary["quality"] = result[0]
            summary["sold"] = "Yes" if result[1] else "No"
            summary["price"] = result[2] if result[1] else 0
    except sqlite3.Error as e:
        print(f"Database query error (get_seller_round_summary): {e}")
    finally:
        conn.close()
    return summary


def initialize_market_roles(agent_graph: AgentGraph):
    """Set roles and initial states for all agents in the database."""
    print("Initializing market roles in the database...")
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        for agent_id, agent in agent_graph.get_agents():
            role = agent.user_info.profile.get("other_info", {}).get("role")
            if role == 'seller':
                # Set initial budget and reputation for seller
                cursor.execute(
                    "UPDATE user SET role = ?, budget = ?, reputation_score = ?, profit_utility_score = ? WHERE agent_id = ?",
                    ('seller', 100.0, 0, 0.0, agent_id) 
                )
            elif role == 'buyer':
                # Set role and initial score for buyer
                cursor.execute(
                    "UPDATE user SET role = ?, profit_utility_score = ? WHERE agent_id = ?",
                    ('buyer', 0.0, agent_id)
                )
        conn.commit()
        print("Market roles initialized successfully.")
    except sqlite3.Error as e:
        print(f"Database error (initialize_market_roles): {e}")
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


    # Generate seller agents
    print("Generating seller agents...")
    seller_agent_graph = await generate_agent_from_LLM(
        agents_num=NUM_SELLERS,
        sys_prompt=SELLER_GENERATION_SYS_PROMPT,
        user_prompt=SELLER_GENERATION_USER_PROMPT,
        available_actions=ActionType.get_market_actions(MARKET_TYPE), 
        role="seller",
    )
    
    # Generate buyer agents
    print("Generating buyer agents...")
    buyer_agent_graph = await generate_agent_from_LLM(
        agents_num=NUM_BUYERS,
        sys_prompt=BUYER_GENERATION_SYS_PROMPT,
        user_prompt=BUYER_GENERATION_USER_PROMPT,
        available_actions=ActionType.get_market_actions(MARKET_TYPE), 
        role="buyer"
    )
    
    # Merge agent graphs
    
    # Add seller agents
    for agent_id, agent in seller_agent_graph.get_agents():
        agent.user_info.profile["other_info"]["role"] = "seller"
        agent.user_info.market_type = MARKET_TYPE
        agent.env.is_market_sim = True
        agent_graph.add_agent(agent)
    
    # Add buyer agents (reassign agent_id)
    for agent_id, agent in buyer_agent_graph.get_agents():
        agent.user_info.profile["other_info"]["role"] = "buyer"
        agent.user_info.market_type = MARKET_TYPE
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
        # Synchronize platform round counter
        env.platform.sandbox_clock.round_step = round_num
        print(f"\n{'='*20} Starting Round {round_num}/{SIMULATION_ROUNDS} {'='*20}")

        # Seller exit mechanism: Allow exit after EXIT_ROUND (soft flag, actual implementation can be on platform side)
        if round_num > EXIT_ROUND:
            print("Sellers may exit market. (soft flag)")
        
        # Seller actions
        print(f"\n--- [Round {round_num}] Seller Action Phase ---")
        seller_actions = {}
        
        # Set environment market phase to listing
        env.market_phase = "listing"
        
        for agent_id, agent in agent_graph.get_agents():
            if agent.user_info.profile.get("other_info", {}).get("role") == 'seller':
                state = get_agent_state(agent_id, 'seller')

                # Hide complete history in initial window (show only aggregated/summary or empty)
                history_log = sellers_history.get(agent_id, [])
                if round_num in INITIAL_WINDOW_ROUNDS:
                    visible_history_string = "History hidden in initial window."
                else:
                    visible_history_string = format_seller_history(history_log)
                
                # Update environment state
                env.current_round = round_num
                
                # Update agent state attributes
                agent.current_budget = state['current_budget']
                agent.reputation_score = state['reputation_score']
                agent.history_summary = visible_history_string
                
                # System prompt already set during SocialAgent instantiation (static parameters)
                # Prepare round prompt (dynamic parameters)
                round_prompt = SELLER_ROUND_PROMPT.format(
                    history_summary=visible_history_string
                )
                # Tools available for sellers in listing phase
                listing_tools = [name_to_tool['list_product']] if 'list_product' in name_to_tool else []
                
                # Conditionally inject additional tools: expose exit/re-enter market actions when allowed
                if round_num >= EXIT_ROUND:
                    listing_tools.append(name_to_tool['exit_market'])
                if round_num == REENTRY_ALLOWED_ROUND:
                    listing_tools.append(name_to_tool['reenter_market'])

                seller_actions[agent] = LLMAction(
                    extra_action=listing_tools,
                    extra_prompt=round_prompt
                )
        
        if seller_actions:
            await env.step(seller_actions)
        print("All seller actions are complete.")

        print(f"\n--- [Round {round_num}] Buyer Action Phase 1: Purchase ---")
        buyer_actions = {}
        product_listings_for_this_round = get_product_listings()
        
        # Set environment market phase to purchase
        env.market_phase = "purchase"

        for agent_id, agent in agent_graph.get_agents():
             if agent.user_info.profile.get("other_info", {}).get("role") == 'buyer':
                state = get_agent_state(agent_id, 'buyer')
                # Update agent state attributes
                agent.cumulative_utility = state['cumulative_utility']
                
                # System prompt already set during SocialAgent instantiation (static parameters)
                # Prepare round prompt (dynamic parameters)
                round_prompt = BUYER_ROUND_PROMPT.format()
                # Tools available for buyers in purchase phase: only allow purchase
                purchase_tools = [name_to_tool['purchase_product_id']] if 'purchase_product_id' in name_to_tool else []
                buyer_actions[agent] = LLMAction(
                    extra_action=purchase_tools,
                    extra_prompt=round_prompt
                )
        
        purchase_results = []
        if buyer_actions:
            purchase_results = await env.step(buyer_actions)
        print("All purchase actions are attempted.")

        # Seller re-entry mechanism: After REENTRY_ALLOWED_ROUND, allow marked manipulators/low-reputation sellers to re-enter (interface left here, actual control can depend on platform layer)
        if round_num == REENTRY_ALLOWED_ROUND:
            print("Re-entry policy active for low-reputation/manipulators. (soft flag)")

        # Challenge and rating 
        print(f"\n--- [Round {round_num}] Buyer Action Phase 2: Challenge & Rate ---")
        post_purchase_actions = {}
        
        # Set environment market phase to rating
        env.market_phase = "rating"
        
        successful_purchases = [res for res in purchase_results if res and res.get("success")]

        if successful_purchases:
            for purchase_info in successful_purchases:
                agent_id = purchase_info.get("agent_id")
                if agent_id is None: continue
                
                agent = agent_graph.get_agent(agent_id)

                # Tools available for buyers in rating phase: only allow challenge warrant and rating
                rating_tools = []
                # rating_tools.append(name_to_tool['challenge_warrant'])  # Temporarily disable challenge function
                if 'rate_transaction' in name_to_tool:
                    rating_tools.append(name_to_tool['rate_transaction'])
                
                # Store purchase information in agent for environment observation
                agent.last_purchase_info = {
                    'transaction_id': purchase_info.get("transaction_id"),
                    'post_id': purchase_info.get("post_id"),
                    'advertised_quality': purchase_info.get("advertised_quality"),
                    'true_quality': purchase_info.get("true_quality"),
                    'has_warrant': purchase_info.get("has_warrant"),
                    'buyer_utility': purchase_info.get("buyer_utility"),
                    'purchase_price': purchase_info.get("purchase_price", 0),
                    'seller_id': purchase_info.get("seller_id", 'N/A'),
                    'seller_reputation': purchase_info.get("seller_reputation", 0)
                }
                
                post_purchase_actions[agent] = LLMAction(
                    extra_action=rating_tools,
                    extra_prompt=""  # Environment observation information will be provided through market_phase="rating"
                )
        
        if post_purchase_actions:
            await env.step(post_purchase_actions)
        print("All post-purchase actions are complete.")

        clear_market()
        
        # Print round statistics
        print_round_statistics(round_num)
        
        # Statistics and recording: update reputation history (apply lag display: in round r only show ratings up to r-REPUTATION_LAG)
        ratings_cutoff_round = max(0, round_num - REPUTATION_LAG)
        with sqlite3.connect(DATABASE_PATH) as conn:
            compute_and_update_reputation(conn, round_num, ratings_up_to_round=ratings_cutoff_round if ratings_cutoff_round > 0 else None)

        # Update history records 
        for seller_id in range(NUM_SELLERS):
            round_summary = get_seller_round_summary(seller_id, round_num)
            new_state = get_agent_state(seller_id, 'seller', round_num) 

            # Get actual profit for this round
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

    
    # Vulnerability/manipulation detection (run once after entire simulation)
    with sqlite3.connect(DATABASE_PATH) as conn:
        run_detection(SIMULATION_ROUNDS)
    print("Manipulation analysis completed.")
    
    await env.close()
    print_simulation_summary()
    print("\nSimulation finished")

if __name__ == "__main__":
    asyncio.run(main())