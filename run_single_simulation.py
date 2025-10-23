"""
Single market simulation run module
Core simulation logic extracted from run_market_simulation.py
"""

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
from config import SimulationConfig

from dotenv import load_dotenv
load_dotenv(override=True)


def reset_agent_id_counter():
    """Reset agent ID counter to ensure each run starts from 1"""
    import sys
    import itertools
    
    # Method 1: Directly import and reset counter in module
    from oasis.social_agent import agents_generator
    agents_generator.id_gen = itertools.count(1)
    print("Agent ID counter reset to 1 (method 1)")
    
    # Method 2: If module is already in sys.modules, reset it too
    if 'oasis.social_agent.agents_generator' in sys.modules:
        module = sys.modules['oasis.social_agent.agents_generator']
        module.id_gen = itertools.count(1)
        print("Agent ID counter reset to 1 (method 2)")


def get_agent_state(agent_id: int, role: str, round_num: int = -1, database_path: str = None) -> dict:
    """Get the state of an agent at a specific round or current state."""
    if database_path is None:
        database_path = os.environ.get('MARKET_DB_PATH', 'market_sim.db')
    
    if not os.path.exists(database_path):
        # If seller, return initial reputation
        if role == 'seller':
            return {'reputation_score': 0, 'total_profit': 0}
        # If buyer, return initial utility
        else:
            return {'cumulative_utility': 0, 'total_utility': 0}

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    state = {}
    try:
        if role == 'seller':
            # Get seller basic information
            cursor.execute(
                "SELECT reputation_score, profit_utility_score FROM user WHERE agent_id = ?",
                (agent_id,)
            )
            result = cursor.fetchone()
            state['reputation_score'] = result[0] if result else 0
            state['total_profit'] = result[1] if result else 0
            
            # Get sales information for this round
            if round_num > 0:
                # First get user_id from agent_id
                cursor.execute(
                    "SELECT user_id FROM user WHERE agent_id = ?",
                    (agent_id,)
                )
                user_result = cursor.fetchone()
                if user_result:
                    user_id = user_result[0]
                    cursor.execute(
                        "SELECT COUNT(*), SUM(seller_profit) FROM transactions WHERE seller_id = ? AND round_number = ?",
                        (user_id, round_num)
                    )
                    sales_result = cursor.fetchone()
                    state['round_sales'] = sales_result[0] if sales_result else 0
                    state['round_profit'] = sales_result[1] if sales_result and sales_result[1] is not None else 0
                else:
                    state['round_sales'] = 0
                    state['round_profit'] = 0
        
        elif role == 'buyer':
            # Get buyer basic information
            cursor.execute(
                "SELECT profit_utility_score FROM user WHERE agent_id = ?",
                (agent_id,)
            )
            result = cursor.fetchone()
            state['cumulative_utility'] = result[0] if result and result[0] is not None else 0
            state['total_utility'] = result[0] if result and result[0] is not None else 0
            
            # Get purchase information for this round
            if round_num > 0:
                # First get user_id from agent_id
                cursor.execute(
                    "SELECT user_id FROM user WHERE agent_id = ?",
                    (agent_id,)
                )
                user_result = cursor.fetchone()
                if user_result:
                    user_id = user_result[0]
                    cursor.execute(
                        "SELECT COUNT(*), SUM(buyer_utility) FROM transactions WHERE buyer_id = ? AND round_number = ?",
                        (user_id, round_num)
                    )
                    purchase_result = cursor.fetchone()
                    state['round_purchases'] = purchase_result[0] if purchase_result else 0
                    state['round_utility'] = purchase_result[1] if purchase_result and purchase_result[1] is not None else 0
                else:
                    state['round_purchases'] = 0
                    state['round_utility'] = 0
    
    except sqlite3.Error as e:
        print(f"Database query error (get_agent_state): {e}")
        # Return default values on error
        if role == 'seller':
            state = {'reputation_score': 0, 'total_profit': 0}
        else:
            state = {'cumulative_utility': 0, 'total_utility': 0}
    finally:
        conn.close()
        
    return state


def get_product_listings(database_path: str = None) -> str:
    """Get current product listings."""
    if database_path is None:
        database_path = os.environ.get('MARKET_DB_PATH', 'market_sim.db')
    
    if not os.path.exists(database_path):
        return "No products are currently on sale."
    
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    listings = "No products are currently on sale."
    try:
        cursor.execute("PRAGMA table_info(post)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'status' not in columns:
            return "Database schema is incorrect: 'post' table is missing the 'status' column."
        cursor.execute(
            """
            SELECT p.post_id, u.agent_id, p.advertised_quality, p.price, p.has_warrant, 
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


def get_seller_round_summary(seller_id: int, round_num: int, database_path: str = None) -> dict:
    """Get seller's product listing information and sales status for a specific round."""
    if database_path is None:
        database_path = os.environ.get('MARKET_DB_PATH', 'market_sim.db')
    
    summary = {"quality": "Did not list", "sold": "No", "price": 0}
    if not os.path.exists(database_path):
        return summary
        
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    try:
        # First get user_id from agent_id
        cursor.execute(
            "SELECT user_id FROM user WHERE agent_id = ?",
            (seller_id,)
        )
        user_result = cursor.fetchone()
        if not user_result:
            return summary
        user_id = user_result[0]
        
        # Query in post table by user_id and round_number
        cursor.execute(
            "SELECT advertised_quality, is_sold, price FROM post WHERE user_id = ? AND round_number = ? ORDER BY post_id DESC LIMIT 1",
            (user_id, round_num) # Query current round (round_num)
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


def initialize_market_roles(agent_graph: AgentGraph, database_path: str = None):
    """Set roles and initial states for all agents in the database."""
    if database_path is None:
        database_path = os.environ.get('MARKET_DB_PATH', 'market_sim.db')
    
    print("Initializing market roles in the database...")
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    try:
        # Get information for all agents
        agents_info = []
        actual_agent_ids = []
        for agent_id, agent in agent_graph.get_agents():
            role = agent.user_info.profile.get("other_info", {}).get("role")
            agents_info.append((agent_id, role))
            actual_agent_ids.append(agent_id)
        
        print(f"Found {len(agents_info)} agents to initialize")
        print(f"Actual agent IDs in graph: {sorted(actual_agent_ids)}")
        
        # Set seller roles (first NUM_SELLERS agents)
        for i in range(SimulationConfig.NUM_SELLERS):
            agent_id = i + 1
            cursor.execute(
                "UPDATE user SET role = ?, reputation_score = ?, profit_utility_score = ? WHERE agent_id = ?",
                ('seller', 0, 0.0, agent_id) 
            )
            print(f"Set agent {agent_id} as seller")
        
        # Set buyer roles (next NUM_BUYERS agents)
        for i in range(SimulationConfig.NUM_BUYERS):
            agent_id = SimulationConfig.NUM_SELLERS + i + 1
            cursor.execute(
                "UPDATE user SET role = ?, profit_utility_score = ? WHERE agent_id = ?",
                ('buyer', 0.0, agent_id)
            )
            print(f"Set agent {agent_id} as buyer")
        
        conn.commit()
        
        # Verify setup results
        cursor.execute("SELECT COUNT(*) FROM user WHERE role = 'seller'")
        seller_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM user WHERE role = 'buyer'")
        buyer_count = cursor.fetchone()[0]
        
        print(f"Market roles initialized successfully: {seller_count} sellers, {buyer_count} buyers")
    except sqlite3.Error as e:
        print(f"Database error (initialize_market_roles): {e}")
    finally:
        conn.close()


async def run_single_simulation(database_path: str):
    """Run single market simulation
    
    Args:
        database_path: Database file path
    """
    print("Starting market simulation initialization...")
    
    # Reset agent ID counter to ensure each run starts from 1
    reset_agent_id_counter()
    
    # Set environment variables
    os.environ['MARKET_DB_PATH'] = database_path
    
    # Clean up existing database
    if os.path.exists(database_path):
        os.remove(database_path)
    
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
        agents_num=SimulationConfig.NUM_SELLERS,
        sys_prompt=SELLER_GENERATION_SYS_PROMPT,
        user_prompt=SELLER_GENERATION_USER_PROMPT,
        market_type=SimulationConfig.MARKET_TYPE,
        role="seller",
        db_path=database_path,
    )
    
    # Generate buyer agents
    print("Generating buyer agents...")
    buyer_agent_graph = await generate_agent_from_LLM(
        agents_num=SimulationConfig.NUM_BUYERS,
        sys_prompt=BUYER_GENERATION_SYS_PROMPT,
        user_prompt=BUYER_GENERATION_USER_PROMPT,
        market_type=SimulationConfig.MARKET_TYPE,
        role="buyer",
        db_path=database_path,
    )
    
    # Add seller agents
    for agent_id, agent in seller_agent_graph.get_agents():
        agent.env.is_market_sim = True
        agent_graph.add_agent(agent)
    
    # Add buyer agents (reassign agent_id)
    for agent_id, agent in buyer_agent_graph.get_agents():
        agent.env.is_market_sim = True
        agent_graph.add_agent(agent)

    env = make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT, 
        database_path=database_path
    )
    await env.reset()

    print(f"Environment initialized. Database at '{database_path}'.")
    initialize_market_roles(agent_graph, database_path)
    sellers_history = {i+1: [] for i in range(SimulationConfig.NUM_SELLERS)}
    
    for round_num in range(1, SimulationConfig.SIMULATION_ROUNDS + 1):
        # Synchronize platform round counter
        env.platform.sandbox_clock.round_step = round_num
        print(f"\n{'='*20} Starting Round {round_num}/{SimulationConfig.SIMULATION_ROUNDS} {'='*20}")

        # Seller exit mechanism: Allow exit after EXIT_ROUND (soft flag, actual implementation can be on platform side)
        if round_num > SimulationConfig.EXIT_ROUND:
            print("Sellers may exit market. (soft flag)")
        
        # Seller actions
        print(f"\n--- [Round {round_num}] Seller Action Phase ---")
        seller_actions = {}
        
        # Set environment market phase to listing
        env.market_phase = "listing"
        
        for agent_id, agent in agent_graph.get_agents():
            if agent.user_info.profile.get("role") == 'seller':
                state = get_agent_state(agent_id, 'seller', database_path=database_path)

                # Hide complete history in initial window (show only aggregated/summary or empty)
                history_log = sellers_history.get(agent_id, [])
                if round_num in SimulationConfig.INITIAL_WINDOW_ROUNDS:
                    visible_history_string = "History hidden in initial window."
                else:
                    visible_history_string = format_seller_history(history_log)
                
                # Update environment state
                env.current_round = round_num
                
                # Update agent state attributes
                agent.reputation_score = state['reputation_score']
                agent.history_summary = visible_history_string
                
                # System prompt already set during SocialAgent instantiation (static parameters)
                # Prepare round prompt (dynamic parameters)
                round_prompt = SELLER_ROUND_PROMPT.format(
                    history_summary=visible_history_string
                )
                # Tools available for sellers in listing phase
                listing_tools = ['list_product']
                
                # Conditionally inject additional tools: expose exit/re-enter market actions when allowed
                if round_num >= SimulationConfig.EXIT_ROUND:
                    listing_tools.append('exit_market')
                if round_num == SimulationConfig.REENTRY_ALLOWED_ROUND:
                    listing_tools.append('reenter_market')

                seller_actions[agent] = LLMAction(
                    extra_action=listing_tools,
                    extra_prompt=round_prompt
                )
        
        if seller_actions:
            await env.step(seller_actions)
        print("All seller actions are complete.")

        print(f"\n--- [Round {round_num}] Buyer Action Phase 1: Purchase ---")
        buyer_actions = {}
        
        # Set environment market phase to purchase
        env.market_phase = "purchase"

        for agent_id, agent in agent_graph.get_agents():
             if agent.user_info.profile.get("role") == 'buyer':
                state = get_agent_state(agent_id, 'buyer', database_path=database_path)
                # Update agent state attributes
                agent.cumulative_utility = state['cumulative_utility']
                
                # System prompt already set during SocialAgent instantiation (static parameters)
                # Prepare round prompt (dynamic parameters)
                round_prompt = BUYER_ROUND_PROMPT.format()
                # Tools available for buyers in purchase phase: only allow purchase
                purchase_tools = ['purchase_product_id']
                buyer_actions[agent] = LLMAction(
                    extra_action=purchase_tools,
                    extra_prompt=round_prompt
                )
        
        purchase_results = []
        if buyer_actions:
            purchase_results = await env.step(buyer_actions)
        print("All purchase actions are attempted.")

        # Seller re-entry mechanism: After REENTRY_ALLOWED_ROUND, allow marked manipulators/low-reputation sellers to re-enter (interface left here, actual control can depend on platform layer)
        if round_num == SimulationConfig.REENTRY_ALLOWED_ROUND:
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
                rating_tools = ['rate_transaction'] if SimulationConfig.MARKET_TYPE == 'reputation_only' else ['rate_transaction', 'challenge_warrant']
                
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

        clear_market(database_path)
        
        # Print round statistics
        print_round_statistics(round_num, database_path)
        
        # Statistics and recording: update reputation history (apply lag display: in round r only show ratings up to r-REPUTATION_LAG)
        ratings_cutoff_round = max(0, round_num - SimulationConfig.REPUTATION_LAG)
        with sqlite3.connect(database_path) as conn:
            compute_and_update_reputation(conn, round_num, ratings_up_to_round=ratings_cutoff_round)

        # Update history records 
        for agent_id, agent in agent_graph.get_agents():
            if agent.user_info.profile.get("role") == 'seller':
                round_summary = get_seller_round_summary(agent_id, round_num, database_path)
                new_state = get_agent_state(agent_id, 'seller', round_num, database_path) 

                # Get actual profit for this round
                round_profit = new_state.get('round_profit', 0)
                total_profit = new_state.get('total_profit', 0)

                # Calculate reputation that will be shown in the NEXT round (include current round ratings)
                next_round_cutoff = max(0, round_num + 1 - SimulationConfig.REPUTATION_LAG)
                with sqlite3.connect(database_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT COUNT(t.rating) as cnt, COALESCE(SUM(t.rating), 0)
                        FROM transactions t
                        WHERE t.rating IS NOT NULL AND t.round_number <= ? AND t.seller_id = ?
                        """,
                        (next_round_cutoff, agent_id),
                    )
                    result = cursor.fetchone()
                    if result and result[0] > 0:
                        next_round_reputation = round(result[1] / result[0])
                    else:
                        next_round_reputation = 0

                sellers_history[agent_id].append({
                    "round": round_num,
                    "quality": round_summary["quality"], 
                    "sold": round_summary["sold"], 
                    "profit": round_profit,
                    "reputation": next_round_reputation,  # Use reputation that will be shown next round
                    "total_profit": total_profit
                })

        print(f"\n{'='*20} End of Round {round_num} {'='*20}")

    
    # Vulnerability/manipulation detection (run once after entire simulation)
    run_detection(SimulationConfig.SIMULATION_ROUNDS, database_path)
    print("Manipulation analysis completed.")
    
    await env.close()
    print_simulation_summary(database_path)
    print("\nSimulation finished")


if __name__ == "__main__":
    # Allow direct running of single simulation for testing
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "test_single_run.db"
    asyncio.run(run_single_simulation(db_path))