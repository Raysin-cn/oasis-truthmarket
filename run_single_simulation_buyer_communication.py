"""
Single market simulation run module
Core simulation logic extracted from run_market_simulation.py
"""

import asyncio
import sqlite3
import os
import oasis
import random
import json
from datetime import datetime
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


def get_action_log_path(database_path: str) -> str:
    """Generate action log JSON file path from database path"""
    return database_path.replace('.db', '_actions.json')


def cleanup_action_log(database_path: str):
    """Clean up action log JSON file if it exists"""
    log_path = get_action_log_path(database_path)
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"Action log cleaned up: {log_path}")


def save_action_records(env, round_num: int, phase: str, database_path: str):
    """Save env.step() detailed results to JSON file"""
    if not hasattr(env, '_last_step_detailed_results') or not env._last_step_detailed_results:
        return
    
    log_path = get_action_log_path(database_path)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    all_records = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    for result in env._last_step_detailed_results:
        all_records.append({
            'round': round_num,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'agent_id': result.get('agent_id'),
            'action_name': result.get('action_name'),
            'action_args': result.get('action_args'),
            'action_result': result.get('action_result'),
            'reasoning': result.get('reasoning', '')
        })
    
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False, default=str)
    except IOError as e:
        print(f"Warning: Failed to save action records to {log_path}: {e}")


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
    if role == 'seller':
        # Get seller basic information
        cursor.execute(
            "SELECT reputation_score, profit_utility_score FROM user WHERE agent_id = ?",
            (agent_id,)
        )
        result = cursor.fetchone()
        state['reputation_score'] = result[0] if result else 0
        state['total_profit'] = result[1] if result else 0
        
        # Get sales information for this round  #TODO maybe fetch from user table rather than transactions table
        if round_num > 0:
            user_id = agent_id
            cursor.execute(
                "SELECT COUNT(*), SUM(seller_profit) FROM transactions WHERE seller_id = ? AND round_number = ?",
                (user_id, round_num)
            )
            sales_result = cursor.fetchone()
            state['round_sales'] = sales_result[0] if sales_result else 0
            state['round_profit'] = sales_result[1] if sales_result and sales_result[1] is not None else 0
    
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
            user_id = agent_id
            cursor.execute(
                "SELECT COUNT(*), SUM(buyer_utility) FROM transactions WHERE buyer_id = ? AND round_number = ?",
                (user_id, round_num)
            )
            purchase_result = cursor.fetchone()
            state['round_purchases'] = purchase_result[0] if purchase_result else 0
            state['round_utility'] = purchase_result[1] if purchase_result and purchase_result[1] is not None else 0
    
    conn.close()
        
    return state



def get_seller_round_summary(seller_id: int, round_num: int, database_path: str = None) -> dict:
    """Get seller's product listing information and sales status for a specific round."""
    if database_path is None:
        database_path = os.environ.get('MARKET_DB_PATH', 'market_sim.db')
    
    # Initialize summary with default values outside try block
    summary = {"advertised_quality": None, "true_quality": None, "warrant": None, "is_sold": 0, "sold_numbers": 0, "cost": 0, "price": 0}
    
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    try:
        # First get user_id from agent_id
        cursor.execute(
            "SELECT user_id FROM user WHERE agent_id = ?",
            (seller_id,)
        )
        user_result = cursor.fetchone()
        user_id = user_result[0]
        
        # Query in product table by user_id and round_number
        cursor.execute(
            "SELECT advertised_quality, true_quality, has_warrant, is_sold, cost, price FROM product WHERE user_id = ? AND round_number = ? ORDER BY product_id",
            (user_id, round_num)
        )
        all_result = cursor.fetchall()
        if all_result:
            one_result = all_result[0]
            summary["advertised_quality"] = one_result[0]
            summary["true_quality"] = one_result[1]
            summary["warrant"] = one_result[2]
            summary["is_sold"] = one_result[3]
            summary["sold_numbers"] = sum(1 for p in all_result if p[3])
            summary["cost"] = one_result[4]
            summary["price"] = one_result[5]
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


async def run_single_simulation(database_path: str, market_type: str = None, conditions: str = None):
    """Run single market simulation
    
    Args:
        database_path: Database file path
    """
    print("Starting market simulation initialization...")
    
    # Reset agent ID counter to ensure each run starts from 1
    reset_agent_id_counter()
    
    # Set environment variables
    os.environ['MARKET_DB_PATH'] = database_path


    if market_type is None:
        market_type = SimulationConfig.MARKET_TYPE


    # Clean up existing database and action log
    if os.path.exists(database_path):
        os.remove(database_path)
    cleanup_action_log(database_path)

    model = ModelFactory.create(
        model_platform=SimulationConfig.MODEL_PLATFORM,
        model_type=SimulationConfig.MODEL_TYPE,
        api_key=os.getenv("MODEL_API_KEY"), 
        url=os.getenv("MODEL_BASE_URL"), 
    )

    agent_graph = AgentGraph()

    # Generate seller agents
    print("Generating seller agents...")
    seller_agent_graph = await generate_agent_from_LLM(
        agents_num=SimulationConfig.NUM_SELLERS,
        sys_prompt=SELLER_GENERATION_SYS_PROMPT,
        user_prompt=SELLER_GENERATION_USER_PROMPT,
        market_type=market_type,
        role="seller",
        model=model,
        db_path=database_path,
    )
    
    # Generate buyer agents
    print("Generating buyer agents...")
    buyer_agent_graph = await generate_agent_from_LLM(
        agents_num=SimulationConfig.NUM_BUYERS,
        sys_prompt=BUYER_GENERATION_SYS_PROMPT,
        user_prompt=BUYER_GENERATION_USER_PROMPT,
        market_type=market_type,
        role="buyer",
        model=model,
        db_path=database_path,
    )
    
    # Add seller agents
    for agent_id, agent in seller_agent_graph.get_agents():
        agent.env.is_market_sim = True
        agent_graph.add_agent(agent)
    
    # Add buyer agents (reassign agent_id)
    for agent_id, agent in buyer_agent_graph.get_agents():
        agent.env.is_market_sim = True
        if conditions != None:
            agent.env.conditions = conditions
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
                state = get_agent_state(agent_id, 'seller', round_num=round_num, database_path=database_path)

                # Hide complete history in initial window (show only aggregated/summary or empty)
                history_log = sellers_history.get(agent_id, [])
                visible_history_string = format_seller_history(history_log)
                
                # Update environment state
                env.current_round = round_num
                
                # Update agent state attributes
                agent.reputation_score = state['reputation_score']
                agent.history_summary = visible_history_string
                
                # System prompt already set during SocialAgent instantiation (static parameters)
                # Prepare round prompt (dynamic parameters)
                seller_round_prompt = SELLER_ROUND_PROMPT.format(
                    history_summary=visible_history_string
                )
                # Tools available for sellers in listing phase
                listing_tools = ['list_product']
                
                # Conditionally inject additional tools: expose exit/re-enter market actions when allowed
                if round_num >= SimulationConfig.EXIT_ROUND:
                    listing_tools.append('exit_market')
                    seller_round_prompt += "\n\nYou are now allowed to exit the market.\n"
                if round_num == SimulationConfig.REENTRY_ALLOWED_ROUND:
                    listing_tools.append('reenter_market')
                    seller_round_prompt += "\n\nYou are now allowed to re-enter the market.\n"

                seller_actions[agent] = LLMAction(
                    extra_action=listing_tools,
                    extra_prompt=seller_round_prompt,
                    level = "market"
                )
        
        if seller_actions:
            await env.step(seller_actions)
            save_action_records(env, round_num, 'seller_listing', database_path)
        print("All seller actions are complete.")

        # Buyer communication
        print(f"\n--- [Round {round_num}] Buyer Communication Phase ---")
        buyer_communication_actions = {}
        
        # Set environment market phase to communication
        env.market_phase = "communication"
        
        for agent_id, agent in agent_graph.get_agents():
            if agent.user_info.profile.get("role") == 'buyer':
                state = get_agent_state(agent_id, 'buyer', round_num=round_num, database_path=database_path)
                
                # Update agent state attributes
                agent.cumulative_utility = state['cumulative_utility']
                
                # Update environment state
                env.current_round = round_num

                # System prompt already set during SocialAgent instantiation (static parameters)
                # Prepare round prompt (dynamic parameters)
                buyer_communication_round_prompt = (
                    "\n\nIn this phase, you are allowed to perform some social platform actions to communicate with other buyers. "
                    "You cannot perform any other actions during this phase.\n"
                    "You can share your purchase experience, product information, seller reputation, or any other information with other buyers to help them make a purchase decision. "
                    "\n\nWhen you call `create_post(content, useful_info)`, the `useful_info` argument MUST be a JSON string that captures your private ground-truth log in this exact format: "
                    "{\"seller\": \"<seller_id>\", \"outcome\": \"<FRAUD or HONEST>\"}. "
                    "Set `outcome` to `\"FRAUD\"` when a seller advertised high quality but delivered low quality; set `outcome` to `\"HONEST\"` when a seller advertised high quality and delivered high quality. "
                    "If you are not sharing a specific transaction experience, pass an empty string for `useful_info`."
                )
                # Tools available for buyers in communication phase: only allow social platform actions
                communication_tools = ['create_post', 'quote_post', 'like_post', 'dislike_post']
                buyer_communication_actions[agent] = LLMAction(
                    extra_action=communication_tools,
                    extra_prompt=buyer_communication_round_prompt,
                    level = "communication"
                )
        if buyer_communication_actions:
            results = await env.step(buyer_communication_actions)
            save_action_records(env, round_num, 'buyer_communication', database_path)
        print("All buyer communication actions are complete.")

        print(f"\n--- [Round {round_num}] Buyer Action Phase 1: Purchase ---")

        
        # Set environment market phase to purchase
        env.market_phase = "purchase"

        purchase_results = []
        agent_lists = agent_graph.get_agents()
        # Shuffle and select buyers randomly
        buyers = [pair for pair in agent_lists if pair[1].user_info.profile.get("role") == "buyer"]
        random.shuffle(buyers)
        for agent_id, agent in buyers:
            buyer_actions = {}
            state = get_agent_state(agent_id, 'buyer', round_num=round_num, database_path=database_path)
            # Update agent state attributes
            agent.cumulative_utility = state['cumulative_utility']
            
            # System prompt already set during SocialAgent instantiation (static parameters)
            # Prepare round prompt (dynamic parameters)
            buyer_round_prompt = (
                "\n\nIn this phase, you are only allowed to perform the purchase_product_id action to purchase a product. "
                "Based on the market environment, product information, and your preferences, choose whether and which product to purchase. "
                "You cannot perform any other actions during this phase.\n"
            )
            # Tools available for buyers in purchase phase: only allow purchase
            purchase_tools = ['purchase_product_id']
            buyer_actions[agent] = LLMAction(
                extra_action=purchase_tools,
                extra_prompt=buyer_round_prompt,
                level="market"
            )
            results = await env.step(buyer_actions)
            save_action_records(env, round_num, 'buyer_purchase', database_path)
            # env.step returns a list, so extend instead of append
            if isinstance(results, list):
                purchase_results.extend(results)
            elif isinstance(results, dict):
                purchase_results.append(results)
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
                rating_tools = ['rate_transaction'] if market_type == 'reputation_only' else ['rate_transaction', 'challenge_warrant']
                
                # Store purchase information in agent for environment observation
                agent.last_purchase_info = {
                    'transaction_id': purchase_info.get("transaction_id"),
                    'product_id': purchase_info.get("product_id"),
                    'advertised_quality': purchase_info.get("advertised_quality"),
                    'true_quality': purchase_info.get("true_quality"),
                    'has_warrant': purchase_info.get("has_warrant"),
                    'buyer_utility': purchase_info.get("buyer_utility"),
                    'purchase_price': purchase_info.get("purchase_price", 0),
                    'seller_id': purchase_info.get("seller_id", 'N/A'),
                    'seller_reputation': purchase_info.get("seller_reputation", 0)
                }
                buyer_rating_prompt = (
                    "\n\nIn this phase, you are allowed to perform the rate_transaction action to rate a transaction. " + "Or perform the challenge_warrant action to challenge the warrant of a transaction." if market_type == 'reputation_only' else "Or perform the challenge_warrant action to challenge the warrant of a transaction."
                    "Based on the market environment, product information, and your preferences, choose whether and which product to rate. " + "Or challenge the warrant of a transaction." if market_type == 'reputation_only' else "Or challenge the warrant of a transaction."
                    "You cannot perform any other actions during this phase.\n"
                )
                post_purchase_actions[agent] = LLMAction(
                    extra_action=rating_tools,
                    extra_prompt=buyer_rating_prompt,  # Environment observation information will be provided through market_phase="rating"
                    level = "market"
                )
        
        if post_purchase_actions:
            await env.step(post_purchase_actions)
            save_action_records(env, round_num, 'buyer_rating', database_path)
        print("All post-purchase actions are complete.")

        # clear_market(database_path)
        
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
                new_state = get_agent_state(agent_id, 'seller', round_num=round_num, database_path=database_path) 

                # Get actual profit for this round
                round_profit = (round_summary.get('price', 0) - round_summary.get('cost', 0)) * round_summary.get('sold_numbers', 0)
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
                    if result:
                        next_round_reputation = result[1]
                    else:
                        next_round_reputation = 0

                sellers_history[agent_id].append({
                    "round": round_num,
                    "true_quality": round_summary["true_quality"], 
                    "advertised_quality": round_summary["advertised_quality"], 
                    "warrant": round_summary["warrant"], 
                    "is_sold": round_summary["is_sold"], 
                    "sold_numbers": round_summary["sold_numbers"], 
                    "cost": round_summary["cost"], 
                    "price": round_summary["price"], 
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
    db_path = sys.argv[1] if len(sys.argv) > 1 else "test_single_run_buyer_communication.db"
    market_type = sys.argv[2] if len(sys.argv) > 2 else None
    conditions = sys.argv[3] if len(sys.argv) > 3 else None
    asyncio.run(run_single_simulation(db_path, market_type=market_type, conditions=conditions))