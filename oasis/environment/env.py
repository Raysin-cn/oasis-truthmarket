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
import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Union

from oasis.environment.env_action import LLMAction, ManualAction
from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_agent.agents_generator import generate_custom_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import (ActionType, DefaultPlatformType,
                                          RecsysType)

# Create log directory if it doesn't exist
log_dir = "./log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logger
env_log = logging.getLogger("oasis.env")
env_log.setLevel("INFO")

# Add file handler to save logs to file
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_handler = logging.FileHandler(f"{log_dir}/oasis-{current_time}.log",
                                   encoding="utf-8")
file_handler.setLevel("INFO")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
env_log.addHandler(file_handler)


class OasisEnv:

    def __init__(
        self,
        agent_graph: AgentGraph,
        platform: Union[DefaultPlatformType, Platform],
        database_path: str = None,
        semaphore: int = 128,
    ) -> None:
        r"""Init the oasis environment.

        Args:
            agent_graph: The AgentGraph to use in the simulation.
            platform: The platform type to use. Including
                `DefaultPlatformType.TWITTER` or `DefaultPlatformType.REDDIT`.
                Or you can pass a custom `Platform` instance.
            database_path: The path to create a sqlite3 database. The file
                extension must be `.db` such as `twitter_simulation.db`.
        """
        # Initialize the agent graph
        self.agent_graph = agent_graph
        # Use a semaphore to limit the number of concurrent requests
        self.llm_semaphore = asyncio.Semaphore(semaphore)
        # Environment state
        self.current_round = 1
        self.market_phase = "general"  # Current market phase: listing, purchase, rating, general
        if isinstance(platform, DefaultPlatformType):
            if database_path is None:
                raise ValueError(
                    "database_path is required for DefaultPlatformType")
            self.platform = platform
            if platform == DefaultPlatformType.TWITTER:
                self.channel = Channel()
                self.platform = Platform(
                    db_path=database_path,
                    channel=self.channel,
                    recsys_type="twhin-bert",
                    refresh_rec_post_count=2,
                    max_rec_post_len=2,
                    following_post_count=3,
                )
                self.platform_type = DefaultPlatformType.TWITTER
            elif platform == DefaultPlatformType.REDDIT:
                self.channel = Channel()
                self.platform = Platform(
                    db_path=database_path,
                    channel=self.channel,
                    recsys_type="reddit",
                    allow_self_rating=True,
                    show_score=True,
                    max_rec_post_len=100,
                    refresh_rec_post_count=5,
                )
                self.platform_type = DefaultPlatformType.REDDIT
            else:
                raise ValueError(f"Invalid platform: {platform}. Only "
                                 "DefaultPlatformType.TWITTER or "
                                 "DefaultPlatformType.REDDIT are supported.")
        elif isinstance(platform, Platform):
            if database_path != platform.db_path:
                env_log.warning("database_path is not the same as the "
                                "platform.db_path, using the platform.db_path")
            self.platform = platform
            self.channel = platform.channel
            if platform.recsys_type == RecsysType.REDDIT:
                self.platform_type = DefaultPlatformType.REDDIT
            else:
                self.platform_type = DefaultPlatformType.TWITTER
        else:
            raise ValueError(
                f"Invalid platform: {platform}. You should pass a "
                "DefaultPlatformType or a Platform instance.")

    async def reset(self) -> None:
        r"""Start the platform and sign up the agents."""
        self.platform_task = asyncio.create_task(self.platform.running())
        self.agent_graph = await generate_custom_agents(
            channel=self.channel, agent_graph=self.agent_graph)

    async def _perform_llm_action(self, agent, llm_action):
        r"""Send the request to the llm model and execute the action.
        """
        async with self.llm_semaphore:
            extra_action = llm_action.extra_action if hasattr(llm_action, 'extra_action') else None
            extra_prompt = llm_action.extra_prompt if hasattr(llm_action, 'extra_prompt') else None
            return await agent.perform_market_action(extra_action, extra_prompt, self.current_round, self.market_phase)

    async def _perform_interview_action(self, agent, interview_prompt: str):
        r"""Send the request to the llm model and execute the interview.
        """
        async with self.llm_semaphore:
            return await agent.perform_interview(interview_prompt)

    async def step(
        self, actions: dict[SocialAgent, Union[ManualAction, LLMAction,
                                               List[Union[ManualAction,
                                                          LLMAction]]]]
    ) -> List[Dict[str, Any]]: 
        r"""Update the recommendation system and perform the actions."""
        await self.platform.update_rec_table()
        env_log.info("update rec table.")

        tasks = []
        for agent, action in actions.items():
            if isinstance(action, list):
                for single_action in action:
                    if isinstance(single_action, ManualAction):
                        if single_action.action_type == ActionType.INTERVIEW:
                            interview_prompt = single_action.action_args.get("prompt", "")
                            tasks.append(self._perform_interview_action(agent, interview_prompt))
                        else:
                            tasks.append(agent.perform_action_by_data(single_action.action_type, **single_action.action_args))
                    elif isinstance(single_action, LLMAction):
                        tasks.append(self._perform_llm_action(agent, single_action))
            else:
                if isinstance(action, ManualAction):
                    if action.action_type == ActionType.INTERVIEW:
                        interview_prompt = action.action_args.get("prompt", "")
                        tasks.append(self._perform_interview_action(agent, interview_prompt))
                    else:
                        tasks.append(agent.perform_action_by_data(action.action_type, **action.action_args))
                elif isinstance(action, LLMAction):
                    tasks.append(self._perform_llm_action(agent, action))

        # Execute all tasks and extract, return result list
        responses = await asyncio.gather(*tasks)
        results_reasoning = [response[1] for response in responses]
        responses= [response[0] for response in responses]
        results = []
        for response in responses:
            if response and hasattr(response, 'info') and response.info and 'tool_calls' in response.info and response.info['tool_calls']:
                # Extract result of each successful call
                for tool_call in response.info['tool_calls']:
                    if tool_call.result:
                        results.append(tool_call.result)

        env_log.info("performed all actions.")
        
        if self.platform_type == DefaultPlatformType.REDDIT:
            self.platform.sandbox_clock.time_step += 1
            
        return results

    async def close(self) -> None:
        r"""Stop the platform and close the environment.
        """
        await self.channel.write_to_receive_queue(
            (None, None, ActionType.EXIT))
        await self.platform_task
        env_log.info("Simulation finished! Please check the results in the "
                     f"database: {self.platform.db_path}. Note that the trace "
                     "table stored all the actions of the agents.")
