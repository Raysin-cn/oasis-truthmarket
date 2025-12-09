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

import inspect
import json
import logging
import re
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelManager
from camel.prompts import TextPrompt
from camel.toolkits import FunctionTool
from camel.types import OpenAIBackendRole

from oasis.social_agent.agent_action import SocialAction
from oasis.social_agent.agent_environment import SocialEnvironment
from oasis.social_platform import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType

if TYPE_CHECKING:
    from oasis.social_agent import AgentGraph

if "sphinx" not in sys.modules:
    agent_log = logging.getLogger(name="social.agent")
    agent_log.setLevel("DEBUG")

    if not agent_log.handlers:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(
            f"./log/social.agent-{str(now)}.log")
        file_handler.setLevel("DEBUG")
        file_handler.setFormatter(
            logging.Formatter(
                "%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
        agent_log.addHandler(file_handler)

ALL_SOCIAL_ACTIONS = [action.value for action in ActionType]


class SocialAgent(ChatAgent):
    r"""Social Agent."""

    def __init__(self,
                 agent_id: int,
                 user_info: UserInfo,
                 user_info_template: TextPrompt | None = None,
                 channel: Channel | None = None,
                 model: Optional[Union[BaseModelBackend,
                                       List[BaseModelBackend],
                                       ModelManager]] = None,
                 agent_graph: "AgentGraph" = None,
                 available_actions: list[ActionType] = [],
                 tools: Optional[List[Union[FunctionTool, Callable]]] = [],
                 max_iteration: int = 1,
                 interview_record: bool = False,
                 db_path: str = ""):
        self.social_agent_id = agent_id
        self.user_info = user_info
        self.channel = channel or Channel()
        self.env = SocialEnvironment(SocialAction(agent_id, self.channel), db_path=db_path)
        
        # Agent state attributes
        self.initial_budget = 10  # Seller and Buyer initial budget in simulation
        self.reputation_score = 0    # Seller reputation
        self.cumulative_utility = 0  # Buyer cumulative utility
        self.history_summary = "This is the first round. You have no past performance data."
        if user_info_template is None:
            system_message_content = self.user_info.to_system_message()
        else:
            system_message_content = self.user_info.to_custom_system_message(
                user_info_template)
        system_message = BaseMessage.make_assistant_message(
            role_name="system",
            content=system_message_content,  # system prompt
        )


        all_possible_tools = self.env.action.get_openai_function_list()
        all_possible_actions = [tool.func.__name__ for tool in all_possible_tools]
        self.all_possible_actions_dict = {tool.func.__name__: tool for tool in all_possible_tools}

        for action in available_actions:
            action_name = action.value if isinstance(
                action, ActionType) else action
            if action_name not in all_possible_actions:
                agent_log.warning(
                    f"Action {action_name} is not supported. Supported "
                    f"actions are: {', '.join(all_possible_actions)}")
        self.action_tools = [
            tool for tool in all_possible_tools if tool.func.__name__ in [
                a.value if isinstance(a, ActionType) else a
                for a in available_actions
            ]
        ]
        all_tools = (tools or []) + (self.action_tools or [])
        super().__init__(system_message=system_message,
                         model=model,
                         scheduling_strategy='random_model',
                         tools=all_tools,
                         max_iteration=max_iteration
                         )
        self.interview_record = interview_record
        self.agent_graph = agent_graph
        self.test_prompt = (
            "\n"
            "Helen is a successful writer who usually writes popular western "
            "novels. Now, she has an idea for a new novel that could really "
            "make a big impact. If it works out, it could greatly "
            "improve her career. But if it fails, she will have spent "
            "a lot of time and effort for nothing.\n"
            "\n"
            "What do you think Helen should do?")
        self.communication_memory = []

    async def perform_market_action(self, extra_action: List[Union[FunctionTool, Callable]] = None, extra_prompt: str = None, current_round: int = 1, market_phase: str = "general"):
        """
        Execute market simulation actions, including environment observation and extra prompts.
        
        Args:
            extra_action: Additional tool list
            extra_prompt: Additional prompt information
            current_round: Current round number
            market_phase: Market phase ("listing", "purchase", "rating", "general")
        """
        role = self.user_info.profile.get("role")

        # Get corresponding environment observation based on market phase
        env_prompt = await self.env.to_text_prompt(agent=self, current_round=current_round, market_phase=market_phase)
        agent_log.info(
            f"Agent {self.social_agent_id} ({role}) observing environment in {market_phase} phase: "
            f"{env_prompt}")

        # Build user message content: environment observation + extra prompt
        if role == 'seller':
            base_content = (
                "Based on your system instructions, which include your "
                "history and current state, you must now execute your "
                "chosen action for this round."
            )
        elif role == 'buyer':
            base_content = (
                "You have observed the current state of the market. "
                "Based on your role, objectives, and the market rules "
                "outlined in your system instructions, please decide on the "
                "best action to take now."
            )
        else:
            base_content = ""
        
        # Combine environment observation and extra prompt
        user_msg_content = f"{base_content}\n\n{env_prompt}"
        if extra_prompt:
            user_msg_content += f"\n\n## Additional Information:\n{extra_prompt}"
        user_msg_content += (
            "\n## Notice:\n"
            """
            You can execute your action in either of the following ways:
            
            1. **Preferred method**: Use tool_call to directly call the available tools.
            2. **Alternative method**: If tool_call is not available, output your action as JSON:
            
            <THOUGHT>
            1. Analyze the current situation and requirements.
            2. Determine the best action to take.
            3. Explain your reasoning for the selected action and the specific parameters you will use.
            </THOUGHT>
            
            <ACTION>
            {
                "function_name": "name_of_the_function",
                "arguments": {
                    "param1": "value1",
                    "param2": "value2"
                }
            }
            </ACTION>
            
            **IMPORTANT**: The JSON in <ACTION> must be valid JSON format. Do NOT add any comments (// or /* */) inside the JSON. All explanations should be in the <THOUGHT> section, not in the JSON itself.
            """
        )

        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=user_msg_content
        )

        if extra_action:
            extra_tool = [self.all_possible_actions_dict[extra_action_name] 
                         for extra_action_name in extra_action 
                         if extra_action_name in self.all_possible_actions_dict]
            self.add_tools(extra_tool)
        
        action_reasoning = ""
        action_result = None
        
        try:
            response = await self.astep(user_msg)
            action_reasoning = response.msg.content
            
            # Method 1: Check for tool_calls (preferred method)
            if response.info and 'tool_calls' in response.info and response.info['tool_calls']:
                for tool_call in response.info['tool_calls']:
                    action_name = tool_call.tool_name
                    args = tool_call.args
                    if isinstance(tool_call.result, dict):
                        tool_call.result['agent_id'] = self.social_agent_id
                        tool_call.result['_action_name'] = action_name
                        tool_call.result['_action_args'] = args
                    action_result = tool_call.result
                    agent_log.info(f"Agent {self.social_agent_id} performed "
                                f"action: {action_name} with args: {args} "
                                f"and reasoning: {action_reasoning}")
            
            # Method 2: Fallback to JSON parsing if no tool_calls
            elif not action_result:
                action_json = self._extract_action_json(action_reasoning)
                if action_json:
                    function_name = self._clean_function_name(action_json.get("function_name", ""))
                    # Try cleaned name first, then original name as fallback
                    func = self.all_possible_actions_dict.get(function_name)
                    if not func:
                        original_name = action_json.get("function_name", "")
                        func = self.all_possible_actions_dict.get(original_name)
                        if func:
                            function_name = original_name
                    
                    if not func:
                        available = list(self.all_possible_actions_dict.keys())[:5]
                        raise ValueError(f"Unknown action: {function_name}. Available: {available}...")
                    
                    action_args = action_json.get("arguments", {})
                    action_result = await func.func(**action_args)
                    if isinstance(action_result, dict):
                        action_result['agent_id'] = self.social_agent_id
                        action_result['_action_name'] = function_name
                        action_result['_action_args'] = action_args
                    agent_log.info(f"Agent {self.social_agent_id} performed {function_name} "
                                  f"(via JSON) with args: {action_args} "
                                  f"and reasoning: {action_reasoning}")
                else:
                    raise ValueError(f"No action found in response. Reasoning: {action_reasoning}")
            
        except Exception as e:
            agent_log.error(f"Agent {self.social_agent_id} error: {e}")
            action_result = {"success": False, "error": str(e)}
        finally:
            if extra_action:
                self.remove_tools(extra_action)
        
        return action_result, action_reasoning

    async def perform_communication_action(self, extra_action: List[Union[FunctionTool, Callable]] = None, extra_prompt: str = None, current_round: int = 1, market_phase: str = "general"):
        """
        Execute communication actions, including environment observation and extra prompts.
        
        Args:
            extra_action: Additional tool list
            extra_prompt: Additional prompt information
            current_round: Current round number
            market_phase: Market phase ("communication", "general")
        """
        role = self.user_info.profile.get("role")

        # Get corresponding environment observation based on market phase
        env_prompt = await self.env.to_text_prompt(agent=self, current_round=current_round, market_phase=market_phase)
        agent_log.info(
            f"Agent {self.social_agent_id} ({role}) observing environment in {market_phase} phase: "
            f"{env_prompt}")
        
        # In communication phase, parse posts' useful_info and add FRAUD records to communication_memory
        if market_phase == "communication":
            posts_data = self.env.get_posts_communication_data()
            for post in posts_data:
                useful_info = post.get('useful_info', '')
                if useful_info:
                    try:
                        # Parse useful_info as JSON
                        info_dict = json.loads(useful_info)
                        outcome = info_dict.get('outcome', '')
                        # If outcome is "FRAUD", add to communication_memory for data analysis
                        if outcome == "FRAUD":
                            self.communication_memory.append({
                                "seller": info_dict.get('seller', ''),
                                "outcome": outcome,
                                "post_id": post.get('post_id'),
                                "user_id": post.get('user_id'),
                                "content": post.get('content', '')
                            })
                            agent_log.info(
                                f"Agent {self.social_agent_id} recorded FRAUD from post {post.get('post_id')} "
                                f"by user {post.get('user_id')} about seller {info_dict.get('seller', '')}"
                            )
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        # Skip invalid useful_info format
                        agent_log.debug(f"Agent {self.social_agent_id} skipped invalid useful_info: {e}")
                        continue
            
        # Combine environment observation and extra prompt
        user_msg_content = f"{env_prompt}"
        if extra_prompt:
            user_msg_content += f"\n\n## Additional Information:\n{extra_prompt}"
        user_msg_content += (
            "\n## Notice:\n"
            """
            You can execute your action in either of the following ways:
            
            1. **Preferred method**: Use tool_call to directly call the available tools.
            2. **Alternative method**: If tool_call is not available, output your action as JSON:
            
            <THOUGHT>
            1. Analyze the current situation and requirements.
            2. Determine the best action to take.
            3. Explain your reasoning for the selected action and the specific parameters you will use.
            </THOUGHT>
            
            <ACTION>
            {
                "function_name": "name_of_the_function",
                "arguments": {
                    "param1": "value1",
                    "param2": "value2"
                }
            }
            </ACTION>
            
            **IMPORTANT**: The JSON in <ACTION> must be valid JSON format. Do NOT add any comments (// or /* */) inside the JSON. All explanations should be in the <THOUGHT> section, not in the JSON itself.
            """
        )

        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=user_msg_content
        )

        if extra_action:
            extra_tool = [self.all_possible_actions_dict[extra_action_name] 
                         for extra_action_name in extra_action 
                         if extra_action_name in self.all_possible_actions_dict]
            self.add_tools(extra_tool)
        
        action_reasoning = ""
        action_result = None
        
        try:
            response = await self.astep(user_msg)
            action_reasoning = response.msg.content
            
            # Method 1: Check for tool_calls (preferred method)
            if response.info and 'tool_calls' in response.info and response.info['tool_calls']:
                for tool_call in response.info['tool_calls']:
                    action_name = tool_call.tool_name
                    args = tool_call.args
                    if isinstance(tool_call.result, dict):
                        tool_call.result['agent_id'] = self.social_agent_id
                        tool_call.result['_action_name'] = action_name
                        tool_call.result['_action_args'] = args
                    action_result = tool_call.result
                    agent_log.info(f"Agent {self.social_agent_id} performed "
                                f"action: {action_name} with args: {args} "
                                f"and reasoning: {action_reasoning}")
            
            # Method 2: Fallback to JSON parsing if no tool_calls
            elif not action_result:
                action_json = self._extract_action_json(action_reasoning)
                if action_json:
                    function_name = self._clean_function_name(action_json.get("function_name", ""))
                    # Try cleaned name first, then original name as fallback
                    func = self.all_possible_actions_dict.get(function_name)
                    if not func:
                        original_name = action_json.get("function_name", "")
                        func = self.all_possible_actions_dict.get(original_name)
                        if func:
                            function_name = original_name
                    
                    if not func:
                        available = list(self.all_possible_actions_dict.keys())[:5]
                        raise ValueError(f"Unknown action: {function_name}. Available: {available}...")
                    
                    action_args = action_json.get("arguments", {})
                    action_result = await func.func(**action_args)
                    if isinstance(action_result, dict):
                        action_result['agent_id'] = self.social_agent_id
                        action_result['_action_name'] = function_name
                        action_result['_action_args'] = action_args
                    agent_log.info(f"Agent {self.social_agent_id} performed {function_name} "
                                  f"(via JSON) with args: {action_args} "
                                  f"and reasoning: {action_reasoning}")
                else:
                    raise ValueError(f"No action found in response. Reasoning: {action_reasoning}")
            
        except Exception as e:
            agent_log.error(f"Agent {self.social_agent_id} error: {e}")
            action_result = {"success": False, "error": str(e)}
        finally:
            if extra_action:
                self.remove_tools(extra_action)
        
        # Only record useful_info when agent creates a post with valid useful_info
        if action_result and isinstance(action_result, dict):
            action_name = action_result.get('_action_name', '')
            if action_name == 'create_post' and 'useful_info' in action_result:
                useful_info = action_result.get('useful_info', '')
                if useful_info:
                    try:
                        # Parse useful_info as JSON to validate format
                        info_dict = json.loads(useful_info)
                        # Only add to communication_memory if it's a valid FRAUD or HONEST record
                        if info_dict.get('outcome') in ['FRAUD', 'HONEST']:
                            self.communication_memory.append({
                                "seller": info_dict.get('seller', ''),
                                "outcome": info_dict.get('outcome', ''),
                                "post_id": action_result.get('post_id'),
                                "created_by": self.social_agent_id
                            })
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # Skip invalid useful_info format
                        pass
        
        return action_result, action_reasoning





    async def perform_action_by_llm(self, extra_action: List[Union[FunctionTool, Callable]] = None):
        """
        Original perform_action_by_llm function, keeping original functionality unchanged.
        """
        role = self.user_info.profile.get("other_info", {}).get("role")

        # Get posts:
        env_prompt = await self.env.to_text_prompt()
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"Please perform social media actions after observing the "
                f"platform environments. Notice that don't limit your "
                f"actions for example to just like the posts. "
                f"Here is your social media environment: {env_prompt}"))
        if extra_action:
            self.add_tools(extra_action)
        
        try:
            response = await self.astep(user_msg)
            
            # Inject agent_id into return results
            if response.info and 'tool_calls' in response.info and response.info['tool_calls']:  
                for tool_call in response.info['tool_calls']:
                    action_name = tool_call.tool_name
                    args = tool_call.args
                    # Add agent_id to platform return result dictionary
                    if isinstance(tool_call.result, dict):
                        tool_call.result['agent_id'] = self.social_agent_id
                    agent_log.info(f"Agent {self.social_agent_id} performed "
                                f"action: {action_name} with args: {args}")
            else:
                agent_log.warning(f"Agent {self.social_agent_id} did not perform any action.")

        except Exception as e:
            agent_log.error(f"Agent {self.social_agent_id} error: {e}")
            response = e

        finally:
            if extra_action:
                extra_action_names = [tool.func.__name__ for tool in extra_action]
                self.remove_tools(extra_action_names)
                
            return response

    async def perform_test(self):
        """
        doing group polarization test for all agents.
        TODO: rewrite the function according to the ChatAgent.
        TODO: unify the test and interview function.
        """
        # user conduct test to agent
        _ = BaseMessage.make_user_message(role_name="User",
                                          content=("You are a twitter user."))
        # Test memory should not be writed to memory.
        # self.memory.write_record(MemoryRecord(user_msg,
        #                                       OpenAIBackendRole.USER))

        openai_messages, num_tokens = self.memory.get_context()

        openai_messages = ([{
            "role":
            self.system_message.role_name,
            "content":
            self.system_message.content.split("# RESPONSE FORMAT")[0],
        }] + openai_messages + [{
            "role": "user",
            "content": self.test_prompt
        }])

        agent_log.info(f"Agent {self.social_agent_id}: {openai_messages}")
        # NOTE: this is a temporary solution.
        # Camel can not stop updating the agents' memory after stop and astep
        # now.
        response = await self._aget_model_response(
            openai_messages=openai_messages, num_tokens=num_tokens)
        content = response.output_messages[0].content
        agent_log.info(
            f"Agent {self.social_agent_id} receive response: {content}")
        return {
            "user_id": self.social_agent_id,
            "prompt": openai_messages,
            "content": content
        }

    async def perform_interview(self, interview_prompt: str):
        """
        Perform an interview with the agent.
        """
        # user conduct test to agent
        user_msg = BaseMessage.make_user_message(
            role_name="User", content=("You are a twitter user."))

        if self.interview_record:
            # Test memory should not be writed to memory.
            self.update_memory(message=user_msg, role=OpenAIBackendRole.SYSTEM)

        openai_messages, num_tokens = self.memory.get_context()

        openai_messages = ([{
            "role":
            self.system_message.role_name,
            "content":
            self.system_message.content.split("# RESPONSE FORMAT")[0],
        }] + openai_messages + [{
            "role": "user",
            "content": interview_prompt
        }])

        agent_log.info(f"Agent {self.social_agent_id}: {openai_messages}")
        # NOTE: this is a temporary solution.
        # Camel can not stop updating the agents' memory after stop and astep
        # now.

        response = await self._aget_model_response(
            openai_messages=openai_messages, num_tokens=num_tokens)

        content = response.output_messages[0].content

        if self.interview_record:
            # Test memory should not be writed to memory.
            self.update_memory(message=response.output_messages[0],
                               role=OpenAIBackendRole.USER)
        agent_log.info(
            f"Agent {self.social_agent_id} receive response: {content}")

        # Record the complete interview (prompt + response) through the channel
        interview_data = {"prompt": interview_prompt, "response": content}
        result = await self.env.action.perform_action(
            interview_data, ActionType.INTERVIEW.value)

        # Return the combined result
        return {
            "user_id": self.social_agent_id,
            "prompt": openai_messages,
            "content": content,
            "success": result.get("success", False)
        }

    async def perform_action_by_hci(self) -> Any:
        print("Please choose one function to perform:")
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            agent_log.info(f"Agent {self.social_agent_id} function: "
                           f"{function_list[i].func.__name__}")

        selection = int(input("Enter your choice: "))
        if not 0 <= selection < len(function_list):
            agent_log.error(f"Agent {self.social_agent_id} invalid input.")
            return
        func = function_list[selection].func

        params = inspect.signature(func).parameters
        args = []
        for param in params.values():
            while True:
                try:
                    value = input(f"Enter value for {param.name}: ")
                    args.append(value)
                    break
                except ValueError:
                    agent_log.error("Invalid input, please enter an integer.")

        result = await func(*args)
        return result

    def _clean_function_name(self, function_name: str) -> str:
        """Clean function name by removing meaningless prefixes."""
        if not function_name:
            return function_name
        # Remove common prefixes like "function.", "func.", "tool.", "action."
        prefixes = ["function.", "func.", "tool.", "action.", "functions.", "tools."]
        for prefix in prefixes:
            if function_name.startswith(prefix):
                function_name = function_name[len(prefix):]
        return function_name.strip()

    def _extract_action_json(self, content: str) -> dict | None:
        """Extract JSON action from LLM response, handling nested structures."""
        # Try extracting from tags/code blocks first
        for pattern in [r'<ACTION>(.*?)</ACTION>', r'```(?:json)?\s*\n?(.*?)```']:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = match.group(1).strip()
                if result := self._parse_json_string(json_str):
                    return result
        
        # Find function_name and extract complete JSON using balanced braces
        func_pos = content.find('"function_name"')
        if func_pos == -1:
            func_pos = content.find("'function_name'")
        if func_pos == -1:
            agent_log.warning(f"No function_name found. Content: {content[:200]}")
            return None
        
        brace_start = content.rfind('{', 0, func_pos)
        if brace_start == -1:
            return None
        
        # Extract JSON with balanced brace matching (handles nested structures)
        brace_count = in_string = escape_next = 0
        for i in range(brace_start, len(content)):
            char = content[i]
            if escape_next:
                escape_next = False
            elif char == '\\':
                escape_next = True
            elif char == '"' and not escape_next:
                in_string = not in_string
            elif not in_string:
                brace_count += (char == '{') - (char == '}')
                if brace_count == 0:
                    if result := self._parse_json_string(content[brace_start:i+1]):
                        return result
                    break
        
        agent_log.warning(f"Failed to parse action JSON. Content: {content[:200]}")
        return None
    
    def _parse_json_string(self, json_str: str) -> dict | None:
        """Parse JSON with automatic error recovery."""
        if not (json_str := json_str.strip()):
            return None
        
        # Try direct parse, then with fixes
        for s in [json_str, json_str.replace("'", '"')]:
            try:
                # Fix Python bools/None
                fixed = re.sub(r'\bTrue\b', 'true', re.sub(r'\bFalse\b', 'false', re.sub(r'\bNone\b', 'null', s)))
                data = json.loads(fixed)
                if isinstance(data, dict) and "function_name" in data:
                    return {
                        "function_name": self._clean_function_name(data["function_name"]),
                        "arguments": data.get("arguments", {})
                    }
            except (json.JSONDecodeError, KeyError):
                continue
        return None

    async def perform_action_by_data(self, func_name, *args, **kwargs) -> Any:
        func_name = func_name.value if isinstance(func_name,
                                                  ActionType) else func_name
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            if function_list[i].func.__name__ == func_name:
                func = function_list[i].func
                result = await func(*args, **kwargs)
                self.update_memory(message=BaseMessage.make_user_message(
                    role_name=OpenAIBackendRole.SYSTEM,
                    content=f"Agent {self.social_agent_id} performed "
                    f"{func_name} with args: {args} and kwargs: {kwargs}"
                    f"and the result is {result}"),
                                   role=OpenAIBackendRole.SYSTEM)
                agent_log.info(f"Agent {self.social_agent_id}: {result}")
                return result
        raise ValueError(f"Function {func_name} not found in the list.")

    def perform_agent_graph_action(
        self,
        action_name: str,
        arguments: dict[str, Any],
    ):
        r"""Remove edge if action is unfollow or add edge
        if action is follow to the agent graph.
        """
        if "unfollow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.remove_edge(self.social_agent_id, followee_id)
            agent_log.info(
                f"Agent {self.social_agent_id} unfollowed Agent {followee_id}")
        elif "follow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.add_edge(self.social_agent_id, followee_id)
            agent_log.info(
                f"Agent {self.social_agent_id} followed Agent {followee_id}")

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(agent_id={self.social_agent_id}, "
                f"model_type={self.model_type.value})")
