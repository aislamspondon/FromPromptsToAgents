"""
Multi-Agent System: Recipe Agent + Robotics Agent with A2A Communication.
=========================================================================
Session 5: The Challenge - Robotic Chef Platform

This module implements the Agent-to-Agent (A2A) pipeline:
1. Food Analysis Agent receives a dish name
2. It calls the Recipe MCP Server to analyse the dish
3. It creates a structured task specification for the Robotics Agent
4. The Robotics Agent uses the Robotics MCP Server to design a robot
5. The final robot specification is returned

The two agents communicate via a structured task specification - the output
of Agent 1 becomes the input to Agent 2. This is the core A2A pattern.

All LLM calls go through llm_client (local LLM service via requests).
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import llm_client

# Directory containing the MCP server scripts
SERVER_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Core: Run an agent loop with an MCP server
# ---------------------------------------------------------------------------

async def run_agent_with_mcp(
    server_script: str,
    system_prompt: str,
    user_message: str,
    status_callback=None,
) -> str:
    """
    Generic function to run an LLM agent loop connected to an MCP server.

    The agent will:
    1. Connect to the specified MCP server via stdio
    2. Discover available tools and convert them to a simple format
    3. Send the user message to the LLM with the tool definitions
    4. Execute any tool calls the LLM requests via the MCP session
    5. Feed tool results back to the LLM
    6. Repeat until the LLM produces a final text response (max 10 iterations)

    Args:
        server_script: Absolute or relative path to the MCP server Python file.
        system_prompt: The system prompt defining the agent's role and behaviour.
        user_message: The user's input message to the agent.
        status_callback: Optional callable(str) for real-time status updates.
                         Used by the Streamlit UI to show progress.

    Returns:
        The agent's final text response.
    """

    def _status(msg: str):
        if status_callback:
            status_callback(msg)

    _status(f"Starting MCP server: {Path(server_script).name}")

    # Resolve the server script path
    server_path = str(Path(server_script).resolve())

    # Set up MCP server connection via stdio
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_path],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialise the MCP session
            await session.initialize()
            _status("MCP session initialised")

            # Discover tools and convert to simple format for llm_client
            tools_result = await session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema
                    if t.inputSchema
                    else {"type": "object", "properties": {}},
                }
                for t in tools_result.tools
            ]
            _status(
                f"Discovered {len(tools)} tools: "
                f"{', '.join(t['name'] for t in tools)}"
            )

            # Build initial conversation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            # Agent loop (max 10 iterations to prevent runaway)
            last_content = ""
            for iteration in range(10):
                _status(f"LLM call (iteration {iteration + 1})")

                response = llm_client.chat(messages, tools=tools)

                # If the LLM made tool calls, execute them
                if response["tool_calls"]:
                    # Append the assistant message (with tool calls) to conversation
                    messages.append(
                        {"role": "assistant", "content": response["raw"]}
                    )

                    for tc in response["tool_calls"]:
                        fn_name = tc["name"]
                        fn_args = tc["arguments"]

                        _status(f"Calling tool: {fn_name}")

                        # Execute the tool via MCP
                        try:
                            result = await session.call_tool(fn_name, fn_args)
                            # Extract text content from the MCP result
                            tool_output = ""
                            if result.content:
                                for content_block in result.content:
                                    if hasattr(content_block, "text"):
                                        tool_output += content_block.text
                            _status(
                                f"Tool {fn_name} returned {len(tool_output)} chars"
                            )
                        except Exception as e:
                            tool_output = json.dumps({"error": str(e)})
                            _status(f"Tool {fn_name} error: {e}")

                        # Add tool result to conversation
                        messages.append(
                            {
                                "role": "tool",
                                "name": fn_name,
                                "content": tool_output,
                            }
                        )
                else:
                    # No tool calls -- the agent produced a final text response
                    _status("Agent produced final response")
                    return response["content"] or ""

                last_content = response.get("content") or ""

            # If we exhausted iterations, return whatever we have
            _status("Max iterations reached")
            return (
                last_content
                or "Agent did not produce a final response within the iteration limit."
            )


# ---------------------------------------------------------------------------
# Agent 1: Food Analysis Agent
# ---------------------------------------------------------------------------

FOOD_ANALYSIS_SYSTEM_PROMPT = """\
You are the Food Analysis Agent, an expert culinary analyst. Your role is to \
thoroughly analyse a dish and produce a detailed, structured task specification \
that a Robotics Design Agent can use to design an automated cooking robot.

When given a dish name, use the available tools to:
1. Analyse the dish fully (ingredients, steps, techniques)
2. Get the detailed cooking techniques
3. Get equipment specifications for key equipment
4. Get safety requirements

Then synthesise all this information into a comprehensive TASK SPECIFICATION \
with the following clearly labelled sections:

## Dish Overview
- Name, cuisine, difficulty, servings, total time

## Physical Tasks Required
For each step, describe the physical action needed:
- Cutting/chopping (specify precision, force, dimensions)
- Stirring/mixing (specify speed, duration, force)
- Pouring/dispensing (specify volume, precision, temperature)
- Heating/temperature control (specify temperatures, durations, precision)
- Timing coordination (specify concurrent operations)

## Cooking Techniques with Precision Requirements
List each technique with:
- Temperature requirements (exact values in C)
- Duration requirements
- Precision level (critical/high/medium)
- Failure modes if done incorrectly

## Equipment to Operate
For each piece of equipment:
- What it is and how it must be operated
- Temperature ranges
- Physical interaction needed (knobs, handles, placement)

## Safety Requirements
- Temperature hazards and maximum temperatures
- Splash/splatter risks
- Timing-critical steps
- Food safety considerations

## Robotics Task Specification
A summary designed specifically for a Robotics Design Agent, listing:
- All manipulation tasks (with required degrees of freedom and force ranges)
- All sensing requirements (temperature, vision, force feedback)
- Workspace requirements (dimensions, stations)
- Speed and timing constraints
- Safety constraints for the robot

Be thorough and specific - the Robotics Agent depends entirely on your analysis \
to design an appropriate robot. Include exact temperatures, durations, and force \
estimates wherever possible.
"""


async def run_food_analysis_agent(dish_name: str, status_callback=None) -> str:
    """
    Run Agent 1: Food Analysis Agent.

    Connects to the Recipe MCP Server and thoroughly analyses the specified dish,
    producing a structured task specification for the Robotics Agent.

    Args:
        dish_name: Name of the dish to analyse (e.g. 'pasta carbonara').
        status_callback: Optional callable(str) for real-time status updates.

    Returns:
        A detailed task specification string.
    """
    server_script = str(SERVER_DIR / "recipe_mcp_server.py")
    user_message = (
        f"Please analyse the dish '{dish_name}' in complete detail. "
        f"Use all available tools to gather comprehensive information, then produce "
        f"a full task specification for the Robotics Design Agent."
    )

    return await run_agent_with_mcp(
        server_script=server_script,
        system_prompt=FOOD_ANALYSIS_SYSTEM_PROMPT,
        user_message=user_message,
        status_callback=status_callback,
    )


# ---------------------------------------------------------------------------
# Agent 2: Robotics Designer Agent
# ---------------------------------------------------------------------------

ROBOTICS_DESIGN_SYSTEM_PROMPT = """\
You are the Robotics Design Agent, an expert in designing robotic systems for \
food preparation and cooking tasks. You receive a detailed task specification \
from the Food Analysis Agent and must design a complete robotic cooking platform.

Use the available tools to:
1. Search for suitable robot arms/platforms based on the task requirements
2. Find appropriate sensors for the required sensing capabilities
3. Find actuators and end-effectors for the required manipulation tasks
4. Get detailed specifications for each selected component
5. Use the recommendation tool for an initial platform suggestion

Then design a complete robotic system with these clearly labelled sections:

## Robot Design Overview
- Robot type and form factor rationale
- Single-arm vs dual-arm justification
- Stationary vs mobile justification

## Selected Components
For each component, provide:
- Component ID and name
- Key specifications
- Why it was chosen for this specific dish

## Sensor Suite
For each sensor:
- Sensor ID and name
- What it monitors and why
- Mounting location recommendation

## Actuators and End-Effectors
For each actuator:
- Actuator ID and name
- What task it performs
- Key specifications relevant to the cooking task

## Motion and Control Requirements
- Degrees of freedom needed and why
- Speed requirements for time-critical operations
- Force control requirements
- Coordination between multiple operations

## Safety and Compliance
- How the robot handles high-temperature operations safely
- Human-robot interaction safety measures
- Food safety compliance
- Emergency stop scenarios

## Platform Summary Table
A clear summary table with all selected components, their IDs, and roles.

## Estimated Capabilities
- Which steps the robot can perform fully autonomously
- Which steps may need human oversight
- Overall autonomy percentage estimate

Be specific and reference actual component IDs from the database. Justify every \
selection based on the task specification you received.
"""


async def run_robotics_agent(task_specification: str, status_callback=None) -> str:
    """
    Run Agent 2: Robotics Designer Agent.

    Connects to the Robotics MCP Server and designs a complete robotic platform
    based on the task specification from the Food Analysis Agent.

    Args:
        task_specification: The detailed task specification from Agent 1.
        status_callback: Optional callable(str) for real-time status updates.

    Returns:
        A detailed robot design specification string.
    """
    server_script = str(SERVER_DIR / "robotics_mcp_server.py")
    user_message = (
        f"Based on the following task specification from the Food Analysis Agent, "
        f"design a complete robotic cooking platform. Search the component databases "
        f"thoroughly and select the best components for each requirement.\n\n"
        f"--- TASK SPECIFICATION ---\n{task_specification}\n--- END SPECIFICATION ---"
    )

    return await run_agent_with_mcp(
        server_script=server_script,
        system_prompt=ROBOTICS_DESIGN_SYSTEM_PROMPT,
        user_message=user_message,
        status_callback=status_callback,
    )


# ---------------------------------------------------------------------------
# Pipeline: Full Robotic Chef Pipeline (A2A)
# ---------------------------------------------------------------------------

async def run_robotic_chef_pipeline(
    dish_name: str, status_callback=None
) -> dict:
    """
    Run the full Robotic Chef A2A pipeline.

    This is the main entry point that orchestrates both agents:
    1. Runs the Food Analysis Agent to analyse the dish
    2. Passes the task specification to the Robotics Designer Agent
    3. Returns both outputs

    Args:
        dish_name: Name of the dish (e.g. 'pasta carbonara').
        status_callback: Optional callable(str) for real-time status updates.

    Returns:
        A dict with keys:
            - 'food_analysis': str - The Food Analysis Agent's output
            - 'robot_design': str - The Robotics Designer Agent's output
    """

    def _status(msg: str):
        if status_callback:
            status_callback(msg)

    # ---- Stage 1: Food Analysis Agent ----
    _status("=== Stage 1: Food Analysis Agent ===")
    food_analysis = await run_food_analysis_agent(
        dish_name=dish_name,
        status_callback=status_callback,
    )
    _status("Food Analysis Agent complete")

    # ---- Stage 2: Robotics Designer Agent ----
    _status("=== Stage 2: Robotics Designer Agent ===")
    robot_design = await run_robotics_agent(
        task_specification=food_analysis,
        status_callback=status_callback,
    )
    _status("Robotics Designer Agent complete")

    return {
        "food_analysis": food_analysis,
        "robot_design": robot_design,
    }


# ---------------------------------------------------------------------------
# CLI entry point (for testing without Streamlit)
# ---------------------------------------------------------------------------

async def _main():
    """Run the pipeline from the command line for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Robotic Chef Pipeline - CLI")
    parser.add_argument(
        "dish",
        nargs="?",
        default="pasta carbonara",
        help="Name of the dish to analyse (default: pasta carbonara)",
    )
    args = parser.parse_args()

    def print_status(msg: str):
        print(f"  [{msg}]")

    print(f"\nRobotic Chef Pipeline - Analysing: {args.dish}")
    print("=" * 60)

    result = await run_robotic_chef_pipeline(
        dish_name=args.dish,
        status_callback=print_status,
    )

    print("\n" + "=" * 60)
    print("FOOD ANALYSIS (Agent 1)")
    print("=" * 60)
    print(result["food_analysis"])

    print("\n" + "=" * 60)
    print("ROBOT DESIGN (Agent 2)")
    print("=" * 60)
    print(result["robot_design"])


if __name__ == "__main__":
    asyncio.run(_main())
