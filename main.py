import os
import json
import asyncio

from tqdm import tqdm, trange
from mma import Agent, Message, Swarm

template = """
You are {name}, and you are in a group of workers trying to solve a problem. Here are your teammates:
{teammates}
Make sure to use the tools provided to communicate with your teammates; solely chatting will not be sufficient.
""".strip()

leader_template = """
You are the leader of a group of workers, and you are trying to solve a problem. Here are your teammates:
{teammates}
Work with your teammates to answer this question: {prompt}
Make sure to use the tools provided to communicate with your teammates; solely chatting will not be sufficient.
""".strip()

USE_EXTERNAL = False


async def main(num_iters: int):
    agent_names = ["Assistant A", "Assistant B", "Assistant C", "Leader"]

    template_args = {
        "prompt": "Write a python program to launch a vllm server, send a query to it, and destroy it."
    }
    teammate_strs = [f"- {name}" for name in agent_names]

    agents = {}
    for i, name in enumerate(agent_names):
        extra_args = {}

        msg_template = template
        if name == "Leader":
            extra_args = template_args
            msg_template = leader_template

        start_msg = msg_template.format(
            name=name, teammates="\n".join(teammate_strs[:i] + teammate_strs[i+1:]), **extra_args
        )

        chats = [{"role": "user", "content": start_msg}]

        if USE_EXTERNAL:
            kwargs = {
                'model_name': "gemini-2.5-flash-preview-04-17",
                'url': "https://generativelanguage.googleapis.com/v1beta/openai/",
                'api_key': os.getenv('API_KEY'),
            }
        else:
            kwargs = {
                'model_name': "Qwen/Qwen3-4B-FP8",
                'url': 'http://localhost:8000/v1',
                'tool_choice': 'auto',
                'parallel_tool_calls': True
            }
        agents[name] = Agent.from_server(
            name=name,
            chats=chats,
            **kwargs
        )
    swarm = Swarm(agents)

    # start main loop
    for _ in trange(num_iters):
        if not swarm.is_active():
            break
        swarm = await swarm.step()

    print(swarm.answers)
    for agent in swarm.agents.values():
        print(f"{agent.name} {'=' * 150}")
        print(json.dumps(agent.chats, indent=2))
        print("=" * 160)
    return swarm


if __name__ == "__main__":
    swarm = asyncio.run(main(5))
