import json
import asyncio

from mma import Agent, Message, Swarm

template = """
You are {name}, and you are in a group of workers trying to solve a problem. Here are your teammates:
{teammates}
Work with your teammates to answer this question: {prompt}
""".strip()


async def main(num_iters: int):
    agent_names = ["Assistant A", "Assistant B", "Assistant C"]

    template_args = {
        "prompt": "Write a python program to launch a vllm server, send a query to it, and destroy it. Write each part independently (one component per assistant), then put them all together and submit the combined program"
    }
    teammate_strs = [f"- {name}" for name in agent_names]

    agents = {}
    for i, name in enumerate(agent_names):
        teammates = teammate_strs[:i] + teammate_strs[i + 1 :]
        start_msg = template.format(
            name=name, teammates="\n".join(teammates), **template_args
        )
        chats = [{"role": "user", "content": start_msg}]
        agents[name] = Agent.from_vllm_server(
            name=name,
            model_name="Qwen/Qwen3-4B-FP8",
            url="http://localhost:8000/v1",
            chats=chats,
        )
    swarm = Swarm(agents)

    # promises = []
    # for i, name in enumerate(agent_names):

    #     start_msg = template.format(name=name, **template_args)
    #     message = Message(sender="supervisor", recipient=name, content=start_msg)
    #     promises += [swarm.agents[name].send_msg(message)]
    # await asyncio.gather(*promises)

    # start main loop
    for _ in range(num_iters):
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
