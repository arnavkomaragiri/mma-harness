import asyncio

from typing import Self
from .agent import Agent, Message

class Swarm:
    agents: list[Agent]

    def __init__(self, agents: dict[str, Agent]):
        self.agents = agents

    async def step(self) -> Self:
        results = await asyncio.gather([agent.step() for agent in self.agents.values()])
        agents, messages = list(zip(*results))
        agents = {agent.name: agent for agent in agents}

        futures = []
        for msg in messages:
            if msg.recipient not in agents:
                print(f"message addressed to agent not in agents ({', '.join([a.name for a in agents])}):\n\n{str(msg)}")
            futures += agents[msg.recipient].recv_msg(msg)
        await asyncio.gather(futures)
        return Swarm(agents)
