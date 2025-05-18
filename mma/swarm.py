import json
import asyncio

from functools import reduce
from typing import Self
from .agent import Agent, Message

class Swarm:
    agents: list[Agent]

    def __init__(self, agents: dict[str, Agent], answers: dict[str, str] = None):
        self.agents = agents
        if answers is None:
            answers = {k: None for k in agents.keys()}
        self.answers = answers

    async def step(self) -> Self:
        results = await asyncio.gather(*[agent.step() for agent in self.agents.values()])
        agents, messages = list(zip(*results))
        messages = reduce(lambda a, b: a + b, messages)
        agents = {agent.name: agent for agent in agents}

        futures = []
        for msg in messages:
            if msg.recipient == "all":
                for name in agents.keys():
                    msg_copy = Message(sender=msg.sender, recipient=name, content=msg.content)
                    futures += [agents[name].recv_msg(msg_copy)]
            elif msg.recipient == "admin":
                self.answers[msg.sender] = msg.content
            elif msg.recipient not in agents:
                print(f"message addressed to agent not in agents ({', '.join(agents.keys())}):\n\n{str(msg)}")
                error_msg = Message(sender="admin", recipient=msg.sender, content=f"error: message recipient '{msg.recipient}' not found, recipient must be in group {', '.join(agents.keys())}")
                futures += [agents[msg.sender].recv_msg(error_msg)]
            else:
                futures += [agents[msg.recipient].recv_msg(msg)]
        await asyncio.gather(*futures)
        return Swarm(agents, self.answers)
