import json
import asyncio

from functools import reduce
from typing import Self
from .agent import Agent, Message

class Swarm:
    agents: dict[str, Agent]

    def __init__(self, agents: dict[str, Agent], answers: dict[str, str] = None, active: dict[str, bool] = None):
        self.agents = agents

        for name, agent in self.agents.items():
            if not agent.registered:
                self.agents[name].add_agent_registry(lambda: self.registry())

        if answers is None:
            answers = {k: None for k in agents.keys()}
        if active is None:
            active = {k: True for k in agents.keys()}

        self.active = active
        self.answers = answers
    
    def is_active(self) -> Self:
        return any([a for a in self.active.values()])

    def registry(self) -> str:
        return '\n'.join([f'- {name}' for name in self.agents.keys()])

    async def step(self) -> Self:
        active_agents = {a: self.agents[a] for a, active in self.active.items() if active}

        results = await asyncio.gather(*[agent.step() for agent in active_agents.values()])
        agents, messages = list(zip(*results))
        agents, messages = list(agents), list(messages)

        agents += [agent for a, agent in self.agents.items() if not self.active[a]]
        messages = reduce(lambda a, b: a + b, messages)

        agents = {agent.name: agent for agent in agents}
        # print(list(agents.keys()))

        def send_msg(recp, msg):
            if not self.active[recp]:
                self.active[recp] = True
            return agents[recp].send_msg(msg)

        futures = []
        for msg in messages:
            if msg.recipient == "all":
                for name in agents.keys():
                    futures += [send_msg(name, msg)]
            elif msg.recipient == "admin":
                self.answers[msg.sender] = msg.content
                self.active[msg.sender] = False
            elif msg.recipient not in agents:
                print(f"message addressed to agent not in agents ({', '.join(agents.keys())}):\n\n{str(msg)}")
                error_msg = Message(sender="admin", recipient=msg.sender, content=f"error: message recipient '{msg.recipient}' not found, recipient must be in group {', '.join(agents.keys())}")
                futures += [agents[msg.sender].send_msg(error_msg)]
            else:
                futures += [send_msg(msg.recipient, msg)]
        await asyncio.gather(*futures)
        return Swarm(agents, self.answers, self.active)
