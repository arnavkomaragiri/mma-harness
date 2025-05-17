import asyncio

from vllm import LLM

from pydantic import BaseModel
from typing import Union, Self, Callable

Chat = dict[str, Union[int, float, str]]

class Message(BaseModel):
    sender: str
    recipient: str
    content: str

    def __str__(self):
        f"From: {self.sender}\nTo: {self.recipient}\nContent:\n{self.content}"

def identity_aggregator(msgs: list[Message]) -> list[Message]:
    return msgs

def user_formatter(msgs: list[Message]) -> Chat:
    return {
        "role": "user",
        "content": "You have received the following messages:\n" + "\n\n".join([str(msg) for msg in msgs])
    }

def no_tools(chat: Chat):
    return None

class Agent:
    name: str
    chats: list[Chat] 

    llm: Callable[[list[Chat]], Chat]
    aggregator: Callable[[list[Message]], list[Message]]
    formatter: Callable[[list[Message]], str]

    queue: asyncio.Queue

    def __init__(
            self, 
            name: str, 
            llm: Callable[[list[Chat]], Chat],
            aggregator: Callable[[list[Message]], list[Message]],
            formatter: Callable[[list[Message]], Chat],
            tool_processor: Callable[[Chat], Chat],
            chats: list[dict[str, Union[int, str]]] = [],
            max_msgs: int = 0,
        ):
        self.name = name
        self.llm = llm
        self.aggregator = aggregator
        self.formatter = formatter
        self.tool_processor = tool_processor
        self.chats = chats
        self.queue = asyncio.Queue(max_msgs)

    async def recv_msg(self, msg: Message) -> Self:
        self.queue.put(msg)
        return self
    
    async def step(self) -> Self:
        msgs: list[Message] = []
        while not self.queue.empty():
            msgs += [await self.queue.get()]
        
        comb_msgs = self.aggregator(msgs) 
        chat = self.formatter(comb_msgs)

        self.chats += [chat]
        resp = self.llm(self.chats)
        tool_resp = self.tool_processor(resp)
        self.chat += [resp]
        if tool_resp is not None:
            self.chat += [tool_resp]
        return self

    @staticmethod
    def from_vllm(
            name: str,
            model_name: str, 
            aggregator: Callable[[list[Message]], list[Message]] = None, 
            formatter: Callable[[list[Message]], Chat] = None,
            tool_processor: Callable[[Chat], Chat] = None,
            **kwargs
        ) -> Self:
        if aggregator is None: aggregator = identity_aggregator
        if formatter is None: formatter = user_formatter
        if tool_processor is None: tool_processor = no_tools

        engine = LLM(model_name, **kwargs)
        def llm_func(chats: list[Chat], **kwargs) -> Chat:
            resp = engine.chat(
                chats,
                **kwargs
            )
            return {
                "role": "assistant",
                "content": resp.outputs[0].text
            }
        
        return Agent(name, llm_func, aggregator, formatter, tool_processor)
