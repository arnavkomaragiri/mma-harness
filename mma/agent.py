import asyncio

from pydantic import BaseModel
from typing import Union, Self, Callable

Chat = dict[str, Union[int, float, str]]

class Message(BaseModel):
    sender: str
    recipient: str
    content: str

    def __str__(self):
        return f"From: {self.sender}\nTo: {self.recipient}\nContent:\n{self.content}"

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
        await self.queue.put(msg)
        return self

    def extract_tool_msgs(self, resp: Chat) -> list[Message]:
        # TOOD: implement message parsing
        return Message(sender=self.name, recipient="placeholder", content="placeholder")
    
    async def step(self) -> tuple[Self, list[Message]]:
        msgs: list[Message] = []
        while not self.queue.empty():
            msgs += [await self.queue.get()]
        
        comb_msgs = self.aggregator(msgs) 
        chat = self.formatter(comb_msgs)

        self.chats += [chat]
        resp = self.llm(self.chats)
        tool_resp = self.tool_processor(resp)
        self.chats += [resp]

        messages = []
        if tool_resp is not None:
            self.chats += [tool_resp]
            messages = self.extract_tool_msgs(tool_resp)
        return self, messages

    @staticmethod
    def get_default_funcs(
        aggregator: Callable[[list[Message]], list[Message]] = None, 
        formatter: Callable[[list[Message]], Chat] = None,
        tool_processor: Callable[[Chat], Chat] = None,
    ) -> tuple[Callable, Callable, Callable]:
        if aggregator is None: aggregator = identity_aggregator
        if formatter is None: formatter = user_formatter
        if tool_processor is None: tool_processor = no_tools

        return aggregator, formatter, tool_processor

    @staticmethod
    def from_vllm(
        name: str,
        model_name: str, 
        aggregator: Callable[[list[Message]], list[Message]] = None, 
        formatter: Callable[[list[Message]], Chat] = None,
        tool_processor: Callable[[Chat], Chat] = None,
        **init_kwargs
    ) -> Self:
        from vllm import LLM

        aggregator, formatter, tool_processor = Agent.get_default_funcs(aggregator, formatter, tool_processor)
        
        engine = LLM(model_name, **init_kwargs)
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

    @staticmethod
    def from_vllm_server(
        name: str,
        model_name: str,
        url: str,
        api_key: str = "EMPTY",
        aggregator: Callable[[list[Message]], list[Message]] = None, 
        formatter: Callable[[list[Message]], Chat] = None,
        tool_processor: Callable[[Chat], Chat] = None,
        **kwargs
    ) -> Self:
        from openai import OpenAI

        aggregator, formatter, tool_processor = Agent.get_default_funcs(aggregator, formatter, tool_processor)

        client = OpenAI(
            api_key=api_key,
            base_url=url
        )
        def llm_func(chats: list[Chat]) -> Chat:
            # TODO: fix this monstrosity
            return client.chat.completions.create(
                model=model_name,
                messages=chats,
                **kwargs
            ).choices[0].message.dict()

        return Agent(name, llm_func, aggregator, formatter, tool_processor)
       