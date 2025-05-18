import json
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

def user_formatter(msgs: list[Message], template: str = "{content}") -> Chat:
    return {
        "role": "user",
        "content": template.format(content="You have received the following messages:\n" + "\n\n".join([str(msg) for msg in msgs]))
    }

def no_tools(chat: Chat):
    return None

def openai_tool_processor(chat: Chat):
    return chat['tool_calls']

DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send",
            "description": "Send a brief text message to someone else, such as a collaborator or teammate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "The recipient of your message (all possible recipients should be listed in system prompt or user message)."},
                    "message": {"type": "string", "description": "The message to send the recipient (e.g. questions, statements). Should be brief and avoid small talk."},
                },
                "required": ["recipient", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "broadcast",
            "description": "Send a brief text message to everyone at the same time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message to send everyone (e.g. questions, statements). Should be brief and avoid small talk."},
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit a final answer to a question after collaborating with teammates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The final answer to submit."},
                },
                "required": ["answer"]
            }
        }
    }
]

class Agent:
    # name: str
    # chats: list[Chat] 

    # llm: Callable[[list[Chat]], Chat]
    # aggregator: Callable[[list[Message]], list[Message]]
    # formatter: Callable[[list[Message], str], str]

    # queue: asyncio.Queue

    def __init__(
            self, 
            name: str, 
            llm: Callable[[list[Chat], list[dict]], Chat],
            aggregator: Callable[[list[Message]], list[Message]],
            formatter: Callable[[list[Message], str], Chat],
            tool_processor: Callable[[Chat], Chat],
            chats: list[dict[str, Union[int, str]]] = None,
            extra_tools: list[dict] = None,
            template: str = "{content}",
            max_msgs: int = 0
        ):
        if chats is None:
            chats = []

        if extra_tools is None:
            extra_tools = []

        self.name = name
        self.llm = llm
        self.aggregator = aggregator
        self.formatter = formatter
        self.tool_processor = tool_processor
        self.chats = chats
        self.template = template
        self.queue = asyncio.Queue(max_msgs)

        self.tools = DEFAULT_TOOLS + extra_tools

    async def recv_msg(self, msg: Message) -> Self:
        await self.queue.put(msg)
        return self

    def extract_tool_msgs(self, tool_calls: list[dict]) -> list[Message]:
        # TOOD: implement message parsing
        resps = []
        for call in tool_calls:
            name, arguments = call['function']['name'], json.loads(call['function']['arguments'])
            match call['function']['name']:
                case 'submit':
                    resps += [Message(sender=self.name, recipient="admin", content=arguments['answer'])]
                case 'send':
                    resps += [Message(sender=self.name, recipient=arguments['recipient'], content=arguments['message'])]
                case 'broadcast':
                    # add a placeholder string for broadcast and let swarm handle the individual messages
                    resps += [Message(sender=self.name, recipient="all", content=arguments['message'])]
                case _:
                    raise ValueError(f"found unrecognized tool call: {name}")
        return resps
    
    async def step(self) -> tuple[Self, list[Message]]:
        msgs: list[Message] = []
        while not self.queue.empty():
            msgs += [await self.queue.get()]
        if len(msgs) == 0: # TODO: add an edge case here for prior tool calls
            return self, []
        
        comb_msgs = self.aggregator(msgs) 
        chat = self.formatter(comb_msgs, template=self.template)

        self.chats += [chat]
        resp = self.llm(self.chats, tools=self.tools)
        tool_resp = self.tool_processor(resp)
        self.chats += [resp]

        # TODO: add non-comms tool responses to chat history

        messages = []
        if tool_resp is not None:
            messages = self.extract_tool_msgs(tool_resp)
        return self, messages

    @staticmethod
    def get_default_funcs(
        aggregator: Callable[[list[Message]], list[Message]] = None, 
        formatter: Callable[[list[Message], str], Chat] = None,
        tool_processor: Callable[[Chat], Chat] = None,
    ) -> tuple[Callable, Callable, Callable]:
        if aggregator is None: aggregator = identity_aggregator
        if formatter is None: formatter = user_formatter
        if tool_processor is None: tool_processor = openai_tool_processor

        return aggregator, formatter, tool_processor

    @staticmethod
    def from_vllm(
        name: str,
        model_name: str, 
        aggregator: Callable[[list[Message]], list[Message]] = None, 
        formatter: Callable[[list[Message], str], Chat] = None,
        tool_processor: Callable[[Chat], Chat] = None,
        template: str = "{content}",
        chats: list[dict] = None,
        extra_tools: list[dict] = None,
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
        
        return Agent(name, llm_func, aggregator, formatter, tool_processor, template=template, chats=chats, extra_tools=extra_tools)

    @staticmethod
    def from_vllm_server(
        name: str,
        model_name: str,
        url: str,
        api_key: str = "EMPTY",
        aggregator: Callable[[list[Message]], list[Message]] = None, 
        formatter: Callable[[list[Message], str], Chat] = None,
        tool_processor: Callable[[Chat], Chat] = None,
        chats: list[dict] = None, 
        extra_tools: list[dict] = None,
        template: str = "{content}",
        **kwargs
    ) -> Self:
        from openai import OpenAI

        aggregator, formatter, tool_processor = Agent.get_default_funcs(aggregator, formatter, tool_processor)

        client = OpenAI(
            api_key=api_key,
            base_url=url
        )
        def llm_func(chats: list[Chat], tools: list[dict] = []) -> Chat:
            # TODO: fix this monstrosity
            if len(tools) != 0 and 'tools' in kwargs:
                kwargs['tools'] += tools
            elif len(tools) != 0:
                kwargs['tools'] = tools

            return client.chat.completions.create(
                model=model_name,
                messages=chats,
                tool_choice="auto",
                parallel_tool_calls=True,
                **kwargs
            ).choices[0].message.dict()

        return Agent(name, llm_func, aggregator, formatter, tool_processor, template=template, chats=chats, extra_tools=extra_tools)
