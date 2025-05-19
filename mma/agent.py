import json
import asyncio

from pydantic import BaseModel
from typing import Union, Self, Callable

Chat = dict[str, Union[int, float, str, list]]

DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send",
            "description": "Send a brief text message to someone else, such as a collaborator or teammate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "The recipient of your message (all possible recipients should be listed in system prompt or user message).",
                    },
                    "message": {
                        "type": "string",
                        "description": "The message to send the recipient (e.g. questions, statements). Should be brief and avoid small talk.",
                    },
                },
                "required": ["recipient", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "broadcast",
            "description": "Broadcast a brief text message to everyone at the same time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send everyone (e.g. questions, statements). Should be brief and avoid small talk.",
                    },
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_registry",
            "description": "Get a list of all currently available teammates to work with.",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit a final answer to a question after collaborating with teammates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to submit.",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


def merge_chats(chats: list[Chat], delimiter: str = "\n\n") -> list[Chat]:
    last_role = chats[0]["role"]
    running_content = chats[0]["content"]
    merged_chats = []
    # maybe change this impl to be more general instead of hardcoding each field
    for i in range(1, len(chats)):
        if (
            chats[i]["role"] != last_role or chats[i]["role"] != "user"
        ):  # only merge user messages
            merged = {"role": last_role, "content": running_content}
            if chats[i - 1]["role"] == "tool":
                merged["tool_call_id"] = chats[i - 1]["tool_call_id"]
            elif chats[i - 1]["role"] == "assistant":
                merged["tool_calls"] = chats[i - 1]["tool_calls"]
            merged_chats += [merged]
            last_role = chats[i]["role"]
            running_content = chats[i]["content"]
        else:
            running_content += delimiter + chats[i]["content"]
    merged = {"role": last_role, "content": running_content}
    if chats[-1]["role"] == "tool":
        merged["tool_call_id"] = chats[i - 1]["tool_call_id"]
    elif chats[-1]["role"] == "assistant":
        merged["tool_calls"] = chats[i - 1]["tool_calls"]
    merged_chats += [merged]
    return merged_chats


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
        "content": template.format(
            content="You have received the following messages:\n"
            + "\n\n".join([str(msg) for msg in msgs])
        ),
    }


def no_tools(chat: Chat):
    return []


class Agent:
    name: str
    chats: list[Chat]

    llm: Callable[[list[Chat]], Chat]
    aggregator: Callable[[list[Message]], list[Message]]
    formatter: Callable[[list[Message], str], str]
    tool_processor: Callable[[Chat], list[Chat]]

    template: str
    tools: list[dict]

    queue: asyncio.Queue

    def __init__(
        self,
        name: str,
        llm: Callable[[list[Chat], list[dict]], Chat],
        aggregator: Callable[[list[Message]], list[Message]],
        formatter: Callable[[list[Message], str], Chat],
        tool_processor: Callable[[Chat], list[Chat]],
        chats: list[dict[str, Union[int, str]]] = None,
        extra_tools: list[dict] = None,
        template: str = "{content}",
        max_msgs: int = 0,
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

        self.registry_func = lambda: None
        self.registered = False

        self.tools = DEFAULT_TOOLS + extra_tools

    def add_agent_registry(self, registry_func: Callable[[], str]) -> Self:
        self.registry_func = registry_func
        self.registered = True
        return self

    async def send_msg(self, msg: Message) -> Self:
        await self.queue.put(msg)
        return self

    def extract_tool_msgs(self, tool_calls: list[dict]) -> list[Message]:
        resps = []
        for call in tool_calls:
            name, arguments = call["function"]["name"], json.loads(
                call["function"]["arguments"]
            )
            match call["function"]["name"]:
                case "submit":
                    resps += [
                        Message(
                            sender=self.name,
                            recipient="admin",
                            content=arguments["answer"],
                        )
                    ]
                case "send":
                    resps += [
                        Message(
                            sender=self.name,
                            recipient=arguments["recipient"],
                            content=arguments["message"],
                        )
                    ]
                case "broadcast":
                    # add a placeholder recipient for broadcast and let swarm handle the individual messages
                    resps += [
                        Message(
                            sender=self.name,
                            recipient="all",
                            content=arguments["message"],
                        )
                    ]
                case _:
                    valid_tool = any(
                        [tool["function"]["name"] == name for tool in self.tools]
                    )
                    if not valid_tool:
                        raise ValueError(f"found unrecognized tool call: {name}")
        return resps

    async def step(self) -> tuple[Self, list[Message]]:
        msgs: list[Message] = []
        while not self.queue.empty():
            msgs += [await self.queue.get()]

        # process any messages we've received
        if len(msgs) != 0:
            comb_msgs = self.aggregator(msgs)
            chat = self.formatter(comb_msgs, template=self.template)
            self.chats += [chat]

        # if we've already responded to something, just idle
        if self.chats[-1]["role"] == "assistant":
            return self, []

        self.chats = merge_chats(self.chats)
        resp = self.llm(self.chats, tools=self.tools)
        tool_calls = resp["tool_calls"]
        self.chats += [resp]

        messages = []
        if len(tool_calls) > 0:
            messages = self.extract_tool_msgs(tool_calls)
            tool_resps = self.tool_processor(tool_calls)

            for call in tool_calls:
                # TODO: make this cleaner and bring it into a non_msg tool call func or something idk just do something not thought up at like 3am
                if call["function"]["name"] == "get_registry":
                    tool_resps += [
                        {
                            "tool_call_id": call["id"],
                            "role": "tool",
                            "content": self.registry_func(),
                        }
                    ]
            self.chats += tool_resps
        return self, messages

    @staticmethod
    def get_default_funcs(
        aggregator: Callable[[list[Message]], list[Message]] = None,
        formatter: Callable[[list[Message], str], Chat] = None,
        tool_processor: Callable[[Chat], list[Chat]] = None,
    ) -> tuple[Callable, Callable, Callable]:
        if aggregator is None:
            aggregator = identity_aggregator
        if formatter is None:
            formatter = user_formatter
        if tool_processor is None:
            tool_processor = no_tools

        return aggregator, formatter, tool_processor

    @staticmethod
    def from_vllm(
        name: str,
        model_name: str,
        aggregator: Callable[[list[Message]], list[Message]] = None,
        formatter: Callable[[list[Message], str], Chat] = None,
        tool_processor: Callable[[Chat], list[Chat]] = None,
        template: str = "{content}",
        chats: list[dict] = None,
        extra_tools: list[dict] = None,
        **init_kwargs,
    ) -> Self:
        from vllm import LLM

        aggregator, formatter, tool_processor = Agent.get_default_funcs(
            aggregator, formatter, tool_processor
        )

        engine = LLM(model_name, **init_kwargs)

        def llm_func(chats: list[Chat], **kwargs) -> Chat:
            resp = engine.chat(chats, **kwargs)
            return {"role": "assistant", "content": resp.outputs[0].text}

        return Agent(
            name,
            llm_func,
            aggregator,
            formatter,
            tool_processor,
            template=template,
            chats=chats,
            extra_tools=extra_tools,
        )

    @staticmethod
    def from_vllm_server(
        name: str,
        model_name: str,
        url: str,
        api_key: str = "EMPTY",
        aggregator: Callable[[list[Message]], list[Message]] = None,
        formatter: Callable[[list[Message], str], Chat] = None,
        tool_processor: Callable[[Chat], list[Chat]] = None,
        chats: list[dict] = None,
        extra_tools: list[dict] = None,
        template: str = "{content}",
        **kwargs,
    ) -> Self:
        from openai import OpenAI

        aggregator, formatter, tool_processor = Agent.get_default_funcs(
            aggregator, formatter, tool_processor
        )

        client = OpenAI(api_key=api_key, base_url=url)

        def llm_func(chats: list[Chat], tools: list[dict] = []) -> Chat:
            # TODO: fix this monstrosity
            if len(tools) != 0 and "tools" in kwargs:
                kwargs["tools"] += tools
            elif len(tools) != 0:
                kwargs["tools"] = tools

            return (
                client.chat.completions.create(
                    model=model_name,
                    messages=chats,
                    tool_choice="auto",
                    parallel_tool_calls=True,
                    **kwargs,
                )
                .choices[0]
                .message.dict()
            )

        return Agent(
            name,
            llm_func,
            aggregator,
            formatter,
            tool_processor,
            template=template,
            chats=chats,
            extra_tools=extra_tools,
        )
