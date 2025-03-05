# Copyright (c) Microsoft. All rights reserved.

import logging

import chainlit as cl
from pydantic import BaseModel

from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import (
    KernelFunctionSelectionStrategy,
)
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import (
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
)
from semantic_kernel.core_plugins.math_plugin import MathPlugin
from semantic_kernel.core_plugins.time_plugin import TimePlugin
from semantic_kernel.filters import FunctionInvocationContext
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
from semantic_kernel.kernel import Kernel

logger = logging.getLogger(__name__)


chat_completion_service = OpenAIChatCompletion(service_id="chat")
planner = ChatCompletionAgent(
    service=chat_completion_service,
    plugins=[MathPlugin(), TimePlugin()],
    name="planner",
    instructions="""You only respond to question directed at you, there are other agents to handle some tasks.""",
    function_choice_behavior=FunctionChoiceBehavior.Auto(),
)
reviewer = ChatCompletionAgent(
    service=chat_completion_service,
    name="reviewer",
    instructions="""
You are an art director who has opinions about copywriting born of a love for David Ogilvy.
The goal is to determine if the given copy is acceptable to print.
If so, state that it is approved.
If not, provide insight on how to refine suggested copy without example.
""",
)
writer = ChatCompletionAgent(
    service=chat_completion_service,
    name="writer",
    instructions="""
You are a copywriter with ten years of experience and are known for brevity and a dry humor.
The goal is to refine and decide on the single best copy as an expert in the field.
Only provide a single proposal per response.
You're laser focused on the goal at hand.
Don't waste time with chit chat.
Consider suggestions when refining an idea.
""",
)

selection_function = KernelFunctionFromPrompt(
    function_name="selection",
    prompt="""
        Determine which participant takes the next turn in a conversation based on the the most recent participant.
        State only the name of the participant to take the next turn. 
        Do not use the same participant twice in a row, unless it is the planner.
        
        Choose only from these participants:
        - reviewer
        - planner
        - writer
        
        Always follow these rules when selecting the next participant:
        - For user input about copywriting, use the writer.
        - After writer replies, it is reviewer's turn.
        - After reviewer provides feedback, it is writer's turn.
        - Any other question should be answered by planner.

        History:
        {{$history}}
        """,
)
termination_function = KernelFunctionFromPrompt(
    function_name="termination",
    prompt="""
        Determine if the question from the user has been answered. 
        Whenever the reviewer responds with critique then it is not done. 
        If so, respond with a single word: yes

        History:
        {{$history}}
        """,
)

kernel = Kernel()
kernel.add_service(chat_completion_service)
selection_function = kernel.add_function("strategies", selection_function)
termination_function = kernel.add_function("strategies", termination_function)
group_chat = AgentGroupChat(
    agents=[
        planner,
        writer,
        reviewer,
    ],
    termination_strategy=KernelFunctionTerminationStrategy(
        agents=[reviewer, planner],
        function=termination_function,
        kernel=kernel,
        result_parser=lambda result: str(result.value[0]).lower() == "yes",
        history_variable_name="history",
        maximum_iterations=10,
        automatic_reset=True,
    ),
    selection_strategy=KernelFunctionSelectionStrategy(
        function=selection_function,
        kernel=kernel,
        result_parser=lambda result: result.value[0].content if result.value is not None else "planner",
        agent_variable_name="agents",
        history_variable_name="history",
    ),
)


async def func_step(context: FunctionInvocationContext, next):
    if context.function.plugin_name == "ChatBot":
        await next(context)
        return
    async with cl.Step(type="tool", name=context.function.fully_qualified_name) as step:
        input_dict = {}
        for key, value in context.arguments.items():
            if key == "chat_history":
                continue
            if isinstance(value, BaseModel):
                input_dict[key] = value.model_dump()
            else:
                input_dict[key] = value
        step.input = input_dict or ""
        await step.send()
        await next(context)
        if context.result:
            step.output = context.result.value
        await step.update()


planner.kernel.add_filter("function_invocation", func_step)


@cl.on_message
async def chat(message: cl.Message):
    await group_chat.add_chat_message(message=message.content)

    last_msg_author = None
    async for msg in group_chat.invoke_stream():
        if msg.role == "assistant":
            if msg.name != last_msg_author:
                cl_msg = cl.Message(content="", author=msg.name)
                last_msg_author = msg.name
            await cl_msg.stream_token(msg.content)
            await cl_msg.update()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Copy writing",
            message="Write a slogan for electric vehicles for a teen magazine.",
        ),
        cl.Starter(
            label="Math",
            message="What is the current time + 293847 minus 2934?",
        ),
    ]
