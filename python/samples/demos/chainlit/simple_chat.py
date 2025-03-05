# Copyright (c) Microsoft. All rights reserved.

import logging

import chainlit as cl
from pydantic import BaseModel

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.core_plugins.math_plugin import MathPlugin
from semantic_kernel.core_plugins.time_plugin import TimePlugin
from semantic_kernel.filters.auto_function_invocation.auto_function_invocation_context import (
    AutoFunctionInvocationContext,
)
from semantic_kernel.filters.functions.function_invocation_context import FunctionInvocationContext

logger = logging.getLogger(__name__)

#####################################################################
# This sample demonstrates how to build a conversational chatbot    #
# using Semantic Kernel, featuring auto function calling,           #
# non-streaming responses, and support for math and time plugins.   #
# The chatbot is designed to interact with the user, call functions #
# as needed, and return responses.                                  #
#####################################################################
# Copyright (c) Microsoft. All rights reserved.


# System message defining the behavior and persona of the chat bot.
system_message = """
You are a chat bot. Your name is Mosscap and
you have one goal: figure out what people need.
Your full name, should you need to know it, is
Splendid Speckled Mosscap. You communicate
effectively, but you tend to answer with long
flowery prose. You are also a math wizard,
especially for adding and subtracting.
You also excel at joke telling, where your tone is often sarcastic.
Once you have the answer I am looking for,
you will return a full answer to me as soon as possible.
"""

# Create and configure the kernel.
kernel = Kernel()

# Load some sample plugins (for demonstration of function calling).
kernel.add_plugin(MathPlugin(), plugin_name="math")
kernel.add_plugin(TimePlugin(), plugin_name="time")

# Please make sure you have configured your environment correctly for the selected chat completion service.
chat_completion_service = OpenAIChatCompletion(service_id="chat")
request_settings = OpenAIChatPromptExecutionSettings(
    function_choice_behavior=FunctionChoiceBehavior.Auto(filters={"excluded_plugins": ["ChatBot"]})
)

kernel.add_service(chat_completion_service)

# Create a chat history to store the system message, initial messages, and the conversation.
history = ChatHistory()
history.add_system_message(system_message)
history.add_user_message("Hi there, who are you?")
history.add_assistant_message("I am Mosscap, a chat bot. I'm trying to figure out what people need.")


class SemanticKernelFilter(BaseModel):
    excluded_plugins: list[str] | None = None

    async def auto_func_loop(self, context: AutoFunctionInvocationContext, next):
        if "__step_id__" in context.arguments:
            step = context.arguments["__step_id__"]
            async with step:
                await next(context)
        else:
            await next(context)

    async def func_step(self, context: FunctionInvocationContext, next):
        if self.excluded_plugins and context.function.plugin_name in self.excluded_plugins:
            await next(context)
            return
        async with cl.Step(type="tool", name=context.function.fully_qualified_name) as step:
            input_dict = {}
            for key, value in context.arguments.items():
                if key in ["chat_history", "__step_id__"]:
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


filter = SemanticKernelFilter(excluded_plugins=["ChatBot"])
kernel.add_filter("function_invocation", filter.func_step)
# kernel.add_filter("auto_function_invocation", filter.auto_func_loop)


@cl.on_message
async def chat(message: cl.Message):
    history.add_user_message(message.content)
    step = None
    cl_msg = cl.Message(content="")
    assistant_responses = []
    tool_responses = []
    async for msg in chat_completion_service.get_streaming_chat_message_content(
        chat_history=history,
        settings=request_settings,
        kernel=kernel,
    ):
        if msg and msg.role == "assistant":
            if not step and any(isinstance(item, FunctionCallContent) for item in msg.items):
                step = cl.Step(type="tool", name="Auto Function Calling")
                await step.__aenter__()
                cl_msg.parent_id = step.id
            assistant_responses.append(msg)
            await cl_msg.stream_token(msg.content)
        if msg and msg.role == "tool":
            tool_responses.append(msg)
    if step:
        await step.__aexit__(None, None, None)
    full_response = sum(assistant_responses[1:], assistant_responses[0])
    history.add_message(full_response)
    if tool_responses:
        tool_response = sum(tool_responses[1:], tool_responses[0])
        if tool_response:
            history.add_message(tool_response)
    await cl_msg.send()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Simple function call without arguments",
            message="What time is it?",
        ),
        cl.Starter(
            label="Math",
            message="What is the current time + 293847 minus 2934?",
        ),
    ]
