# Copyright (c) Microsoft. All rights reserved.

import base64
from datetime import datetime
from random import randint
from uuid import uuid4

import chainlit as cl
import numpy as np
from chainlit.logger import logger
from pydantic import BaseModel

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIRealtimeWebsocket
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_realtime_execution_settings import (
    OpenAIRealtimeExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_realtime import ListenEvents, SendEvents
from semantic_kernel.connectors.ai.realtime_client_base import RealtimeClientBase
from semantic_kernel.contents import RealtimeEvent, RealtimeTextEvent
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.filters.functions.function_invocation_context import FunctionInvocationContext
from semantic_kernel.functions import kernel_function


class HelperPlugin:
    """Helper plugin for the Semantic Kernel."""

    @kernel_function
    def get_weather(self, location: str) -> str:
        """Get the weather for a location."""
        logger.info(f"@ Getting weather for {location}")
        weather_conditions = ("sunny", "hot", "cloudy", "raining", "freezing", "snowing")
        weather = weather_conditions[randint(0, len(weather_conditions) - 1)]  # nosec
        return f"The weather in {location} is {weather}."

    @kernel_function
    def get_date_time(self) -> str:
        """Get the current date and time."""
        logger.info("@ Getting current datetime")
        return f"The current date and time is {datetime.now().isoformat()}."


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


kernel = Kernel()
kernel.add_plugin(HelperPlugin(), plugin_name="helper")
kernel.add_filter("function_invocation", func_step)


# async def audio_callback(audio: np.ndarray):
#     """Callback function to handle audio data."""
#     await cl.context.emitter.send_audio_chunk(
#         cl.OutputAudioChunk(
#             mimeType="pcm16", data=base64.b64encode(audio.tobytes()), track=cl.user_session.get("track_id")
#         )
#     )


@cl.on_chat_start
async def start():
    cl.user_session.set("track_id", str(uuid4()))
    openai_realtime = OpenAIRealtimeWebsocket()
    settings = OpenAIRealtimeExecutionSettings(
        instructions="You are a helpfull assistant.",
        voice="shimmer",
        function_choice_behavior="auto",
    )
    await cl.Message(content="Welcome to the Chainlit x Semantic Kernel realtime example. Press `P` to talk!").send()
    await openai_realtime.create_session(settings=settings, kernel=kernel, create_response=True)
    cl.user_session.set("openai_realtime", openai_realtime)
    current_msg = cl.Message(content="")
    async for event in openai_realtime.receive():
        match event.event_type:
            case "audio":
                await cl.context.emitter.send_audio_chunk(
                    cl.OutputAudioChunk(
                        mimeType="pcm16",
                        data=event.audio.data,
                        track=cl.user_session.get("track_id"),
                    )
                )
            case "text":
                await current_msg.stream_token(event.text.text)
            case _:
                match event.service_type:
                    case ListenEvents.RESPONSE_CREATED:
                        current_msg = cl.Message(content="")
                    case ListenEvents.ERROR:
                        logger.warning(event.service_event)


@cl.on_message
async def on_message(message: cl.Message):
    openai_realtime: RealtimeClientBase = cl.user_session.get("openai_realtime")
    await openai_realtime.send(RealtimeTextEvent(text=TextContent(text=message.content)))
    await openai_realtime.send(RealtimeEvent(service_type=SendEvents.RESPONSE_CREATE))


@cl.on_audio_start
async def on_audio_start():
    openai_realtime: RealtimeClientBase = cl.user_session.get("openai_realtime")
    if not openai_realtime:
        await cl.ErrorMessage(content="RealtimeClient is not connected").send()
        return False
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    array_buffer = np.array(chunk.data)
    if array_buffer.dtype == np.float32:
        int16_array = np.clip(array_buffer, -1, 1) * 32767
        array_buffer = int16_array.astype(np.int16)
    elif array_buffer.dtype == np.int16:
        array_buffer = array_buffer.tobytes()
    else:
        array_buffer = array_buffer.tobytes()

    audio = base64.b64encode(array_buffer).decode("utf-8")

    openai_realtime: RealtimeClientBase = cl.user_session.get("openai_realtime")

    await openai_realtime.send(
        RealtimeEvent(service_type=SendEvents.INPUT_AUDIO_BUFFER_APPEND, service_event={"audio": audio})
    )


# @cl.on_audio_end
@cl.on_chat_end
@cl.on_stop
async def on_end():
    openai_realtime: RealtimeClientBase = cl.user_session.get("openai_realtime")
    await openai_realtime.close_session()
