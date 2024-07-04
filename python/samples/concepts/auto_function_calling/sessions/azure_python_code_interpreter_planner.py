# Copyright (c) Microsoft. All rights reserved.

import asyncio
import datetime
import os

from azure.core.credentials import AccessToken
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential

from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.core_plugins.sessions_python_tool.sessions_python_plugin import SessionsPythonTool
from semantic_kernel.core_plugins.time_plugin import TimePlugin
from semantic_kernel.exceptions.function_exceptions import FunctionExecutionException
from semantic_kernel.filters.auto_function_invocation.auto_function_invocation_context import (
    AutoFunctionInvocationContext,
)
from semantic_kernel.filters.filter_types import FilterTypes
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.kernel import Kernel

auth_token: AccessToken | None = None

ACA_TOKEN_ENDPOINT: str = "https://acasessions.io/.default"  # nosec


def auth_callback_factory(scope):
    auth_token = None

    async def auth_callback() -> str:
        """Auth callback for the SessionsPythonTool.
        This is a sample auth callback that shows how to use Azure's DefaultAzureCredential
        to get an access token.
        """
        token = os.environ.get("ACA_TOKEN", None)
        if token:
            return token
        nonlocal auth_token

        current_utc_timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())

        if not auth_token or auth_token.expires_on < current_utc_timestamp:
            credential = DefaultAzureCredential()

            try:
                auth_token = credential.get_token(scope)
            except ClientAuthenticationError as cae:
                err_messages = getattr(cae, "messages", [])
                raise FunctionExecutionException(
                    f"Failed to retrieve the client auth token with messages: {' '.join(err_messages)}"
                ) from cae

        return auth_token.token

    return auth_callback


kernel = Kernel()


@kernel.filter(FilterTypes.AUTO_FUNCTION_INVOCATION)
async def auto_function_invocation_filter(context: AutoFunctionInvocationContext, next):
    """A filter that will be called for each function call in the response."""
    print("\033[92m\n  Function called\033[0m")
    print(f"    \033[96mFunction: {context.function.fully_qualified_name}")
    print(f"    Arguments: {context.arguments}")
    await next(context)
    print(f"    Result: {context.function_result}\n\033[0m")


service_id = "sessions-tool"
chat_service = AzureChatCompletion(
    service_id=service_id,
)
kernel.add_service(chat_service)

sessions_tool = SessionsPythonTool(
    auth_callback=auth_callback_factory(ACA_TOKEN_ENDPOINT),
    kernel=kernel,
)

kernel.add_plugin(sessions_tool, "SessionsTool")
kernel.add_plugin(TimePlugin(), "Time")

chat_function = kernel.add_function(
    prompt="{{$chat_history}}{{$user_input}}",
    plugin_name="ChatBot",
    function_name="Chat",
)

req_settings = AzureChatPromptExecutionSettings(service_id=service_id)

req_settings.function_choice_behavior = FunctionChoiceBehavior.Auto(
    filters={"excluded_plugins": ["ChatBot"], "excluded_functions": ["SessionsTool-execute_code"]}
)

arguments = KernelArguments(settings=req_settings)

history = ChatHistory()


async def chat() -> bool:
    try:
        user_input = input("User:> ")
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False

    arguments["chat_history"] = history
    arguments["user_input"] = user_input
    answer = await kernel.invoke(
        function=chat_function,
        arguments=arguments,
    )
    print(f"Mosscap:> {answer}")
    history.add_user_message(user_input)
    history.add_assistant_message(str(answer))
    return True


async def main() -> None:
    print(
        "Welcome to the chat bot!\
        \n  Type 'exit' to exit.\
        \n  Try a Python code execution question to see the function calling in action (i.e. what is 1+1?)."
    )
    chatting = True
    while chatting:
        chatting = await chat()


if __name__ == "__main__":
    asyncio.run(main())
