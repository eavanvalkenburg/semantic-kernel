# Copyright (c) Microsoft. All rights reserved.

import asyncio
import datetime
import os
from typing import Annotated

from azure.core.credentials import AccessToken
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.core_plugins.sessions_python_tool.sessions_python_plugin import SessionsPythonTool
from semantic_kernel.exceptions import FunctionExecutionException
from semantic_kernel.filters import AutoFunctionInvocationContext, FilterTypes
from semantic_kernel.functions import kernel_function

auth_token: AccessToken | None = None

ACA_TOKEN_ENDPOINT: str = "https://dynamicsessions.io/"  # nosec


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
            credential = DefaultAzureCredential(
                interactive_browser_tenant_id=os.environ.get("AZURE_TENANT_ID"),
                exclude_managed_identity_credential=True,
                exclude_developer_cli_credential=True,
                exclude_workload_identity_credential=True,
                exclude_environment_credential=True,
            )

            try:
                auth_token = credential.get_token(scope)
            except ClientAuthenticationError as cae:
                err_messages = getattr(cae, "messages", [])
                raise FunctionExecutionException(
                    f"Failed to retrieve the client auth token with messages: {' '.join(err_messages)}"
                ) from cae

        return auth_token.token

    return auth_callback


class DomoticaPlugin:
    def __init__(self, lights: list[str]):
        self.lights: dict[str, bool] = {light: False for light in lights}

    @kernel_function
    def list_lights(self) -> list[str]:
        """Returns the list of lights, by name."""
        return list(self.lights.keys())

    @kernel_function
    def is_on(self, light: Annotated[str, "the name of the light"]) -> bool:
        """Indicates whether the light is on (true) or off (false)."""
        return self.lights.get(light, False)

    @kernel_function
    def turn_on(self, light: Annotated[str, "the name of the light"]) -> str:
        """Turns the light on."""
        if light in self.lights:
            self.lights[light] = True
            return f"Turned on {light}"
        return f"Light {light} not found"

    @kernel_function
    def turn_off(self, light: Annotated[str, "the name of the light"]) -> str:
        """Turns the light off"""
        if light in self.lights:
            self.lights[light] = False
            return f"Turned off {light}"
        return f"Light {light} not found"

    @kernel_function
    def toggle(self, light: Annotated[str, "the name of the light"]) -> str:
        """Toggles the light."""
        if light in self.lights:
            self.lights[light] = not self.lights[light]
            return f"Toggled {light}"
        return f"Light {light} not found"


kernel = Kernel()
kernel.add_service(AzureChatCompletion(service_id="gpt-4o"))
kernel.add_plugin(DomoticaPlugin(["downstairs", "study", "kitchen", "master_bedroom"]), "house")
kernel.add_plugin(
    SessionsPythonTool(
        auth_callback=auth_callback_factory(ACA_TOKEN_ENDPOINT),
        kernel=kernel,
    ),
    "SessionsTool",
)
kernel.add_function(
    prompt="{{$chat_history}}",
    plugin_name="Chat",
    function_name="Chat",
    prompt_execution_settings=AzureChatPromptExecutionSettings(
        service_id="gpt-4o",
        function_choice_behavior=FunctionChoiceBehavior.Auto(filters={"excluded_plugins": ["Chat"]}),
    ),
)


@kernel.filter(FilterTypes.AUTO_FUNCTION_INVOCATION)
async def auto_function_invocation_filter(context: AutoFunctionInvocationContext, next):
    """A filter that will be called for each function call in the response."""
    print("\033[92m\n  Function called\033[0m")
    print(f"    \033[96mFunction: {context.function.fully_qualified_name}")
    for name, value in context.arguments.items():
        if name == "chat_history":
            print(f"    User message: {value.messages[-1]}")
        else:
            print(f"    Argument {name}: {context.arguments[name]}")
    await next(context)
    print(f"    Result: {context.function_result}\n\033[0m")


chat_history = ChatHistory()
chat_history.add_system_message(
    "You are a agent working with people to control their home. "
    "You have access to the lights and can list them, "
    "turn them on, off or toggle."
    "turning a light off that is already off should be avoided."
    "You have access to a python code executor to perform more complex actions."
)


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

    chat_history.add_user_message(user_input)
    answer = await kernel.invoke(plugin_name="Chat", function_name="Chat", chat_history=chat_history)
    print(f"Agent:> {answer}")
    chat_history.add_message(answer.value[0])
    return True


async def main() -> None:
    print(
        "Welcome to the chat bot!\
        \n  Type 'exit' to exit.\
        \n  This can use python code to do complex things with lights in your home.\
        \n  For instance checking which lights are on and turning all off, or toggling all."
    )
    chatting = True
    while chatting:
        chatting = await chat()


if __name__ == "__main__":
    asyncio.run(main())
