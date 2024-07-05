# Copyright (c) Microsoft. All rights reserved.

import asyncio
import datetime
import os

from azure.core.credentials import AccessToken
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential

from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.core_plugins.math_plugin import MathPlugin
from semantic_kernel.core_plugins.sessions_python_tool.sessions_python_plugin import SessionsPythonTool
from semantic_kernel.exceptions.function_exceptions import FunctionExecutionException
from semantic_kernel.filters.auto_function_invocation.auto_function_invocation_context import (
    AutoFunctionInvocationContext,
)
from semantic_kernel.filters.filter_types import FilterTypes
from semantic_kernel.kernel import Kernel

auth_token: AccessToken | None = None

ACA_TOKEN_ENDPOINT: str = "https://dynamicsessions.io"  # nosec


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
kernel.add_plugin(MathPlugin(), "math")
kernel.add_plugin(sessions_tool, "SessionsTool")


async def main() -> None:
    code = "def main():\n\tres = yield call_function('math-Add', input=1, amount=2)\n\treturn res"
    result = await kernel.invoke(
        function_name="execute_generator",
        plugin_name="SessionsTool",
        main_function_name="main",
        code=code,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
