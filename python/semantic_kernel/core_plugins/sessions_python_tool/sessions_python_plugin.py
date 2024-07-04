# Copyright (c) Microsoft. All rights reserved.

import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from io import BytesIO
from typing import Annotated, Any, Final

from httpx import AsyncClient, HTTPStatusError
from pydantic import ValidationError

from semantic_kernel.connectors.telemetry import HTTP_USER_AGENT, version_info
from semantic_kernel.const import USER_AGENT
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.core_plugins.sessions_python_tool.sessions_python_settings import (
    ACASessionsSettings,
    SessionsPythonSettings,
)
from semantic_kernel.core_plugins.sessions_python_tool.sessions_remote_file_metadata import SessionsRemoteFileMetadata
from semantic_kernel.exceptions.function_exceptions import FunctionExecutionException, FunctionInitializationError
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel import Kernel
from semantic_kernel.kernel_pydantic import HttpsUrl, KernelBaseModel

logger = logging.getLogger(__name__)


SESSIONS_USER_AGENT = f"{HTTP_USER_AGENT}/{version_info} (Language=Python)"

SESSIONS_API_VERSION = "2024-02-02-preview"


ORCHESTRATOR: Final[str] = r"""
import json


class Orchestrator:
    def __init__(self, generator):
        self.generator = generator

    def run_to_next(self, send_value=None):
        try:
            send_value = json.loads(send_value)
            next_val = self.generator.send(send_value)
            return json.dumps(
                {"result_type": "function_call", "function_name": next_val.function_name, "args": next_val.args}
            )
        except StopIteration as e:
            return json.dumps({"result_type": "return_value", "return_value": e.value})


class FunctionCall:
    def __init__(self, function_name, kwargs):
        self.function_name = function_name
        self.args = kwargs


def call_function(function_name, **kwargs):
    return FunctionCall(function_name, kwargs)
"""


class SessionsPythonTool(KernelBaseModel):
    """A plugin for running Python code in an Azure Container Apps dynamic sessions code interpreter."""

    pool_management_endpoint: HttpsUrl
    settings: SessionsPythonSettings
    kernel: Kernel
    auth_callback: Callable[..., Awaitable[Any]]
    http_client: AsyncClient

    def __init__(
        self,
        auth_callback: Callable[..., Awaitable[Any]],
        kernel: Kernel,
        pool_management_endpoint: str | None = None,
        settings: SessionsPythonSettings | None = None,
        http_client: AsyncClient | None = None,
        env_file_path: str | None = None,
        **kwargs,
    ):
        """Initializes a new instance of the SessionsPythonTool class."""
        try:
            aca_settings = ACASessionsSettings.create(
                env_file_path=env_file_path, pool_management_endpoint=pool_management_endpoint
            )
        except ValidationError as e:
            logger.error(f"Failed to load the ACASessionsSettings with message: {e!s}")
            raise FunctionInitializationError(f"Failed to load the ACASessionsSettings with message: {e!s}") from e

        if not settings:
            settings = SessionsPythonSettings()

        if not http_client:
            http_client = AsyncClient()

        super().__init__(
            pool_management_endpoint=aca_settings.pool_management_endpoint,
            settings=settings,
            auth_callback=auth_callback,
            http_client=http_client,
            kernel=kernel,
            **kwargs,
        )

    # region Helper Methods
    async def _ensure_auth_token(self) -> str:
        """Ensure the auth token is valid."""
        try:
            auth_token = await self.auth_callback()
        except Exception as e:
            logger.error(f"Failed to retrieve the client auth token with message: {e!s}")
            raise FunctionExecutionException(f"Failed to retrieve the client auth token with messages: {e!s}") from e

        return auth_token

    def _sanitize_input(self, code: str) -> str:
        """Sanitize input to the python REPL.

        Remove whitespace, backtick & python (if llm mistakes python console as terminal).

        Args:
            code (str): The query to sanitize
        Returns:
            str: The sanitized query
        """
        # Removes `, whitespace & python from start
        code = re.sub(r"^(\s|`)*(?i:python)?\s*", "", code)
        # Removes whitespace & ` from end
        return re.sub(r"(\s|`)*$", "", code)

    def _construct_remote_file_path(self, remote_file_path: str) -> str:
        """Construct the remote file path.

        Args:
            remote_file_path (str): The remote file path.

        Returns:
            str: The remote file path.
        """
        if not remote_file_path.startswith("/mnt/data/"):
            remote_file_path = f"/mnt/data/{remote_file_path}"
        return remote_file_path

    def _build_url_with_version(self, base_url, endpoint, params):
        """Builds a URL with the provided base URL, endpoint, and query parameters."""
        params["api-version"] = SESSIONS_API_VERSION
        query_string = "&".join([f"{key}={value}" for key, value in params.items()])
        if not base_url.endswith("/"):
            base_url += "/"
        if endpoint.endswith("/"):
            endpoint = endpoint[:-1]
        return f"{base_url}{endpoint}?{query_string}"

    # endregion

    # region Kernel Functions
    @kernel_function(
        description="""Executes the provided Python code.
                     Start and end the code snippet with double quotes to define it as a string.
                     Insert \\n within the string wherever a new line should appear.
                     Add spaces directly after \\n sequences to replicate indentation.
                     Use \" to include double quotes within the code without ending the string.
                     Keep everything in a single line; the \\n sequences will represent line breaks
                     when the string is processed or displayed.
                     """,
        name="execute_code",
    )
    async def execute_code(self, code: Annotated[str, "The valid Python code to execute"]) -> str:
        """Executes the provided Python code.

        Args:
            code (str): The valid Python code to execute
        Returns:
            str: The result of the Python code execution in the form of Result, Stdout, and Stderr
        Raises:
            FunctionExecutionException: If the provided code is empty.
        """
        if not code:
            raise FunctionExecutionException("The provided code is empty")
        result = await self._run_code(code)
        return f"Result: {result['result']}\nStdout: {result['stdout']}\nStderr: {result['stderr']}"

    @kernel_function(
        description="""Executes the provided Python generator function.
                     Start and end the code snippet with double quotes to define it as a string.
                     Insert \\n within the string where a new line should appear.
                     Add spaces directly after \\n to replicate indentation.
                     Use \" to include double quotes within the code without ending the string.
                     Keep everything in a single line; the \\n sequences will represent line breaks
                     Define at least one function, with `def function_name():`.
                     Use `val = yield call_function('plugin_name-function_name', input=1, amount=2)` 
                     to call a different function, 
                     that is defined as the other supplied tools, where the first argument is the function name, 
                     and the rest are the arguments as kwargs.
                     The function defined in main_function_name is called and should be the generator function.""",
        name="execute_generator",
    )
    async def execute_generator(
        self,
        main_function_name: Annotated[str, "The name of the main function to execute"],
        code: Annotated[str, "The valid Python code that defines the functions."],
    ) -> str:
        """Executes the provided Python generator function.

        Args:
            main_function_name (str): The name of the main function to execute
            code (str): The valid Python code to execute
        Returns:
            str: The result of the Python code execution in the form of Result, Stdout, and Stderr
        Raises:
            FunctionExecutionException: If the provided code is empty.
        """
        if not code:
            raise FunctionExecutionException("The provided code is empty")
        # load orchestrator and send to container
        await self._run_code(ORCHESTRATOR)
        # send the code of the user to the container
        await self._run_code(code)
        # start the orchestrator on the container
        code_exec_result = await self._run_code(f"o = Orchestrator({main_function_name}())")
        next_input = json.dumps(None)
        id = 0
        while True:
            logger.debug("Running to next, with input: %s", next_input)
            # next_input is already a json string, double encode it
            code = f"o.run_to_next({json.dumps(next_input)})"
            code_exec_result = await self._run_code(code)
            if "result" not in code_exec_result:
                return f"Stdout: {code_exec_result['stdout']}\nStderr: {code_exec_result['stderr']}"

            result = json.loads(code_exec_result["result"])
            logger.debug("Result: %s", result)
            if result["result_type"] != "function_call":
                return (
                    f"Result: {result['return_value']}\nStdout: "
                    f"{code_exec_result['stdout']}\nStderr: {code_exec_result['stderr']}"
                )
            func_result = await self.kernel.invoke_function_call(
                FunctionCallContent(
                    id=f"local-{id}", name=result["function_name"], arguments=json.dumps(result["args"])
                )
            )
            if func_result and func_result.function_result:
                logger.debug("Function result: %s", func_result.function_result.value)
                next_input = json.dumps(func_result.function_result.value)
            else:
                next_input = json.dumps(None)
            id += 1

    async def _run_code(self, code):
        if not code:
            raise FunctionExecutionException("The provided code is empty")

        if self.settings.sanitize_input:
            code = self._sanitize_input(code)

        auth_token = await self._ensure_auth_token()

        logger.info(f"Executing Python code: {code}")

        self.http_client.headers.update(
            {
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
                USER_AGENT: SESSIONS_USER_AGENT,
            }
        )

        self.settings.python_code = code

        request_body = {
            "properties": self.settings.model_dump(exclude_none=True, exclude={"sanitize_input"}, by_alias=True),
        }
        url = self._build_url_with_version(
            base_url=str(self.pool_management_endpoint),
            endpoint="code/execute/",
            params={"identifier": self.settings.session_id},
        )


<< << << < HEAD
        try:
            response = await self.http_client.post(
                url=url,
                json=request_body,
            )
            response.raise_for_status()
            result = response.json()["properties"]
            return f"Result:\n{result['result']}Stdout:\n{result['stdout']}Stderr:\n{result['stderr']}"
        except HTTPStatusError as e:
            error_message = e.response.text if e.response.text else e.response.reason_phrase
            raise FunctionExecutionException(
                f"Code execution failed with status code {e.response.status_code} and error: {error_message}"
            ) from e
== == == =
        response = await self.http_client.post(
            url=url,
            json=request_body,
        )
        response.raise_for_status()

        return response.json()
>> >> >> > cf923d969(first stuff for local function combined with python)

    @kernel_function(name="upload_file", description="Uploads a file for the current Session ID")
    async def upload_file(
        self,
        *,
        local_file_path: Annotated[str, "The path to the local file on the machine"],
        remote_file_path: Annotated[
            str | None, "The remote path to the file in the session. Defaults to /mnt/data"
        ] = None,
    ) -> Annotated[SessionsRemoteFileMetadata, "The metadata of the uploaded file"]:
        """Upload a file to the session pool.

        Args:
            remote_file_path (str): The path to the file in the session.
            local_file_path (str): The path to the file on the local machine.

        Returns:
            RemoteFileMetadata: The metadata of the uploaded file.

        Raises:
            FunctionExecutionException: If local_file_path is not provided.
        """
        if not local_file_path:
            raise FunctionExecutionException("Please provide a local file path to upload.")

        remote_file_path = self._construct_remote_file_path(remote_file_path or os.path.basename(local_file_path))

        auth_token = await self._ensure_auth_token()
        self.http_client.headers.update(
            {
                "Authorization": f"Bearer {auth_token}",
                USER_AGENT: SESSIONS_USER_AGENT,
            }
        )

        url = self._build_url_with_version(
            base_url=str(self.pool_management_endpoint),
            endpoint="files/upload",
            params={"identifier": self.settings.session_id},
        )

        try:
            with open(local_file_path, "rb") as data:
                files = {"file": (remote_file_path, data, "application/octet-stream")}
                response = await self.http_client.post(url=url, files=files)
                response.raise_for_status()
                response_json = response.json()
                return SessionsRemoteFileMetadata.from_dict(response_json["value"][0]["properties"])
        except HTTPStatusError as e:
            error_message = e.response.text if e.response.text else e.response.reason_phrase
            raise FunctionExecutionException(
                f"Upload failed with status code {e.response.status_code} and error: {error_message}"
            ) from e

    @kernel_function(name="list_files", description="Lists all files in the provided Session ID")
    async def list_files(self) -> list[SessionsRemoteFileMetadata]:
        """List the files in the session pool.

        Returns:
            list[SessionsRemoteFileMetadata]: The metadata for the files in the session pool
        """
        auth_token = await self._ensure_auth_token()
        self.http_client.headers.update(
            {
                "Authorization": f"Bearer {auth_token}",
                USER_AGENT: SESSIONS_USER_AGENT,
            }
        )

        url = self._build_url_with_version(
            base_url=str(self.pool_management_endpoint),
            endpoint="files",
            params={"identifier": self.settings.session_id},
        )

        try:
            response = await self.http_client.get(
                url=url,
            )
            response.raise_for_status()
            response_json = response.json()
            return [SessionsRemoteFileMetadata.from_dict(entry["properties"]) for entry in response_json["value"]]
        except HTTPStatusError as e:
            error_message = e.response.text if e.response.text else e.response.reason_phrase
            raise FunctionExecutionException(
                f"List files failed with status code {e.response.status_code} and error: {error_message}"
            ) from e

    async def download_file(
        self,
        *,
        remote_file_name: Annotated[str, "The name of the file to download, relative to /mnt/data"],
        local_file_path: Annotated[str | None, "The local file path to save the file to, optional"] = None,
    ) -> Annotated[BytesIO | None, "The data of the downloaded file"]:
        """Download a file from the session pool.

        Args:
            remote_file_name: The name of the file to download, relative to `/mnt/data`.
            local_file_path: The path to save the downloaded file to. Should include the extension.
                If not provided, the file is returned as a BufferedReader.

        Returns:
            BufferedReader: The data of the downloaded file.
        """
        auth_token = await self.auth_callback()
        self.http_client.headers.update(
            {
                "Authorization": f"Bearer {auth_token}",
                USER_AGENT: SESSIONS_USER_AGENT,
            }
        )

        url = self._build_url_with_version(
            base_url=str(self.pool_management_endpoint),
            endpoint=f"files/content/{remote_file_name}",
            params={"identifier": self.settings.session_id},
        )

        try:
            response = await self.http_client.get(
                url=url,
            )
            response.raise_for_status()
            if local_file_path:
                with open(local_file_path, "wb") as f:
                    f.write(response.content)
                return None

            return BytesIO(response.content)
        except HTTPStatusError as e:
            error_message = e.response.text if e.response.text else e.response.reason_phrase
            raise FunctionExecutionException(
                f"Download failed with status code {e.response.status_code} and error: {error_message}"
            ) from e
        # endregion
