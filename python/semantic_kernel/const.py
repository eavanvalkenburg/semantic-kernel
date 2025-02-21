# Copyright (c) Microsoft. All rights reserved.

from typing import ClassVar, Final

from pydantic import Field, Literal

from semantic_kernel.contents import FunctionResultContent, RealtimeEvent

METADATA_EXCEPTION_KEY: Final[str] = "exception"
DEFAULT_SERVICE_NAME: Final[str] = "default"
USER_AGENT: Final[str] = "User-Agent"
PARSED_ANNOTATION_UNION_DELIMITER: Final[str] = ","
DEFAULT_FULLY_QUALIFIED_NAME_SEPARATOR: Final[str] = "-"

AUTO_FUNCTION_INVOCATION_SPAN_NAME: Final[str] = "AutoFunctionInvocationLoop"


class RealtimeFunctionResultEvent(RealtimeEvent):
    """Function result event type."""

    event_type: ClassVar[Literal["function_result"]] = "function_result"  # type: ignore
    function_result: FunctionResultContent = Field(..., description="Function result content.")
