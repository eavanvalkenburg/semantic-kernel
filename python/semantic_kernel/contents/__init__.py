# Copyright (c) Microsoft. All rights reserved.

import base64
import binascii
import logging
import mimetypes
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Mapping, MutableMapping, Sequence
from enum import Enum
from functools import singledispatchmethod
from html import unescape
from pathlib import Path
from typing import Annotated, Any, ClassVar, Final, Literal, TypeVar, Union, overload
from xml.etree.ElementTree import Element, tostring

if sys.version_info >= (3, 11):
    from typing import Self  # pragma: no cover
else:
    from typing_extensions import Self  # pragma: no cover
if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


from defusedxml import ElementTree
from defusedxml.ElementTree import XML, ParseError
from numpy import ndarray
from pydantic import (
    BaseModel,
    Field,
    UrlConstraints,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_core import FilePath, PrivateAttr, Url, computed_field
from typing_extensions import deprecated

from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.const import DEFAULT_FULLY_QUALIFIED_NAME_SEPARATOR
from semantic_kernel.exceptions import (
    ChatHistoryReducerException,
    ContentAdditionException,
    ContentException,
    ContentInitializationError,
    ContentSerializationError,
    FunctionCallInvalidArgumentsException,
    FunctionCallInvalidNameException,
)
from semantic_kernel.functions.function_result import FunctionResult
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.utils.experimental_decorator import experimental_class, experimental_function

logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound="KernelContent")

AUDIO_CONTENT_TAG: Final[str] = "audio"
CHAT_MESSAGE_CONTENT_TAG: Final[str] = "message"
CHAT_HISTORY_TAG: Final[str] = "chat_history"
TEXT_CONTENT_TAG: Final[str] = "text"
IMAGE_CONTENT_TAG: Final[str] = "image"
ANNOTATION_CONTENT_TAG: Final[str] = "annotation"
STREAMING_ANNOTATION_CONTENT_TAG: Final[str] = "streaming_annotation"
BINARY_CONTENT_TAG: Final[str] = "binary"
FILE_REFERENCE_CONTENT_TAG: Final[str] = "file_reference"
STREAMING_FILE_REFERENCE_CONTENT_TAG: Final[str] = "streaming_file_reference"
FUNCTION_CALL_CONTENT_TAG: Final[str] = "function_call"
FUNCTION_RESULT_CONTENT_TAG: Final[str] = "function_result"
DISCRIMINATOR_FIELD: Final[str] = "content_type"

EMPTY_VALUES: Final[list[str | None]] = ["", "{}", None]

DEFAULT_SUMMARIZATION_PROMPT = """
Provide a concise and complete summarization of the entire dialog that does not exceed 5 sentences.

This summary must always:
- Consider both user and assistant interactions
- Maintain continuity for the purpose of further dialog
- Include details from any existing summary
- Focus on the most significant aspects of the dialog

This summary must never:
- Critique, correct, interpret, presume, or assume
- Identify faults, mistakes, misunderstanding, or correctness
- Analyze what has not occurred
- Exclude details from any existing summary
"""
SUMMARY_METADATA_KEY = "__summary__"


class ContentTypes(str, Enum):
    """Content types enumeration."""

    AUDIO_CONTENT = AUDIO_CONTENT_TAG
    ANNOTATION_CONTENT = ANNOTATION_CONTENT_TAG
    BINARY_CONTENT = BINARY_CONTENT_TAG
    CHAT_MESSAGE_CONTENT = CHAT_MESSAGE_CONTENT_TAG
    IMAGE_CONTENT = IMAGE_CONTENT_TAG
    FILE_REFERENCE_CONTENT = FILE_REFERENCE_CONTENT_TAG
    FUNCTION_CALL_CONTENT = FUNCTION_CALL_CONTENT_TAG
    FUNCTION_RESULT_CONTENT = FUNCTION_RESULT_CONTENT_TAG
    STREAMING_ANNOTATION_CONTENT = STREAMING_ANNOTATION_CONTENT_TAG
    STREAMING_FILE_REFERENCE_CONTENT = STREAMING_FILE_REFERENCE_CONTENT_TAG
    TEXT_CONTENT = TEXT_CONTENT_TAG


class AuthorRole(str, Enum):
    """Author role enum."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class FinishReason(str, Enum):
    """Finish Reason enum."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    FUNCTION_CALL = "function_call"


DataUrl = Annotated[Url, UrlConstraints(allowed_schemes=["data"])]


class KernelContent(KernelBaseModel, ABC):
    """Base class for all kernel contents."""

    # NOTE: if you wish to hold on to the inner content, you are responsible
    # for saving it before serializing the content/chat history as it won't be included.
    content_type: ContentTypes | None = Field(default=None, init=False)
    inner_content: Annotated[Any | None, Field(exclude=True)] = None
    ai_model_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the content."""
        pass

    @abstractmethod
    def to_element(self) -> Any:
        """Convert the instance to an Element."""
        pass

    @classmethod
    @abstractmethod
    def from_element(cls: type[_T], element: Any) -> _T:
        """Create an instance from an Element."""
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert the instance to a dictionary."""
        pass


@experimental_class
class AnnotationContent(KernelContent):
    """Annotation content."""

    content_type: Literal[ContentTypes.ANNOTATION_CONTENT] = Field(default=ANNOTATION_CONTENT_TAG, init=False)  # type: ignore
    tag: ClassVar[str] = ANNOTATION_CONTENT_TAG
    file_id: str | None = None
    quote: str | None = None
    start_index: int | None = None
    end_index: int | None = None

    def __str__(self) -> str:
        """Return the string representation of the annotation content."""
        return f"AnnotationContent(file_id={self.file_id}, quote={self.quote}, start_index={self.start_index}, end_index={self.end_index})"  # noqa: E501

    def to_element(self) -> Element:
        """Convert the annotation content to an Element."""
        element = Element(self.tag)
        if self.file_id:
            element.set("file_id", self.file_id)
        if self.quote:
            element.set("quote", self.quote)
        if self.start_index is not None:
            element.set("start_index", str(self.start_index))
        if self.end_index is not None:
            element.set("end_index", str(self.end_index))
        return element

    @classmethod
    def from_element(cls: type[_T], element: Element) -> _T:
        """Create an instance from an Element."""
        return cls(
            file_id=element.get("file_id"),
            quote=element.get("quote"),
            start_index=int(element.get("start_index")) if element.get("start_index") else None,  # type: ignore
            end_index=int(element.get("end_index")) if element.get("end_index") else None,  # type: ignore
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the instance to a dictionary."""
        return {
            "type": "text",
            "text": f"{self.file_id} {self.quote} (Start Index={self.start_index}->End Index={self.end_index})",
        }


class DataUri(KernelBaseModel, validate_assignment=True):
    """A class to represent a data uri.

    If a array is provided, that will be used as the data since it is the most efficient,
    otherwise the bytes will be used, or the string will be converted to bytes.

    When updating either array or bytes, the other will not be updated.

    Args:
        data_bytes: The data as bytes.
        data_str: The data as a string.
        data_array: The data as a numpy array.
        mime_type: The mime type of the data.
        parameters: Any parameters for the data.
        data_format: The format of the data (e.g. base64).

    """

    data_array: ndarray | None = None
    data_bytes: bytes | None = None
    mime_type: str | None = None
    parameters: MutableMapping[str, str] = Field(default_factory=dict)
    data_format: str | None = None

    def __init__(
        self,
        data_bytes: bytes | None = None,
        data_str: str | None = None,
        data_array: ndarray | None = None,
        mime_type: str | None = None,
        parameters: Sequence[str] | Mapping[str, str] | None = None,
        data_format: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the data uri.

        Make sure to set the data_format to base64 so that it can be decoded properly.

        Args:
            data_bytes: The data as bytes.
            data_str: The data as a string.
            data_array: The data as a numpy array.
            mime_type: The mime type of the data.
            parameters: Any parameters for the data.
            data_format: The format of the data (e.g. base64).
            kwargs: Any additional arguments.
        """
        args: dict[str, Any] = {}
        if data_bytes is not None:
            args["data_bytes"] = data_bytes
        if data_array is not None:
            args["data_array"] = data_array

        if mime_type is not None:
            args["mime_type"] = mime_type
        if parameters is not None:
            args["parameters"] = parameters
        if data_format is not None:
            args["data_format"] = data_format

        if data_str is not None and not data_bytes:
            if data_format and data_format.lower() == "base64":
                try:
                    args["data_bytes"] = base64.b64decode(data_str, validate=True)
                except binascii.Error as exc:
                    raise ContentInitializationError("Invalid base64 data.") from exc
            else:
                args["data_bytes"] = data_str.encode("utf-8")
        if "data_array" not in args and "data_bytes" not in args:
            raise ContentInitializationError("Either data_bytes, data_str or data_array must be provided.")
        super().__init__(**args, **kwargs)

    def update_data(self, value: str | bytes | ndarray) -> None:
        """Update the data, using either a string or bytes."""
        match value:
            case ndarray():
                self.data_array = value
            case str():
                if self.data_format and self.data_format.lower() == "base64":
                    self.data_bytes = base64.b64decode(value, validate=True)
                else:
                    self.data_bytes = value.encode("utf-8")
            case _:
                self.data_bytes = value

    @field_validator("parameters", mode="before")
    def _validate_parameters(cls, value: list[str] | dict[str, str] | None) -> dict[str, str]:
        if not value:
            return {}
        if isinstance(value, dict):
            return value

        new: dict[str, str] = {}
        for item in value:
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ContentInitializationError("Invalid data uri format. The parameter is missing a value.")
            name, val = item.split("=", maxsplit=1)
            new[name] = val
        return new

    @classmethod
    def from_data_uri(cls: type[_T], data_uri: str | Url, default_mime_type: str = "text/plain") -> _T:
        """Create a DataUri object from a data URI string or pydantic URL."""
        if isinstance(data_uri, str):
            try:
                data_uri = Url(data_uri)
            except ValidationError as exc:
                raise ContentInitializationError("Invalid data uri format.") from exc

        data = data_uri.path
        if not data or "," not in data:
            raise ContentInitializationError("Invalid data uri format. The data is missing.")

        pattern = "(((?P<mime_type>[a-zA-Z]+/[a-zA-Z-]+)(?P<parameters>(;[a-zA-Z0-9]+=+[a-zA-Z0-9]+)*))?(;+(?P<data_format>.*)))?(,(?P<data_str>.*))"  # noqa: E501
        match = re.match(pattern, data)
        if not match:
            raise ContentInitializationError("Invalid data uri format.")
        matches = match.groupdict()
        if not matches.get("data_format"):
            matches.pop("data_format")
        if not matches.get("parameters"):
            matches.pop("parameters")
        else:
            matches["parameters"] = matches["parameters"].strip(";").split(";")
        if not matches.get("mime_type"):
            matches["mime_type"] = default_mime_type
        return cls(**matches)  # type: ignore

    def to_string(self, metadata: dict[str, str] = {}) -> str:
        """Return the data uri as a string."""
        parameters = ";".join([f"{key}={val}" for key, val in metadata.items()])
        parameters = f";{parameters}" if parameters else ""
        data_format = f"{self.data_format}" if self.data_format else ""
        return f"data:{self.mime_type or ''}{parameters};{data_format},{self._data_str()}"

    def __eq__(self, value: object) -> bool:
        """Check if the data uri is equal to another."""
        if not isinstance(value, DataUri):
            return False
        return self.to_string() == value.to_string()

    def _data_str(self) -> str:
        """Return the data as a string."""
        if self.data_array is not None:
            if self.data_format and self.data_format.lower() == "base64":
                return base64.b64encode(self.data_array.tobytes()).decode("utf-8")
            return self.data_array.tobytes().decode("utf-8")
        if self.data_bytes is not None:
            if self.data_format and self.data_format.lower() == "base64":
                return base64.b64encode(self.data_bytes).decode("utf-8")
            return self.data_bytes.decode("utf-8")
        return ""


_TBinary = TypeVar("_TBinary", bound="BinaryContent")


@experimental_class
class BinaryContent(KernelContent):
    """This is a base class for different types of binary content.

    This can be created either the bytes data or a data uri, additionally it can have a uri.
    The uri is a reference to the source, and might or might not point to the same thing as the data.

    Ideally only subclasses of this class are used, like ImageContent.

    Methods:
        __str__: Returns the string representation of the content.

    Raises:
        ValidationError: If any arguments are malformed.

    """

    content_type: Literal[ContentTypes.BINARY_CONTENT] = Field(default=BINARY_CONTENT_TAG, init=False)
    uri: Url | str | None = None

    default_mime_type: ClassVar[str] = "text/plain"
    tag: ClassVar[str] = BINARY_CONTENT_TAG
    _data_uri: DataUri | None = PrivateAttr(default=None)

    def __init__(
        self,
        uri: Url | str | None = None,
        data_uri: DataUrl | str | None = None,
        data: str | bytes | ndarray | None = None,
        data_format: str | None = None,
        mime_type: str | None = None,
        **kwargs: Any,
    ):
        """Create a Binary Content object, either from a data_uri or data.

        Args:
            uri: The reference uri of the content.
            data_uri: The data uri of the content.
            data: The data of the content.
            data_format: The format of the data (e.g. base64).
            mime_type: The mime type of the content, not always relevant.
            kwargs: Any additional arguments:
                inner_content: The inner content of the response,
                    this should hold all the information from the response so even
                    when not creating a subclass a developer can leverage the full thing.
                ai_model_id: The id of the AI model that generated this response.
                metadata: Any metadata that should be attached to the response.
        """
        temp_data_uri: DataUri | None = None
        if data_uri:
            temp_data_uri = DataUri.from_data_uri(data_uri, self.default_mime_type)
            kwargs.setdefault("metadata", {})
            kwargs["metadata"].update(temp_data_uri.parameters)
        elif data is not None:
            match data:
                case bytes():
                    temp_data_uri = DataUri(
                        data_bytes=data, data_format=data_format, mime_type=mime_type or self.default_mime_type
                    )
                case ndarray():
                    temp_data_uri = DataUri(
                        data_array=data, data_format=data_format, mime_type=mime_type or self.default_mime_type
                    )
                case str():
                    temp_data_uri = DataUri(
                        data_str=data, data_format=data_format, mime_type=mime_type or self.default_mime_type
                    )

        if uri is not None:
            if isinstance(uri, str) and os.path.exists(uri):
                if os.path.isfile(uri):
                    uri = str(Path(uri))
                else:
                    raise ContentInitializationError("URI must be a file path, not a directory.")
            elif isinstance(uri, str):
                uri = Url(uri)

        super().__init__(uri=uri, **kwargs)
        self._data_uri = temp_data_uri

    @computed_field  # type: ignore
    @property
    def data_uri(self) -> str:
        """Get the data uri."""
        if self._data_uri:
            return self._data_uri.to_string(self.metadata)
        return ""

    @data_uri.setter
    def data_uri(self, value: str):
        """Set the data uri."""
        if not self._data_uri:
            self._data_uri = DataUri.from_data_uri(value, self.default_mime_type)
        else:
            self._data_uri.update_data(value)
        self.metadata.update(self._data_uri.parameters)

    @property
    def data(self) -> bytes:
        """Get the data."""
        if self._data_uri and self._data_uri.data_array:
            return self._data_uri.data_array.tobytes()
        if self._data_uri and self._data_uri.data_bytes:
            return self._data_uri.data_bytes
        return b""

    @data.setter
    def data(self, value: str | bytes | ndarray):
        """Set the data."""
        if self._data_uri:
            self._data_uri.update_data(value)
            return
        match value:
            case ndarray():
                self._data_uri = DataUri(data_array=value, mime_type=self.mime_type)
            case str():
                self._data_uri = DataUri(data_str=value, mime_type=self.mime_type)
            case bytes():
                self._data_uri = DataUri(data_bytes=value, mime_type=self.mime_type)
            case _:
                raise ContentException("Data must be a string, bytes, or numpy array.")

    @property
    def mime_type(self) -> str:
        """Get the mime type."""
        if self._data_uri and self._data_uri.mime_type:
            return self._data_uri.mime_type
        return self.default_mime_type

    @mime_type.setter
    def mime_type(self, value: str):
        """Set the mime type."""
        if self._data_uri:
            self._data_uri.mime_type = value

    def __str__(self) -> str:
        """Return the string representation of the content."""
        return self.data_uri if self._data_uri else str(self.uri)

    def to_element(self) -> Element:
        """Convert the instance to an Element."""
        element = Element(self.tag)
        if self._data_uri:
            element.text = self.data_uri
        if self.uri:
            element.set("uri", str(self.uri))
        return element

    @classmethod
    def from_element(cls: type[_TBinary], element: Element) -> _TBinary:
        """Create an instance from an Element."""
        if element.tag != cls.tag:
            raise ContentInitializationError(f"Element tag is not {cls.tag}")  # pragma: no cover

        if element.text:
            return cls(data_uri=element.text, uri=element.get("uri", None))

        return cls(uri=element.get("uri", None))

    def write_to_file(self, path: str | FilePath) -> None:
        """Write the data to a file."""
        if self._data_uri and self._data_uri.data_array is not None:
            self._data_uri.data_array.tofile(path)
            return
        with open(path, "wb") as file:
            file.write(self.data)

    def to_dict(self) -> dict[str, Any]:
        """Convert the instance to a dictionary."""
        return {"type": "binary", "binary": {"uri": str(self)}}


@experimental_class
class AudioContent(BinaryContent):
    """Audio Content class.

    This can be created either the bytes data or a data uri, additionally it can have a uri.
    The uri is a reference to the source, and might or might not point to the same thing as the data.

    Use the .from_audio_file method to create an instance from an audio file.
    This reads the file and guesses the mime_type.

    If both data_uri and data is provided, data will be used and a warning is logged.

    Args:
        uri (Url | None): The reference uri of the content.
        data_uri (DataUrl | None): The data uri of the content.
        data (str | bytes | None): The data of the content.
        data_format (str | None): The format of the data (e.g. base64).
        mime_type (str | None): The mime type of the audio, only used with data.
        kwargs (Any): Any additional arguments:
            inner_content (Any): The inner content of the response,
                this should hold all the information from the response so even
                when not creating a subclass a developer can leverage the full thing.
            ai_model_id (str | None): The id of the AI model that generated this response.
            metadata (dict[str, Any]): Any metadata that should be attached to the response.
    """

    content_type: Literal[ContentTypes.AUDIO_CONTENT] = Field(default=AUDIO_CONTENT_TAG, init=False)  # type: ignore
    tag: ClassVar[str] = AUDIO_CONTENT_TAG

    def __init__(
        self,
        uri: str | None = None,
        data_uri: str | None = None,
        data: str | bytes | ndarray | None = None,
        data_format: str | None = None,
        mime_type: str | None = None,
        **kwargs: Any,
    ):
        """Create an Audio Content object, either from a data_uri or data.

        Args:
            uri: The reference uri of the content.
            data_uri: The data uri of the content.
            data: The data of the content.
            data_format: The format of the data (e.g. base64).
            mime_type: The mime type of the audio, only used with data.
            kwargs: Any additional arguments:
                inner_content: The inner content of the response,
                    this should hold all the information from the response so even
                    when not creating a subclass a developer
                    can leverage the full thing.
                ai_model_id: The id of the AI model that generated this response.
                metadata: Any metadata that should be attached to the response.
        """
        super().__init__(
            uri=uri,
            data_uri=data_uri,
            data=data,
            data_format=data_format,
            mime_type=mime_type,
            **kwargs,
        )

    @classmethod
    def from_audio_file(cls: type[_TBinary], path: str) -> "_TBinary":
        """Create an instance from an audio file."""
        mime_type = mimetypes.guess_type(path)[0]
        with open(path, "rb") as audio_file:
            return cls(data=audio_file.read(), data_format="base64", mime_type=mime_type, uri=path)

    def to_dict(self) -> dict[str, Any]:
        """Convert the instance to a dictionary."""
        return {"type": "audio_url", "audio_url": {"uri": str(self)}}


class FunctionCallContent(KernelContent):
    """Class to hold a function call response."""

    content_type: Literal[ContentTypes.FUNCTION_CALL_CONTENT] = Field(FUNCTION_CALL_CONTENT_TAG, init=False)  # type: ignore
    tag: ClassVar[str] = FUNCTION_CALL_CONTENT_TAG
    id: str | None
    index: int | None = None
    name: str | None = None
    function_name: str
    plugin_name: str | None = None
    arguments: str | Mapping[str, Any] | None = None

    def __init__(
        self,
        content_type: Literal[ContentTypes.FUNCTION_CALL_CONTENT] = FUNCTION_CALL_CONTENT_TAG,  # type: ignore
        inner_content: Any | None = None,
        ai_model_id: str | None = None,
        id: str | None = None,
        index: int | None = None,
        name: str | None = None,
        function_name: str | None = None,
        plugin_name: str | None = None,
        arguments: str | Mapping[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create function call content.

        Args:
            content_type: The content type.
            inner_content (Any | None): The inner content.
            ai_model_id (str | None): The id of the AI model.
            id (str | None): The id of the function call.
            index (int | None): The index of the function call.
            name (str | None): The name of the function call.
                When not supplied function_name and plugin_name should be supplied.
            function_name (str | None): The function name.
                Not used when 'name' is supplied.
            plugin_name (str | None): The plugin name.
                Not used when 'name' is supplied.
            arguments (str | dict[str, Any] | None): The arguments of the function call.
            metadata (dict[str, Any] | None): The metadata of the function call.
            kwargs (Any): Additional arguments.
        """
        if function_name and plugin_name and not name:
            name = f"{plugin_name}{DEFAULT_FULLY_QUALIFIED_NAME_SEPARATOR}{function_name}"
        if name and not function_name and not plugin_name:
            if DEFAULT_FULLY_QUALIFIED_NAME_SEPARATOR in name:
                plugin_name, function_name = name.split(DEFAULT_FULLY_QUALIFIED_NAME_SEPARATOR, maxsplit=1)
            else:
                function_name = name
        args = {
            "content_type": content_type,
            "inner_content": inner_content,
            "ai_model_id": ai_model_id,
            "id": id,
            "index": index,
            "name": name,
            "function_name": function_name or "",
            "plugin_name": plugin_name,
            "arguments": arguments,
        }
        if metadata:
            args["metadata"] = metadata

        super().__init__(**args)

    def __str__(self) -> str:
        """Return the function call as a string."""
        if isinstance(self.arguments, dict):
            return f"{self.name}({json.dumps(self.arguments)})"
        return f"{self.name}({self.arguments})"

    def __add__(self, other: "FunctionCallContent | None") -> "FunctionCallContent":
        """Add two function calls together, combines the arguments, ignores the name.

        When both function calls have a dict as arguments, the arguments are merged,
        which means that the arguments of the second function call
        will overwrite the arguments of the first function call if the same key is present.

        When one of the two arguments are a dict and the other a string, we raise a ContentAdditionException.
        """
        if not other:
            return self
        if self.id and other.id and self.id != other.id:
            raise ContentAdditionException("Function calls have different ids.")
        if self.index != other.index:
            raise ContentAdditionException("Function calls have different indexes.")
        return FunctionCallContent(
            id=self.id or other.id,
            index=self.index or other.index,
            name=self.name or other.name,
            arguments=self.combine_arguments(self.arguments, other.arguments),
        )

    def combine_arguments(
        self, arg1: str | Mapping[str, Any] | None, arg2: str | Mapping[str, Any] | None
    ) -> str | Mapping[str, Any]:
        """Combine two arguments."""
        if isinstance(arg1, Mapping) and isinstance(arg2, Mapping):
            return {**arg1, **arg2}
        # when one of the two is a dict, and the other isn't, we raise.
        if isinstance(arg1, Mapping) or isinstance(arg2, Mapping):
            raise ContentAdditionException("Cannot combine a dict with a string.")
        if arg1 in EMPTY_VALUES and arg2 in EMPTY_VALUES:
            return "{}"
        if arg1 in EMPTY_VALUES:
            return arg2 or "{}"
        if arg2 in EMPTY_VALUES:
            return arg1 or "{}"
        return (arg1 or "") + (arg2 or "")

    def parse_arguments(self) -> Mapping[str, Any] | None:
        """Parse the arguments into a dictionary."""
        if not self.arguments:
            return None
        if isinstance(self.arguments, Mapping):
            return self.arguments
        try:
            return json.loads(self.arguments)
        except json.JSONDecodeError as exc:
            logger.debug("Function Call arguments are not valid JSON. Trying to preprocess.")
            try:
                # Python strings can be single quoted, but JSON strings should be double quoted.
                # JSON keys and values should be enclosed in double quotes.
                # Replace single quotes with double quotes, but not if it's an escaped single quote.
                return json.loads(re.sub(r"(?<!\\)'", '"', self.arguments).replace("\\'", "'"))
            except json.JSONDecodeError:
                raise FunctionCallInvalidArgumentsException(
                    "Function Call arguments are not valid JSON even after preprocessing."
                ) from exc

    def to_kernel_arguments(self) -> "KernelArguments":
        """Return the arguments as a KernelArguments instance."""
        from semantic_kernel.functions.kernel_arguments import KernelArguments

        args = self.parse_arguments()
        if not args:
            return KernelArguments()
        return KernelArguments(**args)

    @deprecated("The function_name and plugin_name properties should be used instead.")
    def split_name(self) -> list[str | None]:
        """Split the name into a plugin and function name."""
        if not self.function_name:
            raise FunctionCallInvalidNameException("Function name is not set.")
        return [self.plugin_name or "", self.function_name]

    @deprecated("The function_name and plugin_name properties should be used instead.")
    def split_name_dict(self) -> dict:
        """Split the name into a plugin and function name."""
        return {"plugin_name": self.plugin_name, "function_name": self.function_name}

    def custom_fully_qualified_name(self, separator: str) -> str:
        """Get the fully qualified name of the function with a custom separator.

        Args:
            separator (str): The custom separator.

        Returns:
            The fully qualified name of the function with a custom separator.
        """
        return f"{self.plugin_name}{separator}{self.function_name}" if self.plugin_name else self.function_name

    def to_element(self) -> Element:
        """Convert the function call to an Element."""
        element = Element(self.tag)
        if self.id:
            element.set("id", self.id)
        if self.name:
            element.set("name", self.name)
        if self.arguments:
            element.text = json.dumps(self.arguments) if isinstance(self.arguments, Mapping) else self.arguments
        return element

    @classmethod
    def from_element(cls: type[_T], element: Element) -> _T:
        """Create an instance from an Element."""
        if element.tag != cls.tag:
            raise ContentInitializationError(f"Element tag is not {cls.tag}")  # pragma: no cover

        return cls(name=element.get("name"), id=element.get("id"), arguments=element.text or "")

    def to_dict(self) -> dict[str, str | Any]:
        """Convert the instance to a dictionary."""
        args = json.dumps(self.arguments) if isinstance(self.arguments, Mapping) else self.arguments
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": args}}

    def __hash__(self) -> int:
        """Return the hash of the function call content."""
        args_hashable = frozenset(self.arguments.items()) if isinstance(self.arguments, Mapping) else None
        return hash((
            self.tag,
            self.id,
            self.index,
            self.name,
            self.function_name,
            self.plugin_name,
            args_hashable,
        ))


@experimental_class
class ImageContent(BinaryContent):
    """Image Content class.

    This can be created either the bytes data or a data uri, additionally it can have a uri.
    The uri is a reference to the source, and might or might not point to the same thing as the data.

    Use the .from_image_file method to create an instance from a image file.
    This reads the file and guesses the mime_type.

    If both data_uri and data is provided, data will be used and a warning is logged.

    Args:
        uri (Url | None): The reference uri of the content.
        data_uri (DataUrl | None): The data uri of the content.
        data (str | bytes | None): The data of the content.
        data_format (str | None): The format of the data (e.g. base64).
        mime_type (str | None): The mime type of the image, only used with data.
        kwargs (Any): Any additional arguments:
            inner_content (Any): The inner content of the response,
                this should hold all the information from the response so even
                when not creating a subclass a developer can leverage the full thing.
            ai_model_id (str | None): The id of the AI model that generated this response.
            metadata (dict[str, Any]): Any metadata that should be attached to the response.

    Methods:
        from_image_path: Create an instance from an image file.
        __str__: Returns the string representation of the image.

    Raises:
        ValidationError: If neither uri or data is provided.
    """

    content_type: Literal[ContentTypes.IMAGE_CONTENT] = Field(IMAGE_CONTENT_TAG, init=False)  # type: ignore
    tag: ClassVar[str] = IMAGE_CONTENT_TAG

    def __init__(
        self,
        uri: str | None = None,
        data_uri: str | None = None,
        data: str | bytes | ndarray | None = None,
        data_format: str | None = None,
        mime_type: str | None = None,
        **kwargs: Any,
    ):
        """Create an Image Content object, either from a data_uri or data.

        Args:
            uri: The reference uri of the content.
            data_uri: The data uri of the content.
            data: The data of the content.
            data_format: The format of the data (e.g. base64).
            mime_type: The mime type of the image, only used with data.
            kwargs: Any additional arguments:
            inner_content: The inner content of the response,
                this should hold all the information from the response so even
                when not creating a subclass a developer
                can leverage the full thing.
            ai_model_id: The id of the AI model that generated this response.
            metadata: Any metadata that should be attached to the response.
        """
        super().__init__(
            uri=uri,
            data_uri=data_uri,
            data=data,
            data_format=data_format,
            mime_type=mime_type,
            **kwargs,
        )

    @classmethod
    @deprecated("The `from_image_path` method is deprecated; use `from_image_file` instead.", category=None)
    def from_image_path(cls: type[_T], image_path: str) -> _T:
        """Create an instance from an image file."""
        return cls.from_image_file(image_path)

    @classmethod
    def from_image_file(cls: type[_T], path: str) -> _T:
        """Create an instance from an image file."""
        mime_type = mimetypes.guess_type(path)[0]
        with open(path, "rb") as image_file:
            return cls(data=image_file.read(), data_format="base64", mime_type=mime_type, uri=path)

    def to_dict(self) -> dict[str, Any]:
        """Convert the instance to a dictionary."""
        return {"type": "image_url", "image_url": {"url": str(self)}}


class StreamingContentMixin(KernelBaseModel, ABC):
    """Mixin class for all streaming kernel contents."""

    choice_index: int

    @abstractmethod
    def __bytes__(self) -> bytes:
        """Return the content of the response encoded in the encoding."""
        pass

    @abstractmethod
    def __add__(self, other: Any) -> Self:
        """Combine two streaming contents together."""
        pass

    def _merge_items_lists(self, other_items: list[Any]) -> list[Any]:
        """Create a new list with the items of the current instance and the given list."""
        if not hasattr(self, "items"):
            raise ContentAdditionException(f"Cannot merge items for this instance of type: {type(self)}")

        # Create a copy of the items list to avoid modifying the original instance.
        # Note that the items are not copied, only the list is.
        new_items_list = self.items.copy()

        if new_items_list or other_items:
            for other_item in other_items:
                added = False
                for id, item in enumerate(new_items_list):
                    if type(item) is type(other_item) and hasattr(item, "__add__"):
                        try:
                            new_item = item + other_item  # type: ignore
                            new_items_list[id] = new_item
                            added = True
                        except (ValueError, ContentAdditionException) as ex:
                            logger.debug(f"Could not add item {other_item} to {item}.", exc_info=ex)
                            continue
                if not added:
                    logger.debug(f"Could not add item {other_item} to any item in the list. Adding it as a new item.")
                    new_items_list.append(other_item)

        return new_items_list

    def _merge_inner_contents(self, other_inner_content: Any | list[Any]) -> list[Any]:
        """Create a new list with the inner content of the current instance and the given one."""
        if not hasattr(self, "inner_content"):
            raise ContentAdditionException(f"Cannot merge inner content for this instance of type: {type(self)}")

        # Create a copy of the inner content list to avoid modifying the original instance.
        # Note that the inner content is not copied, only the list is.
        # If the inner content is not a list, it is converted to a list.
        if isinstance(self.inner_content, list):
            new_inner_contents_list = self.inner_content.copy()
        else:
            new_inner_contents_list = [self.inner_content]

        other_inner_content = (
            other_inner_content
            if isinstance(other_inner_content, list)
            else [other_inner_content]
            if other_inner_content
            else []
        )

        new_inner_contents_list.extend(other_inner_content)

        return new_inner_contents_list


class TextContent(KernelContent):
    """This represents text response content.

    Args:
        inner_content: Any - The inner content of the response,
            this should hold all the information from the response so even
            when not creating a subclass a developer can leverage the full thing.
        ai_model_id: str | None - The id of the AI model that generated this response.
        metadata: dict[str, Any] - Any metadata that should be attached to the response.
        text: str | None - The text of the response.
        encoding: str | None - The encoding of the text.

    Methods:
        __str__: Returns the text of the response.
    """

    content_type: Literal[ContentTypes.TEXT_CONTENT] = Field(TEXT_CONTENT_TAG, init=False)  # type: ignore
    tag: ClassVar[str] = TEXT_CONTENT_TAG
    text: str
    encoding: str | None = None

    def __str__(self) -> str:
        """Return the text of the response."""
        return self.text

    def to_element(self) -> Element:
        """Convert the instance to an Element."""
        element = Element(self.tag)
        element.text = self.text
        if self.encoding:
            element.set("encoding", self.encoding)
        return element

    @classmethod
    def from_element(cls: type[_T], element: Element) -> _T:
        """Create an instance from an Element."""
        if element.tag != cls.tag:
            raise ContentInitializationError(f"Element tag is not {cls.tag}")  # pragma: no cover

        return cls(text=unescape(element.text) if element.text else "", encoding=element.get("encoding", None))

    def to_dict(self) -> dict[str, str]:
        """Convert the instance to a dictionary."""
        return {"type": "text", "text": self.text}

    def __hash__(self) -> int:
        """Return the hash of the text content."""
        return hash((self.tag, self.text, self.encoding))


class StreamingTextContent(StreamingContentMixin, TextContent):
    """This represents streaming text response content.

    Args:
        choice_index: int - The index of the choice that generated this response.
        inner_content: Optional[Any] - The inner content of the response,
            this should hold all the information from the response so even
            when not creating a subclass a developer can leverage the full thing.
        ai_model_id: Optional[str] - The id of the AI model that generated this response.
        metadata: Dict[str, Any] - Any metadata that should be attached to the response.
        text: Optional[str] - The text of the response.
        encoding: Optional[str] - The encoding of the text.

    Methods:
        __str__: Returns the text of the response.
        __bytes__: Returns the content of the response encoded in the encoding.
        __add__: Combines two StreamingTextContent instances.
    """

    def __bytes__(self) -> bytes:
        """Return the content of the response encoded in the encoding."""
        return self.text.encode(self.encoding if self.encoding else "utf-8") if self.text else b""

    def __add__(self, other: TextContent) -> "StreamingTextContent":
        """When combining two StreamingTextContent instances, the text fields are combined.

        The addition should follow these rules:
            1. The inner_content of the two will be combined. If they are not lists, they will be converted to lists.
            2. ai_model_id should be the same.
            3. encoding should be the same.
            4. choice_index should be the same.
            5. Metadata will be combined.
        """
        if isinstance(other, StreamingTextContent) and self.choice_index != other.choice_index:
            raise ContentAdditionException("Cannot add StreamingTextContent with different choice_index")
        if self.ai_model_id != other.ai_model_id:
            raise ContentAdditionException("Cannot add StreamingTextContent from different ai_model_id")
        if self.encoding != other.encoding:
            raise ContentAdditionException("Cannot add StreamingTextContent with different encoding")

        return StreamingTextContent(
            choice_index=self.choice_index,
            inner_content=self._merge_inner_contents(other.inner_content),
            ai_model_id=self.ai_model_id,
            metadata=self.metadata,
            text=(self.text or "") + (other.text or ""),
            encoding=self.encoding,
        )


def make_hashable(input: Any, visited=None) -> Any:
    """Recursively convert unhashable types to hashable equivalents.

    Args:
        input: The input to convert to a hashable type.
        visited: A dictionary of visited objects to prevent infinite recursion.

    Returns:
        Any: The input converted to a hashable type.
    """
    if visited is None:
        visited = {}

    # If we've seen this object before, return the stored placeholder or final result
    unique_obj_id = id(input)
    if unique_obj_id in visited:
        return visited[unique_obj_id]

    # Handle Pydantic models by manually traversing fields
    if isinstance(input, BaseModel):
        visited[unique_obj_id] = None
        data = {}
        for field_name in input.model_fields:
            value = getattr(input, field_name)
            data[field_name] = make_hashable(value, visited)
        result = tuple(sorted(data.items()))
        visited[unique_obj_id] = result
        return result

    # Convert dictionaries
    if isinstance(input, dict):
        visited[unique_obj_id] = None
        items = tuple(sorted((k, make_hashable(v, visited)) for k, v in input.items()))
        visited[unique_obj_id] = items
        return items

    # Convert lists, sets, and tuples to tuples
    if isinstance(input, (list, set, tuple)):
        visited[unique_obj_id] = None
        items = tuple(make_hashable(item, visited) for item in input)
        visited[unique_obj_id] = items
        return items

    # If it's already something hashable, just return it
    return input


class FunctionResultContent(KernelContent):
    """This class represents function result content."""

    content_type: Literal[ContentTypes.FUNCTION_RESULT_CONTENT] = Field(FUNCTION_RESULT_CONTENT_TAG, init=False)  # type: ignore
    tag: ClassVar[str] = FUNCTION_RESULT_CONTENT_TAG
    id: str
    result: Any
    name: str | None = None
    function_name: str
    plugin_name: str | None = None
    encoding: str | None = None

    def __init__(
        self,
        content_type: Literal[ContentTypes.FUNCTION_RESULT_CONTENT] = FUNCTION_RESULT_CONTENT_TAG,  # type: ignore
        inner_content: Any | None = None,
        ai_model_id: str | None = None,
        id: str | None = None,
        name: str | None = None,
        function_name: str | None = None,
        plugin_name: str | None = None,
        result: Any | None = None,
        encoding: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create function result content.

        Args:
            content_type: The content type.
            inner_content (Any | None): The inner content.
            ai_model_id (str | None): The id of the AI model.
            id (str | None): The id of the function call that the result relates to.
            name (str | None): The name of the function.
                When not supplied function_name and plugin_name should be supplied.
            function_name (str | None): The function name.
                Not used when 'name' is supplied.
            plugin_name (str | None): The plugin name.
                Not used when 'name' is supplied.
            result (Any | None): The result of the function.
            encoding (str | None): The encoding of the result.
            metadata (dict[str, Any] | None): The metadata of the function call.
            kwargs (Any): Additional arguments.
        """
        if function_name and plugin_name and not name:
            name = f"{plugin_name}{DEFAULT_FULLY_QUALIFIED_NAME_SEPARATOR}{function_name}"
        if name and not function_name and not plugin_name:
            if DEFAULT_FULLY_QUALIFIED_NAME_SEPARATOR in name:
                plugin_name, function_name = name.split(DEFAULT_FULLY_QUALIFIED_NAME_SEPARATOR, maxsplit=1)
            else:
                function_name = name
        args = {
            "content_type": content_type,
            "inner_content": inner_content,
            "ai_model_id": ai_model_id,
            "id": id,
            "name": name,
            "function_name": function_name or "",
            "plugin_name": plugin_name,
            "result": result,
            "encoding": encoding,
        }
        if metadata:
            args["metadata"] = metadata

        super().__init__(**args)

    def __str__(self) -> str:
        """Return the text of the response."""
        return str(self.result)

    def to_element(self) -> Element:
        """Convert the instance to an Element."""
        element = Element(self.tag)
        element.set("id", self.id)
        if self.name:
            element.set("name", self.name)
        element.text = str(self.result)
        return element

    @classmethod
    def from_element(cls: type[_T], element: Element) -> _T:
        """Create an instance from an Element."""
        if element.tag != cls.tag:
            raise ContentInitializationError(f"Element tag is not {cls.tag}")  # pragma: no cover
        return cls(id=element.get("id", ""), result=element.text, name=element.get("name", None))

    @classmethod
    def from_function_call_content_and_result(
        cls: type[_T],
        function_call_content: "FunctionCallContent",
        result: "FunctionResult | TextContent | ChatMessageContent | Any",
        metadata: dict[str, Any] = {},
    ) -> _T:
        """Create an instance from a FunctionCallContent and a result."""
        from semantic_kernel.contents import ChatMessageContent
        from semantic_kernel.functions.function_result import FunctionResult

        metadata.update(function_call_content.metadata or {})
        metadata.update(getattr(result, "metadata", {}))
        inner_content = result
        if isinstance(result, FunctionResult):
            result = result.value
        if isinstance(result, TextContent):
            res = result.text
        elif isinstance(result, ChatMessageContent):
            if isinstance(result.items[0], TextContent):
                res = result.items[0].text
            elif isinstance(result.items[0], ImageContent):
                res = result.items[0].data_uri
            elif isinstance(result.items[0], FunctionResultContent):
                res = result.items[0].result
            res = str(result)
        else:
            res = result
        return cls(
            id=function_call_content.id or "unknown",
            inner_content=inner_content,
            result=res,
            function_name=function_call_content.function_name,
            plugin_name=function_call_content.plugin_name,
            ai_model_id=function_call_content.ai_model_id,
            metadata=metadata,
        )

    def to_chat_message_content(self) -> "ChatMessageContent":
        """Convert the instance to a ChatMessageContent."""
        from semantic_kernel.contents import ChatMessageContent

        return ChatMessageContent(role=AuthorRole.TOOL, items=[self])

    def to_streaming_chat_message_content(self) -> "StreamingChatMessageContent":
        """Convert the instance to a StreamingChatMessageContent."""
        return StreamingChatMessageContent(role=AuthorRole.TOOL, choice_index=0, items=[self])

    def to_dict(self) -> dict[str, str]:
        """Convert the instance to a dictionary."""
        return {
            "tool_call_id": self.id,
            "content": self.result,
        }

    @deprecated("The function_name and plugin_name attributes should be used instead.")
    def split_name(self) -> list[str]:
        """Split the name into a plugin and function name."""
        return [self.plugin_name or "", self.function_name]

    def custom_fully_qualified_name(self, separator: str) -> str:
        """Get the fully qualified name of the function with a custom separator.

        Args:
            separator (str): The custom separator.

        Returns:
            The fully qualified name of the function with a custom separator.
        """
        return f"{self.plugin_name}{separator}{self.function_name}" if self.plugin_name else self.function_name

    @field_serializer("result")
    def serialize_result(self, value: Any) -> str:
        """Serialize the result."""
        return str(value)

    def __hash__(self) -> int:
        """Return the hash of the function result content."""
        hashable_result = make_hashable(self.result)
        return hash((
            self.tag,
            self.id,
            hashable_result,
            self.name,
            self.function_name,
            self.plugin_name,
            self.encoding,
        ))


@experimental_class
class FileReferenceContent(KernelContent):
    """File reference content."""

    content_type: Literal[ContentTypes.FILE_REFERENCE_CONTENT] = Field(FILE_REFERENCE_CONTENT_TAG, init=False)  # type: ignore
    tag: ClassVar[str] = FILE_REFERENCE_CONTENT_TAG
    file_id: str | None = None
    tools: list[Any] = Field(default_factory=list)
    data_source: Any | None = None

    def __str__(self) -> str:
        """Return the string representation of the file reference content."""
        return f"FileReferenceContent(file_id={self.file_id})"

    def to_element(self) -> Element:
        """Convert the file reference content to an Element."""
        element = Element(self.tag)
        if self.file_id:
            element.set("file_id", self.file_id)
        return element

    @classmethod
    def from_element(cls: type[_T], element: Element) -> _T:
        """Create an instance from an Element."""
        return cls(
            file_id=element.get("file_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the instance to a dictionary."""
        return {
            "file_id": self.file_id,
        }


_TStreamingAnnotationContent = TypeVar("_TStreamingAnnotationContent", bound="StreamingAnnotationContent")


@experimental_class
class StreamingAnnotationContent(KernelContent):
    """Streaming Annotation content."""

    content_type: Literal[ContentTypes.STREAMING_ANNOTATION_CONTENT] = Field(
        STREAMING_ANNOTATION_CONTENT_TAG,  # type: ignore
        init=False,
    )
    tag: ClassVar[str] = STREAMING_ANNOTATION_CONTENT_TAG
    file_id: str | None = None
    quote: str | None = None
    start_index: int | None = None
    end_index: int | None = None

    def __str__(self) -> str:
        """Return the string representation of the annotation content."""
        return f"StreamingAnnotationContent(file_id={self.file_id}, quote={self.quote}, start_index={self.start_index}, end_index={self.end_index})"  # noqa: E501

    def to_element(self) -> Element:
        """Convert the annotation content to an Element."""
        element = Element(self.tag)
        if self.file_id:
            element.set("file_id", self.file_id)
        if self.quote:
            element.set("quote", self.quote)
        if self.start_index is not None:
            element.set("start_index", str(self.start_index))
        if self.end_index is not None:
            element.set("end_index", str(self.end_index))
        return element

    @classmethod
    def from_element(cls: type[_TStreamingAnnotationContent], element: Element) -> _TStreamingAnnotationContent:
        """Create an instance from an Element."""
        return cls(
            file_id=element.get("file_id"),
            quote=element.get("quote"),
            start_index=int(element.get("start_index")) if element.get("start_index") else None,  # type: ignore
            end_index=int(element.get("end_index")) if element.get("end_index") else None,  # type: ignore
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the instance to a dictionary."""
        return {
            "type": "text",
            "text": f"{self.file_id} {self.quote} (Start Index={self.start_index}->End Index={self.end_index})",
        }


_TStreamingFileReferenceContent = TypeVar("_TStreamingFileReferenceContent", bound="StreamingFileReferenceContent")


@experimental_class
class StreamingFileReferenceContent(KernelContent):
    """Streaming File reference content."""

    content_type: Literal[ContentTypes.STREAMING_FILE_REFERENCE_CONTENT] = Field(
        STREAMING_FILE_REFERENCE_CONTENT_TAG,  # type: ignore
        init=False,
    )
    tag: ClassVar[str] = STREAMING_FILE_REFERENCE_CONTENT_TAG
    file_id: str | None = None
    tools: list[Any] = Field(default_factory=list)
    data_source: Any | None = None

    def __str__(self) -> str:
        """Return the string representation of the file reference content."""
        return f"StreamingFileReferenceContent(file_id={self.file_id})"

    def to_element(self) -> Element:
        """Convert the file reference content to an Element."""
        element = Element(self.tag)
        if self.file_id:
            element.set("file_id", self.file_id)
        return element

    @classmethod
    def from_element(cls: type[_TStreamingFileReferenceContent], element: Element) -> _TStreamingFileReferenceContent:
        """Create an instance from an Element."""
        return cls(
            file_id=element.get("file_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the instance to a dictionary."""
        return {
            "file_id": self.file_id,
        }


TAG_CONTENT_MAP = {
    ANNOTATION_CONTENT_TAG: AnnotationContent,
    TEXT_CONTENT_TAG: TextContent,
    FILE_REFERENCE_CONTENT_TAG: FileReferenceContent,
    FUNCTION_CALL_CONTENT_TAG: FunctionCallContent,
    FUNCTION_RESULT_CONTENT_TAG: FunctionResultContent,
    IMAGE_CONTENT_TAG: ImageContent,
    STREAMING_FILE_REFERENCE_CONTENT_TAG: StreamingFileReferenceContent,
    STREAMING_ANNOTATION_CONTENT_TAG: StreamingAnnotationContent,
}

CMC_ITEM_TYPES = Annotated[
    AnnotationContent
    | BinaryContent
    | ImageContent
    | TextContent
    | FunctionResultContent
    | FunctionCallContent
    | FileReferenceContent
    | StreamingAnnotationContent
    | StreamingFileReferenceContent,
    Field(discriminator=DISCRIMINATOR_FIELD),
]

STREAMING_CMC_ITEM_TYPES = Annotated[
    BinaryContent
    | ImageContent
    | StreamingTextContent
    | FunctionCallContent
    | FunctionResultContent
    | StreamingFileReferenceContent
    | StreamingAnnotationContent,
    Field(discriminator=DISCRIMINATOR_FIELD),
]


class ChatMessageContent(KernelContent):
    """This is the class for chat message response content.

    All Chat Completion Services should return an instance of this class as response.
    Or they can implement their own subclass of this class and return an instance.

    Args:
        inner_content: Optional[Any] - The inner content of the response,
            this should hold all the information from the response so even
            when not creating a subclass a developer can leverage the full thing.
        ai_model_id: Optional[str] - The id of the AI model that generated this response.
        metadata: Dict[str, Any] - Any metadata that should be attached to the response.
        role: ChatRole - The role of the chat message.
        content: Optional[str] - The text of the response.
        encoding: Optional[str] - The encoding of the text.

    Methods:
        __str__: Returns the content of the response.
    """

    content_type: Literal[ContentTypes.CHAT_MESSAGE_CONTENT] = Field(default=CHAT_MESSAGE_CONTENT_TAG, init=False)  # type: ignore
    tag: ClassVar[str] = CHAT_MESSAGE_CONTENT_TAG
    role: AuthorRole
    name: str | None = None
    items: list[CMC_ITEM_TYPES] = Field(default_factory=list)
    encoding: str | None = None
    finish_reason: FinishReason | None = None

    @overload
    def __init__(
        self,
        role: AuthorRole,
        items: list[CMC_ITEM_TYPES],
        name: str | None = None,
        inner_content: Any | None = None,
        encoding: str | None = None,
        finish_reason: FinishReason | None = None,
        ai_model_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __init__(
        self,
        role: AuthorRole,
        content: str,
        name: str | None = None,
        inner_content: Any | None = None,
        encoding: str | None = None,
        finish_reason: FinishReason | None = None,
        ai_model_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...

    def __init__(  # type: ignore
        self,
        role: AuthorRole,
        items: list[CMC_ITEM_TYPES] | None = None,
        content: str | None = None,
        inner_content: Any | None = None,
        name: str | None = None,
        encoding: str | None = None,
        finish_reason: FinishReason | None = None,
        ai_model_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Create a ChatMessageContent instance.

        Args:
            role: AuthorRole - The role of the chat message.
            items: list[TextContent, StreamingTextContent, FunctionCallContent, FunctionResultContent, ImageContent]
                 - The content.
            content: str - The text of the response.
            inner_content: Optional[Any] - The inner content of the response,
                this should hold all the information from the response so even
                when not creating a subclass a developer can leverage the full thing.
            name: Optional[str] - The name of the response.
            encoding: Optional[str] - The encoding of the text.
            finish_reason: Optional[FinishReason] - The reason the response was finished.
            ai_model_id: Optional[str] - The id of the AI model that generated this response.
            metadata: Dict[str, Any] - Any metadata that should be attached to the response.
            **kwargs: Any - Any additional fields to set on the instance.
        """
        kwargs["role"] = role
        if encoding:
            kwargs["encoding"] = encoding
        if finish_reason:
            kwargs["finish_reason"] = finish_reason
        if name:
            kwargs["name"] = name
        if content:
            item = TextContent(
                ai_model_id=ai_model_id,
                inner_content=inner_content,
                metadata=metadata or {},
                text=content,
                encoding=encoding,
            )
            if items:
                items.append(item)
            else:
                items = [item]
        if items:
            kwargs["items"] = items
        if inner_content:
            kwargs["inner_content"] = inner_content
        if metadata:
            kwargs["metadata"] = metadata
        if ai_model_id:
            kwargs["ai_model_id"] = ai_model_id
        super().__init__(
            **kwargs,
        )

    @property
    def content(self) -> str:
        """Get the content of the response, will find the first TextContent's text."""
        for item in self.items:
            if isinstance(item, TextContent):
                return item.text
        return ""

    @content.setter
    def content(self, value: str):
        """Set the content of the response."""
        if not value:
            logger.warning(
                "Setting empty content on ChatMessageContent does not work, "
                "you can do this through the underlying items if needed, ignoring."
            )
            return
        for item in self.items:
            if isinstance(item, TextContent):
                item.text = value
                item.encoding = self.encoding
                return
        self.items.append(
            TextContent(
                ai_model_id=self.ai_model_id,
                inner_content=self.inner_content,
                metadata=self.metadata,
                text=value,
                encoding=self.encoding,
            )
        )

    def __str__(self) -> str:
        """Get the content of the response as a string."""
        return self.content or ""

    def to_element(self) -> "Element":
        """Convert the ChatMessageContent to an XML Element.

        Args:
            root_key: str - The key to use for the root of the XML Element.

        Returns:
            Element - The XML Element representing the ChatMessageContent.
        """
        root = Element(self.tag)
        for field in self.model_fields_set:
            if field not in ["role", "name", "encoding", "finish_reason", "ai_model_id"]:
                continue
            value = getattr(self, field)
            if isinstance(value, Enum):
                value = value.value
            root.set(field, value)
        for index, item in enumerate(self.items):
            root.insert(index, item.to_element())
        return root

    @classmethod
    def from_element(cls, element: Element) -> "ChatMessageContent":
        """Create a new instance of ChatMessageContent from an XML element.

        Args:
            element: Element - The XML Element to create the ChatMessageContent from.

        Returns:
            ChatMessageContent - The new instance of ChatMessageContent or a subclass.
        """
        if element.tag != cls.tag:
            raise ContentInitializationError(f"Element tag is not {cls.tag}")  # pragma: no cover
        kwargs: dict[str, Any] = {key: value for key, value in element.items()}
        items: list[KernelContent] = []
        if element.text:
            items.append(TextContent(text=unescape(element.text)))
        for child in element:
            if child.tag not in TAG_CONTENT_MAP:
                logger.warning('Unknown tag "%s" in ChatMessageContent, treating as text', child.tag)
                text = ElementTree.tostring(child, encoding="unicode", short_empty_elements=False)
                items.append(TextContent(text=unescape(text) or ""))
            else:
                items.append(TAG_CONTENT_MAP[child.tag].from_element(child))  # type: ignore
        if len(items) == 1 and isinstance(items[0], TextContent):
            kwargs["content"] = items[0].text
        elif all(isinstance(item, TextContent) for item in items):
            kwargs["content"] = "".join(item.text for item in items)  # type: ignore
        else:
            kwargs["items"] = items
        if "choice_index" in kwargs and cls is ChatMessageContent:
            logger.info(
                "Seems like you are trying to create a StreamingChatMessageContent, "
                "use StreamingChatMessageContent.from_element instead, ignoring that field "
                "and creating a ChatMessageContent instance."
            )
            kwargs.pop("choice_index")
        return cls(**kwargs)

    def to_prompt(self) -> str:
        """Convert the ChatMessageContent to a prompt.

        Returns:
            str - The prompt from the ChatMessageContent.
        """
        root = self.to_element()
        return ElementTree.tostring(root, encoding=self.encoding or "unicode", short_empty_elements=False)

    def to_dict(self, role_key: str = "role", content_key: str = "content") -> dict[str, Any]:
        """Serialize the ChatMessageContent to a dictionary.

        Returns:
            dict - The dictionary representing the ChatMessageContent.
        """
        ret: dict[str, Any] = {
            role_key: self.role.value,
        }
        if self.role == AuthorRole.ASSISTANT and any(isinstance(item, FunctionCallContent) for item in self.items):
            ret["tool_calls"] = [item.to_dict() for item in self.items if isinstance(item, FunctionCallContent)]
        else:
            ret[content_key] = self._parse_items()
        if self.role == AuthorRole.TOOL:
            assert isinstance(self.items[0], FunctionResultContent)  # nosec
            ret["tool_call_id"] = self.items[0].id or ""
        if self.role != AuthorRole.TOOL and self.name:
            ret["name"] = self.name
        return ret

    def _parse_items(self) -> str | list[dict[str, Any]]:
        """Parse the items of the ChatMessageContent.

        Returns:
            str | list of dicts - The parsed items.
        """
        if len(self.items) == 1 and isinstance(self.items[0], TextContent):
            return self.items[0].text
        if len(self.items) == 1 and isinstance(self.items[0], FunctionResultContent):
            return str(self.items[0].result)
        return [item.to_dict() for item in self.items]

    def __hash__(self) -> int:
        """Return the hash of the chat message content."""
        hashable_items = [make_hashable(item) for item in self.items] if self.items else []
        return hash((self.tag, self.role, self.content, self.encoding, self.finish_reason, *hashable_items))


class StreamingChatMessageContent(ChatMessageContent, StreamingContentMixin):
    """This is the class for streaming chat message response content.

    All Chat Completion Services should return an instance of this class as streaming response,
    where each part of the response as it is streamed is converted to an instance of this class,
    the end-user will have to either do something directly or gather them and combine them into a
    new instance. A service can implement their own subclass of this class and return instances of that.

    Args:
        choice_index: int - The index of the choice that generated this response.
        inner_content: Optional[Any] - The inner content of the response,
            this should hold all the information from the response so even
            when not creating a subclass a developer can leverage the full thing.
        ai_model_id: Optional[str] - The id of the AI model that generated this response.
        metadata: Dict[str, Any] - Any metadata that should be attached to the response.
        role: Optional[ChatRole] - The role of the chat message, defaults to ASSISTANT.
        content: Optional[str] - The text of the response.
        encoding: Optional[str] - The encoding of the text.

    Methods:
        __str__: Returns the content of the response.
        __bytes__: Returns the content of the response encoded in the encoding.
        __add__: Combines two StreamingChatMessageContent instances.
    """

    function_invoke_attempt: int | None = Field(
        default=0,
        description="Tracks the current attempt count for automatically invoking functions. "
        "This value increments with each subsequent automatic invocation attempt.",
    )

    @overload
    def __init__(
        self,
        role: AuthorRole,
        items: list[STREAMING_CMC_ITEM_TYPES],
        choice_index: int,
        name: str | None = None,
        inner_content: Any | None = None,
        encoding: str | None = None,
        finish_reason: FinishReason | None = None,
        ai_model_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        function_invoke_attempt: int | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        role: AuthorRole,
        content: str,
        choice_index: int,
        name: str | None = None,
        inner_content: Any | None = None,
        encoding: str | None = None,
        finish_reason: FinishReason | None = None,
        ai_model_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        function_invoke_attempt: int | None = None,
    ) -> None: ...

    def __init__(  # type: ignore
        self,
        role: AuthorRole,
        choice_index: int,
        items: list[STREAMING_CMC_ITEM_TYPES] | None = None,
        content: str | None = None,
        inner_content: Any | None = None,
        name: str | None = None,
        encoding: str | None = None,
        finish_reason: FinishReason | None = None,
        ai_model_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        function_invoke_attempt: int | None = None,
    ):
        """Create a new instance of StreamingChatMessageContent.

        Args:
            role: The role of the chat message.
            choice_index: The index of the choice that generated this response.
            items: The content.
            content: The text of the response.
            inner_content: The inner content of the response,
                this should hold all the information from the response so even
                when not creating a subclass a developer can leverage the full thing.
            name: The name of the response.
            encoding: The encoding of the text.
            finish_reason: The reason the response was finished.
            metadata: Any metadata that should be attached to the response.
            ai_model_id: The id of the AI model that generated this response.
            function_invoke_attempt: Tracks the current attempt count for automatically
                invoking functions. This value increments with each subsequent automatic invocation attempt.
        """
        kwargs: dict[str, Any] = {
            "role": role,
            "choice_index": choice_index,
            "function_invoke_attempt": function_invoke_attempt,
        }
        if encoding:
            kwargs["encoding"] = encoding
        if finish_reason:
            kwargs["finish_reason"] = finish_reason
        if name:
            kwargs["name"] = name
        if content:
            item = StreamingTextContent(
                choice_index=choice_index,
                ai_model_id=ai_model_id,
                inner_content=inner_content,
                metadata=metadata or {},
                text=content,
                encoding=encoding,
            )
            if items:
                items.append(item)
            else:
                items = [item]
        if items:
            kwargs["items"] = items
        if inner_content:
            kwargs["inner_content"] = inner_content
        if metadata:
            kwargs["metadata"] = metadata
        if ai_model_id:
            kwargs["ai_model_id"] = ai_model_id
        super().__init__(
            **kwargs,
        )

    def __bytes__(self) -> bytes:
        """Return the content of the response encoded in the encoding."""
        return self.content.encode(self.encoding if self.encoding else "utf-8") if self.content else b""

    def __add__(self, other: "StreamingChatMessageContent") -> "StreamingChatMessageContent":
        """When combining two StreamingChatMessageContent instances, the content fields are combined.

        The addition should follow these rules:
            1. The inner_content of the two will be combined. If they are not lists, they will be converted to lists.
            2. ai_model_id should be the same.
            3. encoding should be the same.
            4. role should be the same.
            5. choice_index should be the same.
            6. Metadata will be combined
        """
        if not isinstance(other, StreamingChatMessageContent):
            raise ContentAdditionException(
                f"Cannot add other type to StreamingChatMessageContent, type supplied: {type(other)}"
            )
        if self.choice_index != other.choice_index:
            raise ContentAdditionException("Cannot add StreamingChatMessageContent with different choice_index")
        if self.ai_model_id != other.ai_model_id:
            raise ContentAdditionException("Cannot add StreamingChatMessageContent from different ai_model_id")
        if self.encoding != other.encoding:
            raise ContentAdditionException("Cannot add StreamingChatMessageContent with different encoding")
        if self.role and other.role and self.role != other.role:
            raise ContentAdditionException("Cannot add StreamingChatMessageContent with different role")

        return StreamingChatMessageContent(
            role=self.role,
            items=self._merge_items_lists(other.items),
            choice_index=self.choice_index,
            inner_content=self._merge_inner_contents(other.inner_content),
            ai_model_id=self.ai_model_id,
            metadata=self.metadata | other.metadata,
            encoding=self.encoding,
            finish_reason=self.finish_reason or other.finish_reason,
            function_invoke_attempt=self.function_invoke_attempt,
        )

    def to_element(self) -> "Element":
        """Convert the StreamingChatMessageContent to an XML Element.

        Args:
            root_key: str - The key to use for the root of the XML Element.

        Returns:
            Element - The XML Element representing the StreamingChatMessageContent.
        """
        root = Element(self.tag)
        for field in self.model_fields_set:
            if field not in ["role", "name", "encoding", "finish_reason", "ai_model_id", "choice_index"]:
                continue
            value = getattr(self, field)
            if isinstance(value, Enum):
                value = value.value
            if isinstance(value, int):
                value = str(value)
            root.set(field, value)
        for index, item in enumerate(self.items):
            root.insert(index, item.to_element())
        return root

    def __hash__(self) -> int:
        """Return the hash of the streaming chat message content."""
        hashable_items = [make_hashable(item) for item in self.items] if self.items else []
        return hash((
            self.tag,
            self.role,
            self.content,
            self.encoding,
            self.finish_reason,
            self.choice_index,
            self.function_invoke_attempt,
            *hashable_items,
        ))


class ChatHistory(KernelBaseModel):
    """This class holds the history of chat messages from a chat conversation.

    Note: the system_message is added to the messages as a ChatMessageContent instance with role=AuthorRole.SYSTEM,
    but updating it will not update the messages list.

    Args:
        messages: The messages to add to the chat history.
        system_message: A system message to add to the chat history, optional.
            if passed, it is added to the messages
            as a ChatMessageContent instance with role=AuthorRole.SYSTEM
            before any other messages.
    """

    messages: list[ChatMessageContent] = Field(default_factory=list, kw_only=False)
    system_message: str | None = Field(default=None, kw_only=False, repr=False)

    @model_validator(mode="before")
    @classmethod
    def _parse_system_message(cls, data: Any) -> Any:
        """Parse the system_message and add it to the messages."""
        if isinstance(data, dict) and (system_message := data.pop("system_message", None)):
            msg = ChatMessageContent(role=AuthorRole.SYSTEM, content=system_message)
            if "messages" in data:
                data["messages"] = [msg] + data["messages"]
            else:
                data["messages"] = [msg]
        return data

    @field_validator("messages", mode="before")
    @classmethod
    def _validate_messages(cls, messages: list[ChatMessageContent]) -> list[ChatMessageContent]:
        if not messages:
            return messages
        out_msgs: list[ChatMessageContent] = []
        for message in messages:
            if isinstance(message, dict):
                out_msgs.append(ChatMessageContent.model_validate(message))
            else:
                out_msgs.append(message)
        return out_msgs

    @singledispatchmethod
    def add_system_message(self, content: str | list[KernelContent], **kwargs) -> None:
        """Add a system message to the chat history.

        Args:
            content: The content of the system message, can be a string or a
            list of KernelContent instances that are turned into a single ChatMessageContent.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError

    @add_system_message.register
    def _(self, content: str, **kwargs: Any) -> None:
        """Add a system message to the chat history."""
        self.add_message(message=self._prepare_for_add(role=AuthorRole.SYSTEM, content=content, **kwargs))

    @add_system_message.register(list)
    def _(self, content: list[KernelContent], **kwargs: Any) -> None:
        """Add a system message to the chat history."""
        self.add_message(message=self._prepare_for_add(role=AuthorRole.SYSTEM, items=content, **kwargs))

    @singledispatchmethod
    def add_developer_message(self, content: str | list[KernelContent], **kwargs) -> None:
        """Add a system message to the chat history.

        Args:
            content: The content of the developer message, can be a string or a
            list of KernelContent instances that are turned into a single ChatMessageContent.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError

    @add_developer_message.register
    def _(self, content: str, **kwargs: Any) -> None:
        """Add a system message to the chat history."""
        self.add_message(message=self._prepare_for_add(role=AuthorRole.DEVELOPER, content=content, **kwargs))

    @add_developer_message.register(list)
    def _(self, content: list[KernelContent], **kwargs: Any) -> None:
        """Add a system message to the chat history."""
        self.add_message(message=self._prepare_for_add(role=AuthorRole.DEVELOPER, items=content, **kwargs))

    @singledispatchmethod
    def add_user_message(self, content: str | list[KernelContent], **kwargs: Any) -> None:
        """Add a user message to the chat history.

        Args:
            content: The content of the user message, can be a string or a
            list of KernelContent instances that are turned into a single ChatMessageContent.
            **kwargs: Additional keyword arguments.

        """
        raise NotImplementedError

    @add_user_message.register
    def _(self, content: str, **kwargs: Any) -> None:
        """Add a user message to the chat history."""
        self.add_message(message=self._prepare_for_add(role=AuthorRole.USER, content=content, **kwargs))

    @add_user_message.register(list)
    def _(self, content: list[KernelContent], **kwargs: Any) -> None:
        """Add a user message to the chat history."""
        self.add_message(message=self._prepare_for_add(role=AuthorRole.USER, items=content, **kwargs))

    @singledispatchmethod
    def add_assistant_message(self, content: str | list[KernelContent], **kwargs: Any) -> None:
        """Add an assistant message to the chat history.

        Args:
            content: The content of the assistant message, can be a string or a
            list of KernelContent instances that are turned into a single ChatMessageContent.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError

    @add_assistant_message.register
    def _(self, content: str, **kwargs: Any) -> None:
        """Add an assistant message to the chat history."""
        self.add_message(message=self._prepare_for_add(role=AuthorRole.ASSISTANT, content=content, **kwargs))

    @add_assistant_message.register(list)
    def _(self, content: list[KernelContent], **kwargs: Any) -> None:
        """Add an assistant message to the chat history."""
        self.add_message(message=self._prepare_for_add(role=AuthorRole.ASSISTANT, items=content, **kwargs))

    @singledispatchmethod
    def add_tool_message(self, content: str | list[KernelContent], **kwargs: Any) -> None:
        """Add a tool message to the chat history.

        Args:
            content: The content of the tool message, can be a string or a
            list of KernelContent instances that are turned into a single ChatMessageContent.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError

    @add_tool_message.register
    def _(self, content: str, **kwargs: Any) -> None:
        """Add a tool message to the chat history."""
        self.add_message(message=self._prepare_for_add(role=AuthorRole.TOOL, content=content, **kwargs))

    @add_tool_message.register(list)
    def _(self, content: list[KernelContent], **kwargs: Any) -> None:
        """Add a tool message to the chat history."""
        self.add_message(message=self._prepare_for_add(role=AuthorRole.TOOL, items=content, **kwargs))

    def add_message(
        self,
        message: ChatMessageContent | dict[str, Any],
        encoding: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a message to the history.

        This method accepts either a ChatMessageContent instance or a
        dictionary with the necessary information to construct a ChatMessageContent instance.

        Args:
            message (Union[ChatMessageContent, dict]): The message to add, either as
                a pre-constructed ChatMessageContent instance or a dictionary specifying 'role' and 'content'.
            encoding (Optional[str]): The encoding of the message. Required if 'message' is a dict.
            metadata (Optional[dict[str, Any]]): Any metadata to attach to the message. Required if 'message' is a dict.
        """
        if isinstance(message, ChatMessageContent):
            self.messages.append(message)
            return
        if "role" not in message:
            raise ContentInitializationError(f"Dictionary must contain at least the role. Got: {message}")
        if encoding:
            message["encoding"] = encoding
        if metadata:
            message["metadata"] = metadata
        self.messages.append(ChatMessageContent(**message))

    def _prepare_for_add(
        self, role: AuthorRole, content: str | None = None, items: list[KernelContent] | None = None, **kwargs: Any
    ) -> dict[str, str]:
        """Prepare a message to be added to the history."""
        kwargs["role"] = role
        if content:
            kwargs["content"] = content
        if items:
            kwargs["items"] = items
        return kwargs

    def remove_message(self, message: ChatMessageContent) -> bool:
        """Remove a message from the history.

        Args:
            message (ChatMessageContent): The message to remove.

        Returns:
            bool: True if the message was removed, False if the message was not found.
        """
        try:
            self.messages.remove(message)
            return True
        except ValueError:
            return False

    def __len__(self) -> int:
        """Return the number of messages in the history."""
        return len(self.messages)

    def __getitem__(self, index: int) -> ChatMessageContent:
        """Get a message from the history using the [] operator.

        Args:
            index (int): The index of the message to get.

        Returns:
            ChatMessageContent: The message at the specified index.
        """
        return self.messages[index]

    def __contains__(self, item: ChatMessageContent) -> bool:
        """Check if a message is in the history.

        Args:
            item (ChatMessageContent): The message to check for.

        Returns:
            bool: True if the message is in the history, False otherwise.
        """
        return item in self.messages

    def __str__(self) -> str:
        """Return a string representation of the history."""
        chat_history_xml = Element(CHAT_HISTORY_TAG)
        for message in self.messages:
            chat_history_xml.append(message.to_element())
        return tostring(chat_history_xml, encoding="unicode", short_empty_elements=True)

    def clear(self) -> None:
        """Clear the chat history."""
        self.messages.clear()

    def extend(self, messages: Iterable[ChatMessageContent]) -> None:
        """Extend the chat history with a list of messages.

        Args:
            messages: The messages to add to the history.
                Can be a list of ChatMessageContent instances or a ChatHistory itself.
        """
        self.messages.extend(messages)

    def replace(self, messages: Iterable[ChatMessageContent]) -> None:
        """Replace the chat history with a list of messages.

        This calls clear() and then extend(messages=messages).

        Args:
            messages: The messages to add to the history.
                Can be a list of ChatMessageContent instances or a ChatHistory itself.
        """
        self.clear()
        self.extend(messages=messages)

    def to_prompt(self) -> str:
        """Return a string representation of the history."""
        chat_history_xml = Element(CHAT_HISTORY_TAG)
        for message in self.messages:
            chat_history_xml.append(message.to_element())
        return tostring(chat_history_xml, encoding="unicode", short_empty_elements=True)

    def __iter__(self) -> Generator[ChatMessageContent, None, None]:  # type: ignore
        """Return an iterator over the messages in the history."""
        yield from self.messages

    def __eq__(self, other: Any) -> bool:
        """Check if two ChatHistory instances are equal."""
        if not isinstance(other, ChatHistory):
            return False

        return self.messages == other.messages

    @classmethod
    def from_rendered_prompt(cls: type[_T], rendered_prompt: str) -> _T:
        """Create a ChatHistory instance from a rendered prompt.

        Args:
            rendered_prompt (str): The rendered prompt to convert to a ChatHistory instance.

        Returns:
            ChatHistory: The ChatHistory instance created from the rendered prompt.
        """
        prompt_tag = "root"
        messages: list["ChatMessageContent"] = []
        prompt = rendered_prompt.strip()
        try:
            xml_prompt = XML(text=f"<{prompt_tag}>{prompt}</{prompt_tag}>")
        except ParseError as exc:
            logger.info(f"Could not parse prompt {prompt} as xml, treating as text, error was: {exc}")
            return cls(messages=[ChatMessageContent(role=AuthorRole.USER, content=unescape(prompt))])
        if xml_prompt.text and xml_prompt.text.strip():
            messages.append(ChatMessageContent(role=AuthorRole.SYSTEM, content=unescape(xml_prompt.text.strip())))
        for item in xml_prompt:
            if item.tag == CHAT_MESSAGE_CONTENT_TAG:
                messages.append(ChatMessageContent.from_element(item))
            elif item.tag == CHAT_HISTORY_TAG:
                for message in item:
                    messages.append(ChatMessageContent.from_element(message))
            if item.tail and item.tail.strip():
                messages.append(ChatMessageContent(role=AuthorRole.USER, content=unescape(item.tail.strip())))
        if len(messages) == 1 and messages[0].role == AuthorRole.SYSTEM:
            messages[0].role = AuthorRole.USER
        return cls(messages=messages)

    def serialize(self) -> str:
        """Serializes the ChatHistory instance to a JSON string.

        Returns:
            str: A JSON string representation of the ChatHistory instance.

        Raises:
            ValueError: If the ChatHistory instance cannot be serialized to JSON.
        """
        try:
            return self.model_dump_json(exclude_none=True, indent=2)
        except Exception as e:  # pragma: no cover
            raise ContentSerializationError(f"Unable to serialize ChatHistory to JSON: {e}") from e

    @classmethod
    def restore_chat_history(cls: type[_T], chat_history_json: str) -> _T:
        """Restores a ChatHistory instance from a JSON string.

        Args:
            chat_history_json (str): The JSON string to deserialize
                into a ChatHistory instance.

        Returns:
            ChatHistory: The deserialized ChatHistory instance.

        Raises:
            ValueError: If the JSON string is invalid or the deserialized data
                fails validation.
        """
        try:
            return cls(**json.loads(chat_history_json))
        except Exception as e:
            raise ContentInitializationError(f"Invalid JSON format: {e}")

    def store_chat_history_to_file(self, file_path: str) -> None:
        """Stores the serialized ChatHistory to a file.

        Args:
            file_path (str): The path to the file where the serialized data will be stored.
        """
        json_str = self.serialize()
        with open(file_path, "w") as file:
            file.write(json_str)

    @classmethod
    def load_chat_history_from_file(cls, file_path: str) -> "ChatHistory":
        """Loads the ChatHistory from a file.

        Args:
            file_path (str): The path to the file from which to load the ChatHistory.

        Returns:
            ChatHistory: The deserialized ChatHistory instance.
        """
        with open(file_path) as file:
            json_str = file.read()
        return cls.restore_chat_history(json_str)


@experimental_class
class ChatHistoryReducer(ChatHistory, ABC):
    """Defines a contract for reducing chat history."""

    target_count: int = Field(..., gt=0, description="Target message count.")
    threshold_count: int = Field(default=0, ge=0, description="Threshold count to avoid orphaning messages.")
    auto_reduce: bool = Field(
        default=False,
        description="Whether to automatically reduce the chat history, this happens when using add_message_async.",
    )

    @abstractmethod
    async def reduce(self) -> Self | None:
        """Reduce the chat history in some way (e.g., truncate, summarize).

        Returns:
            A possibly shorter list of messages, or None if no change is needed.
        """
        ...

    async def add_message_async(
        self,
        message: ChatMessageContent | dict[str, Any],
        encoding: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a message to the chat history.

        If auto_reduce is enabled, the history will be reduced after adding the message.
        """
        if isinstance(message, ChatMessageContent):
            self.messages.append(message)
            if self.auto_reduce:
                await self.reduce()
            return
        if "role" not in message:
            raise ContentInitializationError(f"Dictionary must contain at least the role. Got: {message}")
        if encoding:
            message["encoding"] = encoding
        if metadata:
            message["metadata"] = metadata
        self.messages.append(ChatMessageContent(**message))
        if self.auto_reduce:
            await self.reduce()


@experimental_function
def locate_summarization_boundary(history: list[ChatMessageContent]) -> int:
    """Identify the index of the first message that is not a summary message.

    This is indicated by the presence of the SUMMARY_METADATA_KEY in the message metadata.

    Returns:
        The insertion point index for normal history messages (i.e., after all summary messages).
    """
    for idx, msg in enumerate(history):
        if not msg.metadata or SUMMARY_METADATA_KEY not in msg.metadata:
            return idx
    return len(history)


@experimental_function
def contains_function_call_or_result(msg: ChatMessageContent) -> bool:
    """Return True if the message has any function call or function result."""
    return any(isinstance(item, (FunctionCallContent, FunctionResultContent)) for item in msg.items)


@experimental_function
def locate_safe_reduction_index(
    history: list[ChatMessageContent],
    target_count: int,
    threshold_count: int = 0,
    offset_count: int = 0,
) -> int | None:
    """Identify the index of the first message at or beyond the specified target_count.

    This index does not orphan sensitive content (function calls/results).

    This method ensures that the presence of a function-call always follows with its result,
    so the function-call and its function-result are never separated.

    In addition, it attempts to locate a user message within the threshold window so that
    context with the subsequent assistant response is preserved.

    Args:
        history: The entire chat history.
        target_count: The desired message count after reduction.
        threshold_count: The threshold beyond target_count required to trigger reduction.
                         If total messages <= (target_count + threshold_count), no reduction occurs.
        offset_count: Optional number of messages to skip at the start (e.g. existing summary messages).

    Returns:
        The index that identifies the starting point for a reduced history that does not orphan
        sensitive content. Returns None if reduction is not needed.
    """
    total_count = len(history)
    threshold_index = total_count - (threshold_count or 0) - target_count
    if threshold_index <= offset_count:
        return None

    message_index = total_count - target_count

    # Move backward to avoid cutting function calls / results
    # also skip over developer/system messages
    while message_index >= offset_count:
        if history[message_index].role not in (AuthorRole.DEVELOPER, AuthorRole.SYSTEM):
            break
        if not contains_function_call_or_result(history[message_index]):
            break
        message_index -= 1

    # This is our initial target truncation index
    target_index = message_index

    # Attempt to see if there's a user message in the threshold window
    while message_index >= threshold_index:
        if history[message_index].role == AuthorRole.USER:
            return message_index
        message_index -= 1

    return target_index


@experimental_function
def extract_range(
    history: list[ChatMessageContent],
    start: int,
    end: int | None = None,
    filter_func: Callable[[ChatMessageContent], bool] | None = None,
    preserve_pairs: bool = False,
) -> list[ChatMessageContent]:
    """Extract a range of messages from the source history, skipping any message for which we do not want to keep.

    For example, function calls/results, if desired.

    Args:
        history: The source history.
        start: The index of the first message to extract (inclusive).
        end: The index of the last message to extract (exclusive). If None, extracts through end.
        filter_func: A function that takes a ChatMessageContent and returns True if the message should
                        be skipped, False otherwise.
        preserve_pairs: If True, ensures that function call and result pairs are either both kept or both skipped.

    Returns:
        A list of extracted messages.
    """
    if end is None:
        end = len(history)

    sliced = list(range(start, end))

    # If we need to preserve call->result pairs, gather them
    pair_map = {}
    if preserve_pairs:
        pairs = get_call_result_pairs(history)
        # store in a dict for quick membership checking
        # call_idx -> result_idx, and also result_idx -> call_idx
        for cidx, ridx in pairs:
            pair_map[cidx] = ridx
            pair_map[ridx] = cidx

    extracted: list[ChatMessageContent] = []
    i = 0
    while i < len(sliced):
        idx = sliced[i]
        msg = history[idx]

        # If filter_func excludes it, skip it
        if filter_func and filter_func(msg):
            i += 1
            continue

        # skipping system/developer message
        if msg.role in (AuthorRole.DEVELOPER, AuthorRole.SYSTEM):
            i += 1
            continue

        # If preserve_pairs is on, and there's a paired index, skip or include them both
        if preserve_pairs and idx in pair_map:
            paired_idx = pair_map[idx]
            # If the pair is within [start, end), we must keep or skip them together
            if start <= paired_idx < end:
                # Check if the pair or itself fails filter_func
                if filter_func and (filter_func(history[paired_idx]) or filter_func(msg)):
                    # skip both
                    i += 1
                    # Also skip the paired index if it's in our current slice
                    if paired_idx in sliced:
                        # remove it from the slice so we don't process it again
                        sliced.remove(paired_idx)
                    continue
                # keep both
                extracted.append(msg)
                if paired_idx > idx:
                    # We'll skip the pair in the normal iteration by removing from slice
                    # but add it to extracted right now
                    extracted.append(history[paired_idx])
                    if paired_idx in sliced:
                        sliced.remove(paired_idx)
                else:
                    # if paired_idx < idx, it might appear later, so skip for now
                    # but we may have already processed it if i was the 2nd item
                    # either way, do not add duplicates
                    pass
                i += 1
                continue
            # If the paired_idx is outside [start, end), there's no conflict
            # so we can just do normal logic
            extracted.append(msg)
            i += 1
        else:
            # keep it if filter_func not triggered
            extracted.append(msg)
            i += 1

    return extracted


@experimental_class
class ChatHistorySummarizationReducer(ChatHistoryReducer):
    """A ChatHistory with logic to summarize older messages past a target count.

    This class inherits from ChatHistoryReducer, which in turn inherits from ChatHistory.
    It can be used anywhere a ChatHistory is expected, while adding summarization capability.

    Args:
        target_count: The target message count.
        threshold_count: The threshold count to avoid orphaning messages.
        auto_reduce: Whether to automatically reduce the chat history, default is False.
        service: The ChatCompletion service to use for summarization.
        summarization_instructions: The summarization instructions, optional.
        use_single_summary: Whether to use a single summary message, default is True.
        fail_on_error: Raise error if summarization fails, default is True.
        include_function_content_in_summary: Whether to include function calls/results in the summary, default is False.
        execution_settings: The execution settings for the summarization prompt, optional.

    """

    service: ChatCompletionClientBase
    summarization_instructions: str = Field(
        default=DEFAULT_SUMMARIZATION_PROMPT,
        description="The summarization instructions.",
        kw_only=True,
    )
    use_single_summary: bool = Field(default=True, description="Whether to use a single summary message.")
    fail_on_error: bool = Field(default=True, description="Raise error if summarization fails.")
    include_function_content_in_summary: bool = Field(
        default=False, description="Whether to include function calls/results in the summary."
    )
    execution_settings: PromptExecutionSettings | None = None

    @override
    async def reduce(self) -> Self | None:
        history = self.messages
        if len(history) <= self.target_count + (self.threshold_count or 0):
            return None  # No summarization needed

        logger.info("Performing chat history summarization check...")

        # 1. Identify where existing summary messages end
        insertion_point = locate_summarization_boundary(history)
        if insertion_point == len(history):
            # fallback fix: force boundary to something reasonable
            logger.warning("All messages are summaries, forcing boundary to 0.")
            insertion_point = 0

        # 2. Locate the safe reduction index
        truncation_index = locate_safe_reduction_index(
            history,
            self.target_count,
            self.threshold_count,
            offset_count=insertion_point,
        )
        if truncation_index is None:
            logger.info("No valid truncation index found.")
            return None

        # 3. Extract only the chunk of messages that need summarizing
        #    If include_function_content_in_summary=False, skip function calls/results
        #    Otherwise, keep them but never split pairs.
        messages_to_summarize = extract_range(
            history,
            start=0 if self.use_single_summary else insertion_point,
            end=truncation_index,
            filter_func=(contains_function_call_or_result if not self.include_function_content_in_summary else None),
            preserve_pairs=self.include_function_content_in_summary,
        )

        if not messages_to_summarize:
            logger.info("No messages to summarize.")
            return None

        try:
            # 4. Summarize the extracted messages
            summary_msg = await self._summarize(messages_to_summarize)
            logger.info("Chat History Summarization completed.")
            if not summary_msg:
                return None

            # Mark the newly-created summary with metadata
            summary_msg.metadata[SUMMARY_METADATA_KEY] = True

            # 5. Reassemble the new history
            keep_existing_summaries = []
            if insertion_point > 0 and not self.use_single_summary:
                keep_existing_summaries = history[:insertion_point]

            remainder = history[truncation_index:]
            new_history = [*keep_existing_summaries, summary_msg, *remainder]
            self.messages = new_history

            return self

        except Exception as ex:
            logger.warning("Summarization failed, continuing without summary.")
            if self.fail_on_error:
                raise ChatHistoryReducerException("Chat History Summarization failed.") from ex
            return None

    async def _summarize(self, messages: list[ChatMessageContent]) -> ChatMessageContent | None:
        """Use the ChatCompletion service to generate a single summary message."""
        from semantic_kernel.contents import AuthorRole

        chat_history = ChatHistory(messages=messages)
        execution_settings = self.execution_settings or self.service.get_prompt_execution_settings_from_settings(
            PromptExecutionSettings()
        )
        chat_history.add_message(
            ChatMessageContent(
                role=getattr(execution_settings, "instruction_role", AuthorRole.SYSTEM),
                content=self.summarization_instructions,
            )
        )
        return await self.service.get_chat_message_content(chat_history=chat_history, settings=execution_settings)

    def __eq__(self, other: object) -> bool:
        """Check if two ChatHistorySummarizationReducer objects are equal."""
        if not isinstance(other, ChatHistorySummarizationReducer):
            return False
        return (
            self.threshold_count == other.threshold_count
            and self.target_count == other.target_count
            and self.use_single_summary == other.use_single_summary
            and self.summarization_instructions == other.summarization_instructions
        )

    def __hash__(self) -> int:
        """Hash the object based on its properties."""
        return hash((
            self.__class__.__name__,
            self.threshold_count,
            self.target_count,
            self.summarization_instructions,
            self.use_single_summary,
            self.fail_on_error,
            self.include_function_content_in_summary,
        ))


@experimental_class
class ChatHistoryTruncationReducer(ChatHistoryReducer):
    """A ChatHistory that supports truncation logic.

    Because this class inherits from ChatHistoryReducer (which in turn inherits from ChatHistory),
    it can also be used anywhere a ChatHistory is expected, while adding truncation capability.

    Args:
        target_count: The target message count.
        threshold_count: The threshold count to avoid orphaning messages.
        auto_reduce: Whether to automatically reduce the chat history, default is False.
    """

    @override
    async def reduce(self) -> Self | None:
        history = self.messages
        if len(history) <= self.target_count + (self.threshold_count or 0):
            # No need to reduce
            return None

        logger.info("Performing chat history truncation check...")

        truncation_index = locate_safe_reduction_index(history, self.target_count, self.threshold_count)
        if truncation_index is None:
            logger.info(
                f"No truncation index found. Target count: {self.target_count}, Threshold: {self.threshold_count}"
            )
            return None

        logger.info(f"Truncating history to {truncation_index} messages.")
        truncated_list = extract_range(history, start=truncation_index)
        self.messages = truncated_list
        return self

    def __eq__(self, other: object) -> bool:
        """Compare equality based on truncation settings.

        (We don't factor in the actual ChatHistory messages themselves.)

        Returns:
            True if the other object is a ChatHistoryTruncationReducer with the same truncation settings.
        """
        if not isinstance(other, ChatHistoryTruncationReducer):
            return False
        return self.threshold_count == other.threshold_count and self.target_count == other.target_count

    def __hash__(self) -> int:
        """Return a hash code based on truncation settings.

        Returns:
            A hash code based on the truncation settings.
        """
        return hash((self.__class__.__name__, self.threshold_count, self.target_count))


@experimental_function
def get_call_result_pairs(history: list[ChatMessageContent]) -> list[tuple[int, int]]:
    """Identify all (FunctionCallContent, FunctionResultContent) pairs in the history.

    Return a list of (call_index, result_index) pairs for safe referencing.
    """
    pairs: list[tuple[int, int]] = []  # Correct type: list of tuples with integers
    call_ids_seen: dict[str, int] = {}  # Map call IDs (str) to their indices (int)

    # Gather all function-call IDs and their indices.
    for i, msg in enumerate(history):
        for item in msg.items:
            if isinstance(item, FunctionCallContent) and item.id is not None:
                call_ids_seen[item.id] = i

    # Now, match each FunctionResultContent to the earliest call ID with the same ID.
    for j, msg in enumerate(history):
        for item in msg.items:
            if isinstance(item, FunctionResultContent) and item.id is not None:
                call_id = item.id
                if call_id in call_ids_seen:
                    call_index = call_ids_seen[call_id]
                    pairs.append((call_index, j))
                    # Remove the call ID so we don't match it a second time
                    del call_ids_seen[call_id]
                    break

    return pairs


class RealtimeEvent(KernelBaseModel):
    """Base class for all service events."""

    service_event: Any | None = Field(default=None, description="The event content.")
    service_type: str | None = None
    event_type: ClassVar[Literal["service"]] = "service"


class RealtimeAudioEvent(RealtimeEvent):
    """Audio event type."""

    event_type: ClassVar[Literal["audio"]] = "audio"  # type: ignore
    audio: AudioContent = Field(..., description="Audio content.")


class RealtimeTextEvent(RealtimeEvent):
    """Text event type."""

    event_type: ClassVar[Literal["text"]] = "text"  # type: ignore
    text: TextContent = Field(..., description="Text content.")


class RealtimeFunctionCallEvent(RealtimeEvent):
    """Function call event type."""

    event_type: ClassVar[Literal["function_call"]] = "function_call"  # type: ignore
    function_call: FunctionCallContent = Field(..., description="Function call content.")


class RealtimeFunctionResultEvent(RealtimeEvent):
    """Function result event type."""

    event_type: ClassVar[Literal["function_result"]] = "function_result"  # type: ignore
    function_result: FunctionResultContent = Field(..., description="Function result content.")


class RealtimeImageEvent(RealtimeEvent):
    """Image event type."""

    event_type: ClassVar[Literal["image"]] = "image"  # type: ignore
    image: ImageContent = Field(..., description="Image content.")


RealtimeEvents = Annotated[
    Union[
        "RealtimeEvent",
        "RealtimeAudioEvent",
        "RealtimeTextEvent",
        "RealtimeFunctionCallEvent",
        "RealtimeFunctionResultEvent",
        "RealtimeImageEvent",
    ],
    Field(discriminator="event_type"),
]
