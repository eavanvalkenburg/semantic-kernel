# Copyright (c) Microsoft. All rights reserved.


from enum import Enum

from semantic_kernel.utils.lifecycle_decorators import experimental


@experimental
class OperationExtensions(Enum):
    """The operation extensions."""

    METHOD_KEY = "method"
    OPERATION_KEY = "operation"
    INFO_KEY = "info"
    SECURITY_KEY = "security"
    SERVER_URLS_KEY = "server-urls"
    METADATA_KEY = "operation-extensions"
