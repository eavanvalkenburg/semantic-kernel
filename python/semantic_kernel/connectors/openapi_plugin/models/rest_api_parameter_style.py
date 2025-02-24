# Copyright (c) Microsoft. All rights reserved.

from enum import Enum

from semantic_kernel.utils.lifecycle_decorators import experimental


@experimental
class RestApiParameterStyle(Enum):
    """RestApiParameterStyle."""

    SIMPLE = "simple"
