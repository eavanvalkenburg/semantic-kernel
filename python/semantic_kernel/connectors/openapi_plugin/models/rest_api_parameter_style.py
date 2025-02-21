# Copyright (c) Microsoft. All rights reserved.

from enum import Enum

from semantic_kernel.utils.experimental_decorator import experimental


@experimental
class RestApiParameterStyle(Enum):
    """RestApiParameterStyle."""

    SIMPLE = "simple"
