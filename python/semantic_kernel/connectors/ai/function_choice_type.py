# Copyright (c) Microsoft. All rights reserved.

from enum import Enum


# @experimental
class FunctionChoiceType(Enum):
    """The type of function choice behavior."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
