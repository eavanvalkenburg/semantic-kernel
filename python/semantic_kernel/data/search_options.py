# Copyright (c) Microsoft. All rights reserved.


from collections.abc import Callable
from typing import Generic, TypeVar

from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.utils.experimental_decorator import experimental_class

TModel = TypeVar("TModel")


@experimental_class
class SearchOptions(KernelBaseModel, Generic[TModel]):
    """Options for a search."""

    filter: Callable[[TModel], bool] | None = None
    include_total_count: bool = False
