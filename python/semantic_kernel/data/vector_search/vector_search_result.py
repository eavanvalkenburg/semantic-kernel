# Copyright (c) Microsoft. All rights reserved.

from typing import Generic, TypeVar

from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.utils.lifecycle_decorators import experimental

TModel = TypeVar("TModel")


@experimental
class VectorSearchResult(KernelBaseModel, Generic[TModel]):
    """The result of a vector search."""

    record: TModel
    score: float | None = None
