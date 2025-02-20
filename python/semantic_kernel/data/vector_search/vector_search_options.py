# Copyright (c) Microsoft. All rights reserved.


from typing import Annotated, Generic, TypeVar

from pydantic import Field

from semantic_kernel.data.search_options import SearchOptions
from semantic_kernel.utils.experimental_decorator import experimental_class

TModel = TypeVar("TModel")


@experimental_class
class VectorSearchOptions(SearchOptions[TModel], Generic[TModel]):
    """Options for vector search, builds on TextSearchOptions."""

    vector_field_name: str | None = None
    top: Annotated[int, Field(gt=0)] = 3
    skip: Annotated[int, Field(ge=0)] = 0
    include_vectors: bool = False
