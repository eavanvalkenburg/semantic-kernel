# Copyright (c) Microsoft. All rights reserved.

from semantic_kernel.data.search_filter import SearchFilter
from semantic_kernel.utils.lifecycle_decorators import experimental


@experimental
class TextSearchFilter(SearchFilter):
    """A filter clause for a text search query."""

    pass
