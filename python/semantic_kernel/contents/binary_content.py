# Copyright (c) Microsoft. All rights reserved.

from deprecated import deprecated

from semantic_kernel.contents import BinaryContent

BinaryContent = deprecated(
    "BinaryContent has been moved, use from semantic_kernel.contents import BinaryContent instead."
)(BinaryContent)
