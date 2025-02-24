# Copyright (c) Microsoft. All rights reserved.


from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.utils.lifecycle_decorators import experimental


@experimental
class AzureCosmosDBNoSQLCompositeKey(KernelBaseModel):
    """Azure CosmosDB NoSQL composite key."""

    partition_key: str
    key: str
