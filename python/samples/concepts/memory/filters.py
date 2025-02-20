# Copyright (c) Microsoft. All rights reserved.
from dataclasses import dataclass, field
from typing import Annotated
from uuid import uuid4

from semantic_kernel.data import (
    VectorStoreRecordDataField,
    VectorStoreRecordKeyField,
    VectorStoreRecordVectorField,
    vectorstoremodel,
)
from semantic_kernel.data.filter_parser import FilterParser
from semantic_kernel.data.search_options import SearchOptions


@vectorstoremodel
@dataclass
class DataModelDataclass:
    vector: Annotated[list[float], VectorStoreRecordVectorField]
    key: Annotated[str, VectorStoreRecordKeyField()] = field(default_factory=lambda: str(uuid4()))
    content: Annotated[str, VectorStoreRecordDataField(has_embedding=True, embedding_property_name="vector")] = (
        "content1"
    )
    other: str | None = None


def filter_func(x):
    return x.content == "test" and x.key in ["test1", "test2"]


def main():
    options = SearchOptions[DataModelDataclass](filter=lambda x: x.content == "test" or x.key in ["test1", "test2"])
    filter_parser = FilterParser[DataModelDataclass](
        lambda_expression=options.filter,
        record_definition=DataModelDataclass.__kernel_vectorstoremodel_definition__,
    )
    code1 = filter_parser.parse()
    print("Lambda code")
    print(code1)


if __name__ == "__main__":
    main()
