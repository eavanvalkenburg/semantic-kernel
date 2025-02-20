# Copyright (c) Microsoft. All rights reserved.

from semantic_kernel.data.filter_parser import FilterParser
from semantic_kernel.data.vector_search.vector_search_options import VectorSearchOptions


def test_vector_search_options(data_model_type_dataclass):
    """Test VectorSearchOptions class."""
    options = VectorSearchOptions[data_model_type_dataclass]()
    assert options.vector_field_name is None
    assert options.top == 3
    assert options.skip == 0
    assert options.include_vectors is False
    assert options.filter is None


def test_vector_search_options_with_values(data_model_type_dataclass):
    """Test VectorSearchOptions class with values."""
    options = VectorSearchOptions[data_model_type_dataclass](
        vector_field_name="test_vector",
        top=5,
        skip=2,
        include_vectors=False,
        filter=lambda x: x.content == "test",
    )
    assert options.vector_field_name == "test_vector"
    assert options.top == 5
    assert options.skip == 2
    assert not options.include_vectors
    assert options.filter is not None
    instance = data_model_type_dataclass(content="test", vector=[1, 2, 3], id="test")
    assert options.filter(instance)


def test_vector_search_options_with_filter(data_model_type_dataclass):
    """Test VectorSearchOptions class with values."""
    options = VectorSearchOptions[data_model_type_dataclass](
        vector_field_name="test_vector",
        filter=lambda x: x.content == "test" and x.id in ["test1", "test2"],
    )
    assert options.vector_field_name == "test_vector"
    assert options.filter is not None
    instance = data_model_type_dataclass(content="test", vector=[1, 2, 3], id="test")
    assert not options.filter(instance)


def test_filter_parsing(data_model_type_dataclass):
    """Test parsing of filter clauses."""

    expr = lambda x: x.content == "test" and x.id in ["test1", "test2"]  # noqa: E731
    filter_parser = FilterParser[data_model_type_dataclass](
        lambda_expression=expr,
        record_definition=data_model_type_dataclass.__kernel_vectorstoremodel_definition__,
    )
    filter_parser.parse()
