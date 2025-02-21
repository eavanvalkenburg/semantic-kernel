# Copyright (c) Microsoft. All rights reserved.

from semantic_kernel.utils.experimental_decorator import experimental


@experimental
def my_function() -> None:
    """This is a sample function docstring."""
    pass


@experimental
def my_function_no_doc_string() -> None:
    pass


@experimental(note="this function will be GA by the end of the year")
def my_function_with_note() -> None:
    """This is a sample function docstring."""
    pass


def test_function_experimental_decorator() -> None:
    assert (
        my_function.__doc__
        == "This is a sample function docstring.\n\nNote: This function is experimental and may change in the future."
    )
    assert hasattr(my_function, "__experimental__")
    assert my_function.__experimental__ == "This function is experimental."


def test_function_experimental_decorator_with_note() -> None:
    assert (
        my_function_with_note.__doc__
        == "This is a sample function docstring.\n\nNote: This function is experimental and this function will be GA by the end of the year."  # noqa: E501
    )
    assert hasattr(my_function_with_note, "__experimental__")
    assert my_function_with_note.__experimental__ == "this function will be GA by the end of the year"


def test_function_experimental_decorator_with_no_doc_string() -> None:
    assert my_function_no_doc_string.__doc__ == "Note: This function is experimental and may change in the future."
    assert hasattr(my_function_no_doc_string, "__experimental__")
    assert my_function_no_doc_string.__experimental__ == "This function is experimental."
