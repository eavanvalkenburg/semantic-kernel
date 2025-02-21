# Copyright (c) Microsoft. All rights reserved.

from typing import Any, Literal


def _set_docstring(
    func_or_class: Any,
    state: Literal["experimental", "in preview"] = "experimental",
    note: str | None = None,
) -> Any:
    """Set the docstring with the state of the object and optionally the note."""
    if note:
        extra = f"Note: This {'class' if isinstance(func_or_class, type) else 'function'} is {state} and {note}."
    else:
        extra = f"Note: This {'class' if isinstance(func_or_class, type) else 'function'} is {state} and may change in the future."  # noqa: E501
    if func_or_class.__doc__:
        func_or_class.__doc__ += f"\n\n{extra}"
    else:
        func_or_class.__doc__ = extra
    return func_or_class


def experimental(func_or_class: Any = None, note: str | None = None) -> Any:
    """Decorator to mark something experimental.

    When a note is added, the docstring is appended with:
        Note: This function is experimental and {note}.
    When no note is added, the docstring is appended with:
        Note: This function is experimental and may change in the future.

    Args:
        func_or_class: The function or class to decorate
        note: The note to add to the docstring
    Returns:
        The decorated function or class

    """

    def decorator(func_or_class: Any) -> Any:
        setattr(
            func_or_class,
            "__experimental__",
            note or f"This {'class' if isinstance(func_or_class, type) else 'function'} is experimental.",
        )
        return _set_docstring(func_or_class, "experimental", note)

    if func_or_class is not None:
        return decorator(func_or_class)
    return decorator


def preview(func_or_class: Any = None, note: str | None = None) -> Any:
    """Decorator to mark something as in preview.

    When a note is added, the docstring is appended with:
        Note: This function is in preview and {note}.
    When no note is added, the docstring is appended with:
        Note: This function is in preview and may change in the future.

    Args:
        func_or_class: The function or class to decorate
        note: The note to add to the docstring
    Returns:
        The decorated function or class

    """

    def decorator(func_or_class: Any) -> Any:
        setattr(
            func_or_class,
            "__preview__",
            note or f"This {'class' if isinstance(func_or_class, type) else 'function'} is experimental.",
        )
        return _set_docstring(func_or_class, "in preview", note)

    if func_or_class is not None:
        return decorator(func_or_class)
    return decorator
