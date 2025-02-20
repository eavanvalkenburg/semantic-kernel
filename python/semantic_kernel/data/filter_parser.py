# Copyright (c) Microsoft. All rights reserved.

import ast
import inspect
from collections.abc import Callable
from enum import Enum
from typing import Any, Generic, TypeVar

from semantic_kernel.data.record_definition.vector_store_model_definition import VectorStoreRecordDefinition
from semantic_kernel.exceptions.search_exceptions import SearchException
from semantic_kernel.kernel_pydantic import KernelBaseModel

TModel = TypeVar("TModel")


class Operators(str, Enum):
    """Enum for combining filters."""

    AND = "and"
    OR = "or"
    NOT = "not"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    COMPARE = "cmp"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "ge"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "le"
    IS = "is"
    IS_NOT = "is_not"
    IN = "in"
    NOT_IN = "not_in"


AST_MAPPING = {
    ast.And: Operators.AND,
    ast.Or: Operators.OR,
    ast.Compare: Operators.COMPARE,
    ast.Not: Operators.NOT,
    ast.Eq: Operators.EQUAL,
    ast.NotEq: Operators.NOT_EQUAL,
    ast.Gt: Operators.GREATER_THAN,
    ast.GtE: Operators.GREATER_THAN_OR_EQUAL,
    ast.Lt: Operators.LESS_THAN,
    ast.LtE: Operators.LESS_THAN_OR_EQUAL,
    ast.Is: Operators.IS,
    ast.IsNot: Operators.IS_NOT,
    ast.In: Operators.IN,
    ast.NotIn: Operators.NOT_IN,
}


class FilterParser(KernelBaseModel, Generic[TModel]):
    """Filter parser for search filters."""

    lambda_expression: Callable[[TModel], bool]
    record_definition: VectorStoreRecordDefinition

    def parse(self) -> Any:
        """Parse the expression into a string."""
        source = inspect.getsource(self.lambda_expression).strip()
        parsed_source = ast.parse(source)

        for node in ast.walk(parsed_source):
            if isinstance(node, ast.Lambda):
                break

        print(node)
        if not isinstance(node.body, ast.BoolOp):
            raise SearchException("Invalid filter expression, most be a boolean operation")
        # parse the body of the lambda expression

        expression = []
        full_list = []
        print(ast.dump(node.body, indent=4))
        for sub in ast.walk(node.body):
            if isinstance(sub, ast.BoolOp):
                continue
            full_list.append(sub)
            match sub:
                case ast.And() | ast.Or() | ast.Compare():
                    expression.append({"op": AST_MAPPING[sub.__class__]})
                case (
                    ast.Not()
                    | ast.Eq()
                    | ast.Gt()
                    | ast.GtE()
                    | ast.Lt()
                    | ast.LtE()
                    | ast.In()
                    | ast.NotEq()
                    | ast.Is()
                    | ast.IsNot()
                ):
                    if "ops" in expression[-1]:
                        expression[-1]["ops"].append(AST_MAPPING[sub.__class__])
                    else:
                        expression[-1]["ops"] = [AST_MAPPING[sub.__class__]]
                case ast.Attribute():
                    if sub.attr not in self.record_definition.fields:
                        raise SearchException(f"Invalid field name {sub.attr}")
                case _:
                    print(sub)
        print(full_list)
        return expression
