# Copyright (c) Microsoft. All rights reserved.

from pytest import fixture

from semantic_kernel.contents import FunctionCallContent


@fixture(scope="module")
def function_call():
    return FunctionCallContent(id="test", name="Test-Function", arguments='{"input": "world"}')
