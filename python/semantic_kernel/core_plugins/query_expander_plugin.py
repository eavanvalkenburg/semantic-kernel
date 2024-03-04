# Copyright (c) Microsoft. All rights reserved.
import logging
import sys
from copy import copy
from typing import TYPE_CHECKING, ClassVar, Dict

if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated

from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel

if TYPE_CHECKING:
    from semantic_kernel.functions.kernel_arguments import KernelArguments
    from semantic_kernel.kernel import Kernel

logger: logging.Logger = logging.getLogger(__name__)


class QueryExpanderPlugin(KernelBaseModel):
    prompt: ClassVar[str] = """Based on the question below,
create {{$num_queries}} different queries to search for relevant information in a datastore or search engine
return the queries as a comma separated list and make sure there are no commas in the queries itself.
Question: {{$input}}"""
    execution_settings: Dict[str, PromptExecutionSettings]

    @kernel_function(
        description="Take the user input and expand the query for multiple search terms",
        name="expand_query",
    )
    async def expand_query(
        self,
        input: Annotated[str, "A user question that needs to be turned into multiple queries."],
        kernel: Annotated["Kernel", "The kernel instance."],
        arguments: Annotated["KernelArguments", "Arguments used by the kernel."],
        num_queries: Annotated[int, "The number of queries to generate."] = 3,
    ) -> str:
        """
        Take the user input and expand the query for multiple search queries.

        Example:
            {{queries.expand_query 'Paris France' num_queries=4}} => "Capital of France, Eiffel Tower, Louvre Museum, Paris France"

        Args:
            query -- The query to expand
            num_queries -- The number of queries to generate

        Returns:
            The expanded query as a string
        """
        arguments = copy(arguments)
        arguments["input"] = input
        arguments["num_queries"] = num_queries
        if not arguments.execution_settings:
            arguments.execution_settings = self.execution_settings
        return await kernel.invoke_prompt(self.prompt, arguments=arguments)
