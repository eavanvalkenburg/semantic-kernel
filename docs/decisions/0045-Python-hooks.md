---
# These are optional elements. Feel free to remove any of them.
status: proposed
contact: eavanvalkenburg
date: 2024-05-14
deciders: matthewbolanos
consulted: moonbox3, alliscode
informed: dmytrostruk, markwallace
---

# Pythonic Hooks

## Context and Problem Statement

We need to decide on the best way to implement hooks in Python. Hooks are a way to allow users to extend the functionality of a library or application by providing a way to run custom code at specific points in the application. The approach chosen for dotnet might or might not make sense for Python, given developer expectations and experiences.

<!-- This is an optional element. Feel free to remove. -->

## Decision Drivers

- Pythonic developer experience
- Functional equivalence to dotnet filters

## Considered Options

- Event Handlers (current implementation)
- Filters
- Callbacks

## Decision Outcome

Chosen option: callbacks

### Consequences

- Good, because python developers coming from other python projects will be familiar with the concept
- Bad, because different from dotnet filters


## Pros and Cons of the Options

### Event Handlers

<!-- This is an optional element. Feel free to remove. -->
Current implementation, consists of a defined function that is called with a event object.
At the right place in the code that function is called, passing the event object.

- Good, because defined event input is fully documented.
- Bad, because adding a new event handler, consists of many pieces, a event definition, a function store in the kernel, and a function in the kernel that does the calls to the registered functions.

### Filters

Similar to the Event Handler, a context is created for a filter, the filter is called, and as part of that it gets the context and the next function to call, this allows the developer to manipulate both pre- and post- the event and also do things like error handling there.

#### Sample
filter that uppercases the user input in the chat plugin and catches any exceptions that occur in the function it wraps:

```python
async def function_filter(
    self,
    function_context: FunctionContext,
    next: Callable[[FunctionContext], Coroutine[..., None]],
) -> None:
    if function_context.function.plugin_name != "chat":
        await next(function_context)
        return

    function_context.settings.arguments['user_input'] = function_context.settings.arguments['user_input'].upper()
    try:
        await next(function_context)
    except Exception as e:
        print(f"Error in function {function_context.function.name}: {e}")

```

- Good, because context defined in the filter is fully documented.
- Good, because wrapping the call to the next (and the inner most to the function) allows for pre- and post- event handling.
- Bad, because dotnet specific concept, might not be familiar to python developers.
- Bad, developers will have to always do a extra action in their filter, namely call next

### Callbacks

Callbacks are a common approach in large python projects, like LangChain and Home Assistant. They provide a simple approach, a BackCallBack is defined, containing functions that are called at specific points with specific parameters, the developer creates a subclass of this class and implements only the functions that are needed.
The new class is then registered in the kernel and can be called at the right place in the code, the base class functions are empty and so they are present but do nothing.

#### Sample: 
a callback that uppercases the user input in the chat plugin and catches any exceptions that occur in the function it wraps:

```python

class ChatCallback(BaseCallback):
    def on_chat(self, event: EventContext):
        if event.function.plugin_name != "chat":
            return
        event.arguments['user_input'] = event.arguments['user_input'].upper()

    def on_error(self, event: EventContext):
        print(f"Error in function {event.function.name}: {event.exception}")

```

- Good, because python developers coming from other python projects will be familiar with the concept
- Good, because static type checking will work for developers, since they are overwriting a method
- Good, easy to add another callback, just add to the base callback and users can immediately use it
- Bad, because different from dotnet filters


## More Information
- [LangChain Base Callback](https://github.com/langchain-ai/langchain/blob/714cba96a8f41bae4ece6caa8d4d2f5f409dd25e/libs/core/langchain_core/callbacks/base.py#L263)
- [Home Assistant Event Bus](https://github.com/home-assistant/core/blob/55bf0b66474d8c3786173f399aaaec1967d4c189/homeassistant/core.py#L1437)

