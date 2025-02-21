---
# These are optional elements. Feel free to remove any of them.
status: proposed
contact: edvan
date: 2025-02-21
deciders: taochan, bentho, edvan, evmattso, shahen
---

# Lifecycle management of python features

## Context and Problem Statement

The current lifecycle management of the features in the python version of SK is not well defined. It also differs from the .NET version of SK because of inherent differences in the languages. This ADR proposes a new lifecycle management for the python version of SK.

Currently:
- We only have a single package that contains everything
- We have a self-built `experimental_function` and `experimental_class` decorator
  - This does not have a text field for additional info
- There is a built-in decorator for deprecated features in typing_extensions or warnings

## Decision Drivers

- Easy to understand
- Easy to maintain
- Allow for clear separation of states of features
- Provide a mechanism for when something changes

## Considered options
- Inside the current package, add additional structure and process
- Split into a namespace package and use actual releases to version features

### Additional structure and process

There is a difference between a experimental feature, something as big as the agent framework, or a new API type like realtime api's, and a initial version of a new connector, the latter is likely more of a preview feature then a experimental feature.

So the proposal for this process is to have this:
- **Experimental**
  - This is a feature that is not yet ready for production use. It may be incomplete, buggy, or subject to major changes. It is intended for testing and feedback purposes only.
  - This will be indicated by a decorator that has a text field for additional info
  - Ideally this is covered by unit tests, integration tests are also optional, but they should run in such a way that they do not block a merge, since the underlying api might change without notice in some cases.
- **Preview**
    - This is a feature that is in the final stages of development and is expected to be released soon. It may still have some bugs or minor changes, but it is generally stable and ready for production use.
    - This will be indicated by a decorator that has a text field for additional info, the additional info should indicate when it is expected to be released
    - This should be fully covered by tests, including integration tests
- **Stable**
  - This is the standard state, so undecorated code is by definition stable, it should be covered by tests, including integration tests
- **Deprecated**
  - This is a feature that is no longer recommended for use and will be removed in future versions. It is still functional but should be avoided in new development.
  - This will be indicated by the built-in decorator for deprecated features in typing_extensions or warnings
  - The deprecated decorator should have a indication of when it will be removed, even if not known yet.
  - Integration tests will be disabled for these features, unit tests are still run.

### Namespace package
This option is to make SK into a namespace package, this means that several parts of split into their own packages inside our repo.
This would allow us to release different packages with their own status, so we could have the main package on version 1.23 and then a new feature in a package on version 0.1 to indicate it is in preview.
What this would look like is that the folder structure would become:
```
semantic_kernel/
    <core pieces>
    pyproject.toml
semantic_kernel-connectors-ai-open_ai/
    <services>
    <settings>
    pyproject.toml
etc.
```

The main pyproject.toml could have extras that pull in the other packages, while each of the subpackage pyprojects would contain all dependencies of that part.

A big downside for a user is that if they do not use the extra that code is just missing, while currently, if they install SK and the "right" dependencies, their code will (likely) still work, even if they have a slightly wrong version of a dependency.

## Decision outcome
At this time, the additional effort and developer experience downside are such that improving the current approach and putting more structure around the process is preferred.
