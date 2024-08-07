# Copyright (c) Microsoft. All rights reserved.

from typing import ClassVar

from pydantic import SecretStr

from semantic_kernel.kernel_pydantic import KernelBaseSettings


class OpenAISettings(KernelBaseSettings):
    """OpenAI model settings.

    The settings are first loaded from environment variables with the prefix 'OPENAI_'. If the
    environment variables are not found, the settings can be loaded from a .env file with the
    encoding 'utf-8'. If the settings are not found in the .env file, the settings are ignored;
    however, validation will fail alerting that the settings are missing.

    Args:
        api_key: OpenAI API key, see https://platform.openai.com/account/api-keys
            (Env var OPENAI_API_KEY)
        org_id: This is usually optional unless your account belongs to multiple organizations.
            (Env var OPENAI_ORG_ID)
        chat_model_id: The OpenAI chat model ID to use, for example, gpt-3.5-turbo or gpt-4.
            (Env var OPENAI_CHAT_MODEL_ID)
        text_model_id: The OpenAI text model ID to use, for example, gpt-3.5-turbo-instruct.
            (Env var OPENAI_TEXT_MODEL_ID)
        embedding_model_id: The OpenAI embedding model ID to use, for example, text-embedding-ada-002.
            (Env var OPENAI_EMBEDDING_MODEL_ID)
        env_file_path: if provided, the .env settings are read from this file path location
    """

    env_prefix: ClassVar[str] = "OPENAI_"

    api_key: SecretStr | None = None
    org_id: str | None = None
    chat_model_id: str | None = None
    text_model_id: str | None = None
    embedding_model_id: str | None = None
