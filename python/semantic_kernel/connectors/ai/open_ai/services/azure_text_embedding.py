# Copyright (c) Microsoft. All rights reserved.

import logging
from collections.abc import Mapping
from typing import Any

from openai import AsyncAzureOpenAI
from openai.lib.azure import AsyncAzureADTokenProvider
from pydantic import ValidationError

from semantic_kernel.connectors.ai.open_ai.services.azure_config_base import AzureOpenAIConfigBase
from semantic_kernel.connectors.ai.open_ai.services.open_ai_handler import OpenAIModelTypes
from semantic_kernel.connectors.ai.open_ai.services.open_ai_text_embedding_base import OpenAITextEmbeddingBase
from semantic_kernel.connectors.ai.open_ai.settings.azure_open_ai_settings import AzureOpenAISettings
from semantic_kernel.exceptions.service_exceptions import ServiceInitializationError
from semantic_kernel.utils.experimental_decorator import experimental_class

logger: logging.Logger = logging.getLogger(__name__)


@experimental_class
class AzureTextEmbedding(AzureOpenAIConfigBase, OpenAITextEmbeddingBase):
    """Azure Text Embedding."""

    def __init__(
        self,
        service_id: str | None = None,
        api_key: str | None = None,
        deployment_name: str | None = None,
        endpoint: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        ad_token: str | None = None,
        ad_token_provider: AsyncAzureADTokenProvider | None = None,
        default_headers: Mapping[str, str] | None = None,
        async_client: AsyncAzureOpenAI | None = None,
        env_file_path: str | None = None,
    ) -> None:
        """Initialize an AzureTextEmbedding service.

        All values supplied in the constructor are optional. If a value is not supplied, the service will attempt to
        load it from the environment or a .env file. When supplied the values have precedence over the environment or
        .env file.

        Args:
            service_id: The service ID. (Optional)
            api_key: The api key. (Optional)
            deployment_name: The deployment name. (Optional)
            endpoint: The deployment endpoint. (Optional)
            base_url: The deployment base_url. (Optional)
            api_version: The deployment api version. (Optional)
            ad_token: The Azure AD token for authentication. (Optional)
            ad_token_provider: The Azure AD token provider for authentication. (Optional)
            default_headers: The default headers mapping of string keys to string values for HTTP requests. (Optional)
            async_client: An existing client to use. (Optional)
            env_file_path: Use the environment settings file as a fallback to environment variables. (Optional)
        """
        try:
            azure_openai_settings = AzureOpenAISettings.create(
                env_file_path=env_file_path,
                api_key=api_key,
                embedding_deployment_name=deployment_name,
                endpoint=endpoint,
                base_url=base_url,
                api_version=api_version,
            )
        except ValidationError as exc:
            raise ServiceInitializationError(f"Invalid settings: {exc}") from exc
        if not azure_openai_settings.embedding_deployment_name:
            raise ServiceInitializationError("The Azure OpenAI embedding deployment name is required.")

        super().__init__(
            deployment_name=azure_openai_settings.embedding_deployment_name,
            endpoint=azure_openai_settings.endpoint,
            base_url=azure_openai_settings.base_url,
            api_version=azure_openai_settings.api_version,
            service_id=service_id,
            api_key=azure_openai_settings.api_key.get_secret_value() if azure_openai_settings.api_key else None,
            ad_token=ad_token,
            ad_token_provider=ad_token_provider,
            default_headers=default_headers,
            ai_model_type=OpenAIModelTypes.EMBEDDING,
            client=async_client,
        )

    @classmethod
    def from_dict(cls, settings: dict[str, Any]) -> "AzureTextEmbedding":
        """Initialize an Azure OpenAI service from a dictionary of settings.

        Args:
            settings: A dictionary of settings for the service.
                should contain keys: deployment_name, endpoint, api_key
                and optionally: api_version, ad_auth
        """
        return AzureTextEmbedding(
            service_id=settings.get("service_id"),
            api_key=settings.get("api_key"),
            deployment_name=settings.get("deployment_name"),
            endpoint=settings.get("endpoint"),
            base_url=settings.get("base_url"),
            api_version=settings.get("api_version"),
            ad_token=settings.get("ad_token"),
            ad_token_provider=settings.get("ad_token_provider"),
            default_headers=settings.get("default_headers"),
            env_file_path=settings.get("env_file_path"),
        )
