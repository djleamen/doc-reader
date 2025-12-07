"""
Azure-specific configuration for experimental RAG pipeline.
Uses Azure OpenAI, AI Search, Document Intelligence, and Key Vault.
"""

import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential, ClientSecretCredential
from azure.core.credentials import TokenCredential
from loguru import logger

load_dotenv()


class AzureSettings:
    """Experimental Azure RAG configuration with Managed Identity support."""

    def __init__(self):
        # Authentication Strategy
        # Priority: Managed Identity (production) > Service Principal (CI/CD) > Development
        self.use_managed_identity = os.getenv(
            "AZURE_USE_MANAGED_IDENTITY", "false").lower() == "true"

        # Azure OpenAI Configuration
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.openai_api_version = os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.openai_embedding_deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        self.openai_chat_deployment = os.getenv(
            "AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")
        self.openai_api_key = os.getenv(
            "AZURE_OPENAI_API_KEY")  # Fallback only

        # Azure AI Search Configuration
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "")
        self.search_index_name = os.getenv(
            "AZURE_SEARCH_INDEX_NAME", "rag-documents")
        self.search_api_key = os.getenv(
            "AZURE_SEARCH_API_KEY")  # Fallback only
        self.search_api_version = os.getenv(
            "AZURE_SEARCH_API_VERSION", "2024-05-01-preview")

        # Azure Document Intelligence Configuration
        self.document_intelligence_endpoint = os.getenv(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
        self.document_intelligence_api_key = os.getenv(
            "AZURE_DOCUMENT_INTELLIGENCE_API_KEY")  # Fallback only

        # Azure Key Vault Configuration (for secrets management)
        self.key_vault_url = os.getenv("AZURE_KEY_VAULT_URL", "")

        # Azure Storage Configuration (for document storage)
        self.storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
        self.storage_container_name = os.getenv(
            "AZURE_STORAGE_CONTAINER_NAME", "documents")
        self.storage_connection_string = os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING")  # Fallback only

        # Service Principal (for CI/CD and non-Azure environments)
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET")

        # RAG Pipeline Configuration
        self.chunk_size = int(os.getenv("AZURE_CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("AZURE_CHUNK_OVERLAP", "200"))
        self.top_k_results = int(os.getenv("AZURE_TOP_K_RESULTS", "5"))
        self.embedding_batch_size = int(
            os.getenv("AZURE_EMBEDDING_BATCH_SIZE", "20"))  # S0 tier default
        self.enable_semantic_search = os.getenv(
            "AZURE_ENABLE_SEMANTIC_SEARCH", "true").lower() == "true"
        self.enable_hybrid_search = os.getenv(
            "AZURE_ENABLE_HYBRID_SEARCH", "true").lower() == "true"

        # Performance & Reliability
        self.max_retries = int(os.getenv("AZURE_MAX_RETRIES", "3"))
        self.retry_delay = int(os.getenv("AZURE_RETRY_DELAY", "1"))
        self.timeout = int(os.getenv("AZURE_TIMEOUT", "30"))
        self.enable_caching = os.getenv(
            "AZURE_ENABLE_CACHING", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("AZURE_CACHE_TTL", "3600"))

        # Monitoring & Logging
        self.enable_app_insights = os.getenv(
            "AZURE_ENABLE_APP_INSIGHTS", "false").lower() == "true"
        self.app_insights_connection_string = os.getenv(
            "AZURE_APP_INSIGHTS_CONNECTION_STRING")

        # Temperature and token settings
        self.temperature = float(os.getenv("AZURE_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("AZURE_MAX_TOKENS", "4000"))

    def get_credential(self) -> TokenCredential:
        """
        Get Azure credential with proper authentication flow.

        Authentication Priority:
        1. Managed Identity (for Azure-hosted apps)
        2. Service Principal (for CI/CD)
        3. Azure CLI (for local development)
        4. Interactive Browser (fallback)

        Returns:
            TokenCredential: Azure credential object
        """
        try:
            if self.use_managed_identity:
                logger.info("Using Managed Identity for authentication")
                credential = ManagedIdentityCredential()
            elif self.client_id and self.client_secret and self.tenant_id:
                logger.info("Using Service Principal for authentication")
                credential = ClientSecretCredential(
                    tenant_id=self.tenant_id,
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
            else:
                logger.info(
                    "Using DefaultAzureCredential (Azure CLI or Interactive)")
                credential = DefaultAzureCredential()

            return credential
        except Exception as e:
            logger.error(f"Failed to create Azure credential: {e}")
            raise

    def validate_configuration(self) -> tuple[bool, list[str]]:
        """
        Validate required Azure configuration.

        Returns:
            tuple: (is_valid, list of missing configurations)
        """
        missing = []

        if not self.openai_endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")

        if not self.search_endpoint:
            missing.append("AZURE_SEARCH_ENDPOINT")

        if not self.document_intelligence_endpoint:
            missing.append("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

        # If not using managed identity, check for API keys or service principal
        if not self.use_managed_identity:
            if not self.openai_api_key:
                missing.append(
                    "AZURE_OPENAI_API_KEY (or use Managed Identity)")
            if not self.search_api_key:
                missing.append(
                    "AZURE_SEARCH_API_KEY (or use Managed Identity)")
            if not self.document_intelligence_api_key:
                missing.append(
                    "AZURE_DOCUMENT_INTELLIGENCE_API_KEY (or use Managed Identity)")

        is_valid = len(missing) == 0

        if not is_valid:
            logger.warning(
                f"Missing Azure configuration: {', '.join(missing)}")
        else:
            logger.info("Azure configuration validated successfully")

        return is_valid, missing

    @property
    def is_configured(self) -> bool:
        """Check if Azure RAG pipeline is properly configured."""
        is_valid, _ = self.validate_configuration()
        return is_valid


# Global Azure settings instance
azure_settings = AzureSettings()
