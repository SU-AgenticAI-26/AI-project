"""
Secrets Manager for API Keys

This module provides secure access to API keys and other sensitive configuration.
Keys can be loaded from:
1. Environment variables (recommended for production)
2. Local secrets file (secrets.json) - should be gitignored
3. Fallback files (for development)

Usage:
    from secrets_manager import get_api_key

    openai_key = get_api_key('openai')
    nasa_key = get_api_key('nasa')
    semantic_scholar_key = get_api_key('semantic_scholar')
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SecretsManager:
    """Centralized secrets management for API keys and sensitive data."""

    def __init__(self, secrets_file: str = "secrets.json"):
        self.secrets_file = Path(secrets_file)
        self._secrets: Dict[str, Any] = {}
        self._load_secrets()

    def _load_secrets(self) -> None:
        """Load secrets from multiple sources in order of priority."""

        # 1. Environment variables (highest priority)
        env_mapping = {
            'openai': ['OPENAI_API_KEY', 'OPENAI_KEY'],
            'nasa': ['NASA_API_KEY', 'NASA_KEY'],
            'semantic_scholar': ['SEMANTIC_SCHOLAR_API_KEY', 'SEMANTIC_SCHOLAR_KEY', 'SS_API_KEY'],
            'anthropic': ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY'],
            'google': ['GOOGLE_API_KEY', 'GOOGLE_AI_API_KEY'],
            'huggingface': ['HUGGINGFACE_API_KEY', 'HF_API_KEY'],
        }

        for key_name, env_vars in env_mapping.items():
            for env_var in env_vars:
                value = os.getenv(env_var)
                if value:
                    self._secrets[key_name] = value
                    logger.info(f"Loaded {key_name} from environment variable {env_var}")
                    break

        # 2. Secrets file (medium priority)
        if self.secrets_file.exists():
            try:
                with open(self.secrets_file, 'r') as f:
                    file_secrets = json.load(f)
                for key, value in file_secrets.items():
                    if key not in self._secrets and value:
                        self._secrets[key] = value
                        logger.info(f"Loaded {key} from secrets file")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load secrets file: {e}")

        # 3. Legacy fallback files (lowest priority)
        legacy_files = {
            'nasa': ['nasakey.txt'],
            'semantic_scholar': ['sskey.txt', 'semantic_scholar_key.txt'],
        }

        for key_name, filenames in legacy_files.items():
            if key_name not in self._secrets:
                for filename in filenames:
                    file_path = Path(filename)
                    if file_path.exists():
                        try:
                            with open(file_path, 'r') as f:
                                value = f.read().strip()
                            if value:
                                self._secrets[key_name] = value
                                logger.info(f"Loaded {key_name} from legacy file {filename}")
                                break
                        except IOError as e:
                            logger.warning(f"Failed to read legacy file {filename}: {e}")

    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service."""
        return self._secrets.get(service)

    def has_api_key(self, service: str) -> bool:
        """Check if API key exists for a service."""
        return service in self._secrets and bool(self._secrets[service])

    def get_all_keys(self) -> Dict[str, str]:
        """Get all available API keys (masked for security)."""
        return {k: self._mask_key(v) for k, v in self._secrets.items()}

    def _mask_key(self, key: str) -> str:
        """Mask API key for display purposes."""
        if not key or len(key) < 8:
            return "****"
        return key[:4] + "****" + key[-4:]

    def save_secrets_file(self, secrets: Dict[str, str]) -> None:
        """Save secrets to the secrets file."""
        with open(self.secrets_file, 'w') as f:
            json.dump(secrets, f, indent=2)
        logger.info(f"Saved secrets to {self.secrets_file}")

    def update_secret(self, service: str, value: str) -> None:
        """Update a specific secret."""
        self._secrets[service] = value
        # Optionally save to file
        if self.secrets_file.exists():
            current = {}
            try:
                with open(self.secrets_file, 'r') as f:
                    current = json.load(f)
            except:
                pass
            current[service] = value
            self.save_secrets_file(current)


# Global instance
_secrets_manager = None

def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager

def get_api_key(service: str) -> Optional[str]:
    """Convenience function to get API key for a service."""
    return get_secrets_manager().get_api_key(service)

def has_api_key(service: str) -> bool:
    """Convenience function to check if API key exists."""
    return get_secrets_manager().has_api_key(service)

def setup_secrets_interactive() -> None:
    """Interactive setup for API keys."""
    manager = get_secrets_manager()

    services = {
        'openai': 'OpenAI API Key (for GPT models)',
        'nasa': 'NASA API Key (for astronomy data)',
        'semantic_scholar': 'Semantic Scholar API Key (for academic papers)',
        'anthropic': 'Anthropic API Key (for Claude models)',
        'google': 'Google AI API Key (for Gemini models)',
        'huggingface': 'HuggingFace API Key (for model access)',
    }

    print("🔐 API Keys Setup")
    print("=================")
    print("You can set API keys via:")
    print("1. Environment variables (recommended)")
    print("2. secrets.json file (will be created)")
    print("3. Individual .txt files (legacy)")
    print()

    secrets_to_save = {}

    for service, description in services.items():
        current = manager.get_api_key(service)
        if current:
            masked = manager._mask_key(current)
            print(f"✓ {service}: {masked} (already configured)")
        else:
            print(f"✗ {service}: Not configured")
            print(f"  {description}")
            key = input(f"  Enter {service} API key (or press Enter to skip): ").strip()
            if key:
                secrets_to_save[service] = key
                print(f"  ✓ Added {service}")
            print()

    if secrets_to_save:
        manager.save_secrets_file(secrets_to_save)
        print(f"\n💾 Saved {len(secrets_to_save)} API keys to secrets.json")
        print("⚠️  Make sure to add secrets.json to .gitignore!")
    else:
        print("\nNo new API keys to save.")

    print("\n📋 Current configuration:")
    for service, masked in manager.get_all_keys().items():
        print(f"  {service}: {masked}")


if __name__ == "__main__":
    # Run interactive setup when executed directly
    setup_secrets_interactive()