#!/usr/bin/env python3
"""
Environment Variables Setup Script

This script helps you set up API keys as environment variables.
Run this script to configure your API keys securely.
"""

import os
import sys
import subprocess
from pathlib import Path

def set_environment_variable(name: str, value: str):
    """Set an environment variable for the current session and future sessions."""
    if not value or value.strip() == "":
        return False

    # Set for current session
    os.environ[name] = value

    # Try to add to shell profile for persistence
    shell_profiles = [
        Path.home() / ".bashrc",
        Path.home() / ".zshrc",
        Path.home() / ".profile"
    ]

    export_line = f'export {name}="{value}"'

    for profile in shell_profiles:
        if profile.exists():
            try:
                with open(profile, 'r') as f:
                    content = f.read()

                # Check if already exists
                if f'export {name}=' in content:
                    print(f"⚠️  {name} already exists in {profile.name}, skipping...")
                    continue

                # Add to profile
                with open(profile, 'a') as f:
                    f.write(f'\n# AI Research Project API Key\n{export_line}\n')

                print(f"✅ Added {name} to {profile.name}")
                return True

            except Exception as e:
                print(f"⚠️  Could not update {profile.name}: {e}")

    # Fallback: create .env file
    env_file = Path(".env")
    try:
        if env_file.exists():
            with open(env_file, 'r') as f:
                env_content = f.read()
        else:
            env_content = ""

        if f'{name}=' not in env_content:
            with open(env_file, 'a') as f:
                f.write(f'{export_line}\n')
            print(f"✅ Added {name} to .env file")
            return True
    except Exception as e:
        print(f"⚠️  Could not create .env file: {e}")

    return False

def mask_key(key: str) -> str:
    """Mask an API key for display."""
    if not key or len(key) < 8:
        return "****"
    return key[:6] + "..." + key[-4:]

def main():
    print("🔐 Environment Variables Setup for AI Research Project")
    print("=" * 60)
    print("This will set up API keys as environment variables.")
    print("Keys will be available in:")
    print("1. Current terminal session")
    print("2. Future terminal sessions (added to shell profile)")
    print("3. .env file (fallback)")
    print()

    # Check current environment
    existing_keys = {}
    key_names = {
        'OPENAI_API_KEY': 'OpenAI',
        'NASA_API_KEY': 'NASA',
        'SEMANTIC_SCHOLAR_API_KEY': 'Semantic Scholar',
        'ANTHROPIC_API_KEY': 'Anthropic',
        'GOOGLE_API_KEY': 'Google AI',
        'HUGGINGFACE_API_KEY': 'HuggingFace'
    }

    print("📋 Current API Key Status:")
    for env_var, service in key_names.items():
        value = os.getenv(env_var)
        if value:
            existing_keys[env_var] = value
            print(f"✅ {service}: {mask_key(value)} (already set)")
        else:
            print(f"❌ {service}: Not set")
    print()

    if existing_keys:
        overwrite = input("Some keys are already set. Overwrite? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("Keeping existing keys.")
            return

    # Collect new keys
    new_keys = {}
    print("🔑 Enter your API keys (press Enter to skip):")
    print("-" * 40)

    for env_var, service in key_names.items():
        if env_var in existing_keys:
            current = mask_key(existing_keys[env_var])
            use_current = input(f"Keep current {service} key ({current})? (Y/n): ").lower().strip()
            if use_current in ('', 'y', 'yes'):
                continue

        key = input(f"{service} API Key: ").strip()
        if key:
            new_keys[env_var] = key
            print(f"✓ {service} key added")
        else:
            print(f"⏭️  Skipped {service}")

    print()

    if not new_keys:
        print("No new keys to set.")
        return

    # Set the keys
    print("🔧 Setting environment variables...")
    success_count = 0

    for env_var, key in new_keys.items():
        if set_environment_variable(env_var, key):
            success_count += 1
            print(f"✅ {env_var} set successfully")
        else:
            print(f"❌ Failed to set {env_var}")

    print()
    print("🎉 Setup Complete!")
    print(f"Set {success_count} out of {len(new_keys)} API keys")
    print()
    print("🔄 To apply changes in current session:")
    print("   source ~/.bashrc  # or your shell profile")
    print()
    print("🧪 Test the setup:")
    print("   python -c \"from secrets_manager import get_api_key; print('OpenAI:', get_api_key('openai') is not None)\"")
    print()
    print("🚀 Run your application:")
    print("   streamlit run integrated_research_app.py")

if __name__ == "__main__":
    main()