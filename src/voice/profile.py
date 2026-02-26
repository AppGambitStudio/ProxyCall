"""Voice profile management for VoiceBox."""

import logging

import aiohttp

logger = logging.getLogger(__name__)


async def list_profiles(base_url: str = "http://localhost:17493") -> list[dict]:
    """List all voice profiles from VoiceBox."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/profiles") as resp:
            return await resp.json()


async def get_profile(
    profile_id: str, base_url: str = "http://localhost:17493"
) -> dict | None:
    """Get a specific voice profile by ID."""
    profiles = await list_profiles(base_url)
    for p in profiles:
        if p["id"] == profile_id:
            return p
    return None
