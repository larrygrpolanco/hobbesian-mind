# core/memory.py
import json
import os
from datetime import datetime
import asyncio


class MemoryManager:
    """Manages different memory buckets for thought processes"""

    def __init__(self, storage_dir="./memory_stores"):
        # Directory where memories are stored
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        # Initialize memory buckets organized by Hobbes' cognitive processes
        self.buckets = {
            # Chapter I - Raw sensory data
            "sense_impressions": self._load_bucket("sense_impressions"),
            # Chapter II - Persistent and combined impressions
            "simple_imagination": self._load_bucket("simple_imagination"),
            "compound_imagination": self._load_bucket("compound_imagination"),
            # Chapter III - Different thought processes
            "unguided_thoughts": self._load_bucket("unguided_thoughts"),
            "regulated_thoughts": self._load_bucket("regulated_thoughts"),
            "cause_seeking_thoughts": self._load_bucket("cause_seeking_thoughts"),
            "effect_seeking_thoughts": self._load_bucket("effect_seeking_thoughts"),
        }

        # Locks to prevent file corruption with concurrent writes
        self.locks = {bucket: asyncio.Lock() for bucket in self.buckets}

    def _load_bucket(self, bucket_name):
        """Load memories from disk"""
        file_path = os.path.join(self.storage_dir, f"{bucket_name}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return []

    async def _save_bucket(self, bucket_name):
        """Save memories to disk with lock protection"""
        async with self.locks[bucket_name]:
            file_path = os.path.join(self.storage_dir, f"{bucket_name}.json")
            with open(file_path, "w") as f:
                json.dump(self.buckets[bucket_name], f, indent=2)

    async def add_memory(self, content, bucket_name, metadata=None):
        """Add a thought to a specific memory bucket"""
        memory = {
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self.buckets[bucket_name].append(memory)
        await self._save_bucket(bucket_name)
        return memory

    async def get_recent_memories(self, bucket_name, limit=5):
        """Get the most recent memories from a bucket"""
        memories = self.buckets[bucket_name]
        return memories[-limit:] if memories else []
