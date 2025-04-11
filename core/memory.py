# core/memory.py
import json
import os
from datetime import datetime
import asyncio


class MemoryManager:
    """Manages different memory buckets for thought processes"""

    def __init__(self, storage_dir="./memory_stores", max_recent_memories=5):
        # Directory where memories are stored
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        # Maximum number of recent memories to keep before summarizing
        self.max_recent_memories = max_recent_memories

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
            # Conversation tracking
            "conversation": self._load_bucket("conversation"),
            "memory_summaries": self._load_bucket("memory_summaries"),
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
        
    async def add_conversation_entry(self, role, content, metadata=None):
        """Add an entry to the conversation history"""
        entry = {
            "role": role,  # 'user' or 'system'
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        self.buckets["conversation"].append(entry)
        await self._save_bucket("conversation")
        
        # Check if we need to summarize older conversation entries
        if len(self.buckets["conversation"]) > self.max_recent_memories * 2:
            await self._summarize_conversation()
            
        return entry
    
    async def _summarize_conversation(self):
        """Summarize older conversation entries to prevent context overflow"""
        # Keep the most recent entries
        recent_entries = self.buckets["conversation"][-self.max_recent_memories:]
        entries_to_summarize = self.buckets["conversation"][:-self.max_recent_memories]
        
        if not entries_to_summarize:
            return
            
        # Create a prompt for summarization
        entries_text = "\n".join([
            f"{entry['role'].upper()}: {entry['content'][:200]}..." 
            if len(entry['content']) > 200 else 
            f"{entry['role'].upper()}: {entry['content']}"
            for entry in entries_to_summarize
        ])
        
        # Use the LLM to create a summary
        from core.llm_interface import LLMClient
        llm = LLMClient()
        prompt = f"""
        Summarize the following conversation exchanges while preserving the key points:
        
        {entries_text}
        
        Create a concise summary that captures the essential information.
        Focus on the main topics, requests, and responses while reducing the length significantly.
        """
        
        summary = await llm.generate(prompt, temperature=0.5)
        
        # Store the summary in memory_summaries
        summary_entry = {
            "content": summary,
            "timestamp": datetime.now().isoformat(),
            "entries_summarized": len(entries_to_summarize),
            "first_timestamp": entries_to_summarize[0]["timestamp"],
            "last_timestamp": entries_to_summarize[-1]["timestamp"],
        }
        
        self.buckets["memory_summaries"].append(summary_entry)
        await self._save_bucket("memory_summaries")
        
        # Replace the old entries with just the recent ones
        self.buckets["conversation"] = recent_entries
        await self._save_bucket("conversation")
        
        return summary_entry
        
    async def get_conversation_context(self, include_summaries=True):
        """Get conversation context for the LLM, including summaries of older exchanges"""
        # Start with the most recent conversation entries
        context = self.buckets["conversation"]
        
        # Add summaries of older exchanges if requested
        if include_summaries and self.buckets["memory_summaries"]:
            # Get the most recent summary
            latest_summary = self.buckets["memory_summaries"][-1]
            
            # Insert the summary at the beginning of the context
            context = [{
                "role": "system",
                "content": f"SUMMARY OF PREVIOUS CONVERSATION: {latest_summary['content']}"
            }] + context
            
        return context
