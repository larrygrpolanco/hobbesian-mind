# core/memory.py
import json
import os
from datetime import datetime
import asyncio


class MemoryManager:
    """Manages different memory buckets for thought processes"""

    def __init__(self, storage_dir="./memory_stores", max_recent_memories=5, bucket_configs=None):
        # Directory where memories are stored
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Default maximum number of recent memories to keep before summarizing
        self.max_recent_memories = max_recent_memories
        
        # Custom configuration for buckets (memory length and summarization prompts)
        self.bucket_configs = bucket_configs or {}

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
        
        # Check if we need to summarize this bucket (like we do with conversation)
        if bucket_name in self.bucket_configs and "max_memories" in self.bucket_configs[bucket_name]:
            max_memories = self.bucket_configs[bucket_name]["max_memories"]
            # If we have twice as many memories as our max, summarize
            if len(self.buckets[bucket_name]) > max_memories * 2:
                await self._summarize_bucket(bucket_name)
                
        return memory

    async def get_recent_memories(self, bucket_name, limit=None):
        """Get the most recent memories from a bucket"""
        # Use bucket-specific limit if configured, otherwise use provided limit or default
        if limit is None:
            if bucket_name in self.bucket_configs and 'max_memories' in self.bucket_configs[bucket_name]:
                limit = self.bucket_configs[bucket_name]['max_memories']
            else:
                limit = self.max_recent_memories
                
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
        
        # Get bucket-specific max memories or use default
        max_memories = self.max_recent_memories
        if "conversation" in self.bucket_configs and "max_memories" in self.bucket_configs["conversation"]:
            max_memories = self.bucket_configs["conversation"]["max_memories"]
        
        # Check if we need to summarize older conversation entries
        if len(self.buckets["conversation"]) > max_memories * 2:
            await self._summarize_conversation()
            
        return entry
    
    async def _summarize_conversation(self):
        """Summarize older conversation entries to prevent context overflow"""
        # Get bucket-specific settings or use defaults
        max_memories = self.max_recent_memories
        summary_prompt = None
        
        if "conversation" in self.bucket_configs:
            config = self.bucket_configs["conversation"]
            if "max_memories" in config:
                max_memories = config["max_memories"]
            if "summary_prompt" in config:
                summary_prompt = config["summary_prompt"]
        
        # Keep the most recent entries
        recent_entries = self.buckets["conversation"][-max_memories:]
        entries_to_summarize = self.buckets["conversation"][:-max_memories]
        
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
        
        # Use custom prompt if provided, otherwise use default
        if summary_prompt:
            # Replace {entries} placeholder with the actual entries
            prompt = summary_prompt.replace("{entries}", entries_text)
        else:
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
        
    async def _summarize_bucket(self, bucket_name):
        """Summarize older entries in any memory bucket"""
        # Get bucket-specific settings or use defaults
        max_memories = self.max_recent_memories
        summary_prompt = None
        
        if bucket_name in self.bucket_configs:
            config = self.bucket_configs[bucket_name]
            if "max_memories" in config:
                max_memories = config["max_memories"]
            if "summary_prompt" in config:
                summary_prompt = config["summary_prompt"]
        
        # Keep the most recent entries
        recent_entries = self.buckets[bucket_name][-max_memories:]
        entries_to_summarize = self.buckets[bucket_name][:-max_memories]
        
        if not entries_to_summarize:
            return
            
        # Format entries for summarization
        entries_text = "\n".join([
            f"MEMORY: {entry['content'][:200]}..." 
            if len(entry['content']) > 200 else 
            f"MEMORY: {entry['content']}"
            for entry in entries_to_summarize
        ])
        
        # Use the LLM to create a summary
        from core.llm_interface import LLMClient
        llm = LLMClient()
        
        # Use custom prompt if provided, otherwise use default
        if summary_prompt:
            # Replace {entries} placeholder with the actual entries
            prompt = summary_prompt.replace("{entries}", entries_text)
        else:
            prompt = f"""
            Summarize the following {bucket_name} memories while preserving the key points:
            
            {entries_text}
            
            Create a concise summary that captures the essential information.
            Focus on the main concepts and details while reducing the length significantly.
            """
        
        summary = await llm.generate(prompt, temperature=0.5)
        
        # Store the summary
        summary_entry = {
            "content": summary,
            "timestamp": datetime.now().isoformat(),
            "bucket": bucket_name,
            "entries_summarized": len(entries_to_summarize),
            "first_timestamp": entries_to_summarize[0]["timestamp"],
            "last_timestamp": entries_to_summarize[-1]["timestamp"],
        }
        
        # Create a summary bucket for this type if it doesn't exist
        summary_bucket = f"{bucket_name}_summaries"
        if summary_bucket not in self.buckets:
            self.buckets[summary_bucket] = []
            self.locks[summary_bucket] = asyncio.Lock()
        
        self.buckets[summary_bucket].append(summary_entry)
        await self._save_bucket(summary_bucket)
        
        # Replace the old entries with just the recent ones
        self.buckets[bucket_name] = recent_entries
        await self._save_bucket(bucket_name)
        
        return summary_entry
        
    async def get_bucket_with_summaries(self, bucket_name, include_summaries=True):
        """Get memories from a bucket, including summaries of older entries if available"""
        # Start with the recent memories
        memories = self.buckets[bucket_name]
        
        # Add summaries if requested
        summary_bucket = f"{bucket_name}_summaries"
        if include_summaries and summary_bucket in self.buckets and self.buckets[summary_bucket]:
            # Get the most recent summary
            latest_summary = self.buckets[summary_bucket][-1]
            
            # Create a summary memory entry
            summary_memory = {
                "content": f"SUMMARY OF OLDER {bucket_name.upper()}: {latest_summary['content']}",
                "timestamp": latest_summary["timestamp"],
                "metadata": {
                    "is_summary": True,
                    "entries_summarized": latest_summary.get("entries_summarized", 0)
                }
            }
            
            # Add the summary at the beginning
            return [summary_memory] + memories
            
        return memories
    
    async def get_conversation_context(self, include_summaries=True):
        """Get conversation context for the LLM, including summaries of older exchanges"""
        # Get bucket-specific max memories or use default
        max_memories = self.max_recent_memories
        if "conversation" in self.bucket_configs and "max_memories" in self.bucket_configs["conversation"]:
            max_memories = self.bucket_configs["conversation"]["max_memories"]
        
        # Start with the most recent conversation entries up to the configured limit
        context = self.buckets["conversation"][-max_memories:] if self.buckets["conversation"] else []
        
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
