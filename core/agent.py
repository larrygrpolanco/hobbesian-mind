# core/agent.py
class Agent:
    """Base class for all thought process agents"""

    def __init__(self, name, llm_client, memory_manager, memory_config=None):
        # Agent identifier
        self.name = name
        # LLM interface for generating responses
        self.llm = llm_client
        # Memory system for storing and retrieving thoughts
        self.memory = memory_manager
        
        # Configure memory settings for this agent if provided
        if memory_config:
            self.configure_memory(memory_config)

    def configure_memory(self, config):
        """Configure memory settings for buckets used by this agent
        
        Args:
            config: A dictionary mapping bucket names to their configuration:
                {
                    "bucket_name": {
                        "max_memories": 3,  # Number of recent memories to keep
                        "summary_prompt": "Custom summarization prompt with {entries} placeholder"
                    }
                }
        """
        # Initialize bucket_configs if not already present
        if not hasattr(self.memory, 'bucket_configs'):
            self.memory.bucket_configs = {}
            
        # Update the configuration
        for bucket_name, bucket_config in config.items():
            self.memory.bucket_configs[bucket_name] = bucket_config

    async def process(self, input_text, **kwargs):
        """Process input through this thought process"""
        raise NotImplementedError("Subclasses must implement this")
