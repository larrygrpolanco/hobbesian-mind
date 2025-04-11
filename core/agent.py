# core/agent.py
class Agent:
    """Base class for all thought process agents"""

    def __init__(self, name, llm_client, memory_manager):
        # Agent identifier
        self.name = name
        # LLM interface for generating responses
        self.llm = llm_client
        # Memory system for storing and retrieving thoughts
        self.memory = memory_manager

    async def process(self, input_text, **kwargs):
        """Process input through this thought process"""
        raise NotImplementedError("Subclasses must implement this")
