# chapters/ch02_imagination.py
from core.agent import Agent


class SimpleImaginationAgent(Agent):
    """
    Implements Hobbes' concept of simple imagination.
    "...simple imagination, as when one imagineth a man, or horse, which he hath seen before."
    """

    async def process(self, sense_data, original_input=None):
        """
        Process sense data into simple imagination - the decaying sense that persists
        after the sensory stimulus is gone
        """
        if not self.enabled:
            return f"[{self.name} agent is disabled]"

        # Retrieve recent sense impressions to simulate "memory"
        recent_senses = await self.memory.get_recent_memories("sense_impressions", 3)

        prompt = f"""
        You are emulating simple imagination as described by Thomas Hobbes in Leviathan.
        
        Hobbes defines imagination as "nothing but decaying sense; and is found in men and many other 
        living creatures, as well sleeping as waking." Simple imagination is recalling 
        something as it was perceived before.
        
        Current sense impression:
        {sense_data}
        
        Previous sense impressions (if any):
        {self._format_memories(recent_senses)}
        
        Based on this current sense impression, simulate how it would persist in the mind as a 
        "decaying sense" - how it would be remembered shortly after being experienced. Focus on:
        
        1. What aspects would remain strongest in memory
        2. What might begin to fade or become less distinct
        3. How the core meaning would be preserved even as details decay
        
        Remember that according to Hobbes, "the longer the time is, after the sight or sense of any 
        object, the weaker is the imagination."
        """

        imagination = await self.llm.generate(prompt, temperature=0.7)

        # Store in memory
        await self.memory.add_memory(
            imagination, "simple_imagination", {"original_input": original_input}
        )

        return imagination

    def _format_memories(self, memories):
        if not memories:
            return "None"
        return "\n".join([f"- {m['content'][:150]}..." for m in memories])


class CompoundImaginationAgent(Agent):
    """
    Implements Hobbes' concept of compound imagination.
    "So when a man compoundeth the image of his own person with the image of the actions
    of another man... it is a compound imagination, and properly but a fiction of the mind."
    """

    async def process(self, simple_imagination, original_input=None):
        """
        Process simple imagination into compound imagination by combining elements
        from different memories and impressions
        """
        if not self.enabled:
            return f"[{self.name} agent is disabled]"

        # Get recent simple imaginations and sense impressions
        recent_imaginations = await self.memory.get_recent_memories(
            "simple_imagination", 3
        )
        recent_senses = await self.memory.get_recent_memories("sense_impressions", 2)

        prompt = f"""
        You are emulating compound imagination as described by Thomas Hobbes in Leviathan.
        
        Hobbes explains compound imagination as when "from the sight of a man at one time, and of a horse 
        at another, we conceive in our mind a centaur" or when a person "compoundeth the image of his own 
        person with the image of the actions of another man."
        
        Current simple imagination:
        {simple_imagination}
        
        Recent sense impressions and imaginations:
        {self._format_memories(recent_senses + recent_imaginations)}
        
        Create a compound imagination that combines elements from the current imagination with elements 
        from previous impressions or general knowledge. This should be a creative recombination that goes 
        beyond what was directly perceived - a "fiction of the mind" as Hobbes describes it.
        
        This compound imagination might include:
        1. Analogies or metaphors related to the original input
        2. Hypothetical scenarios extending from the perceived information
        3. Creative combinations of different concepts from the current and past impressions
        """

        compound_imagination = await self.llm.generate(prompt, temperature=0.8)

        # Store in memory
        await self.memory.add_memory(
            compound_imagination,
            "compound_imagination",
            {"original_input": original_input},
        )

        return compound_imagination

    def _format_memories(self, memories):
        if not memories:
            return "None"
        return "\n".join([f"- {m['content'][:150]}..." for m in memories])
