# chapters/ch02_imagination.py
from core.agent import Agent


class SimpleImaginationAgent(Agent):
    """
    Implements Hobbes' concept of simple imagination.
    "...simple imagination, as when one imagineth a man, or horse, which he hath seen before."
    """
    
    def __init__(self, name, llm_client, memory_manager):
        # Configure memory for simple imagination with decay-focused summary
        memory_config = {
            "simple_imagination": {
                "max_memories": 5,  # Standard retention
                "summary_prompt": """
                Summarize these simple imaginations, emphasizing the decay described by Hobbes:
                
                {entries}
                
                Create a summary that shows how these impressions have weakened over time.
                As Hobbes writes: "By time, and by length of time, the image itself weareth out."
                Focus on what core essence remains after details have faded.
                """
            }
        }
        super().__init__(name, llm_client, memory_manager, memory_config)

    async def process(self, sense_data, original_input=None):
        """
        Process sense data into simple imagination - the decaying sense that persists
        after the sensory stimulus is gone
        """

        # Retrieve recent sense impressions with summaries to simulate "memory"
        recent_senses = await self.memory.get_bucket_with_summaries("sense_impressions")

        prompt = f"""
        You are emulating simple imagination as described by Thomas Hobbes in Leviathan.
        
        Hobbes defines imagination with a metaphor to motions caused by our sense impression of objects: "so also it happeneth in that motion which is made in the internal parts of a man, then, when he sees, dreams, etc. For after the object is removed, or the eye shut, we still retain an image of the thing seen, though more obscure than when we see it. And this is it the Latins call imagination, from the image made in seeing, and apply the same, though improperly, to all the other senses. But the Greeks call it fancy, which signifies appearance, and is as proper to one sense as to another. Imagination, therefore, is nothing but decaying sense; and is found in men and many other living creatures, as well sleeping as waking." 
        
        Simple imagination is recalling something as it was perceived before.
        
        Current sense impressions:
        {sense_data}
        
        Previous sense impressions (if any):
        {self._format_memories(recent_senses)}
        
        Based on this current sense impression, simulate how it would persist in the mind as a 
        "decaying sense" as Hobbes put it, "This decaying sense, when we would express the thing itself (I mean fancy itself), we call imagination, as I said before. But when we would express the decay, and signify that the sense is fading, old, and past, it is called memory. So that imagination and memory are but one thing, which for diverse considerations hath diverse names." - how it would be remembered shortly after being experienced. 
        
        Focus on:
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
    
    def __init__(self, name, llm_client, memory_manager):
        # Configure memory for compound imagination
        memory_config = {
            "compound_imagination": {
                "max_memories": 7,  # More durable and complex combinations last longer
                "summary_prompt": """
                Summarize these compound imaginations, which Hobbes describes as "fictions of the mind":
                
                {entries}
                
                Create a synthesis that shows how these creative combinations have evolved.
                Focus on the novel connections and fictional elements constructed
                from simpler impressions, as Hobbes explains when describing how we might
                combine impressions to imagine "a centaur" or other creative combinations.
                """
            }
        }
        super().__init__(name, llm_client, memory_manager, memory_config)

    async def process(self, simple_imagination, original_input=None):
        """
        Process simple imagination into compound imagination by combining elements
        from different memories and impressions
        """

        # Get recent simple imaginations and sense impressions with summaries
        recent_imaginations = await self.memory.get_bucket_with_summaries("simple_imagination")
        recent_senses = await self.memory.get_bucket_with_summaries("sense_impressions")

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
