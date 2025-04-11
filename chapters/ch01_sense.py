# chapters/ch01_sense.py
from core.agent import Agent


class SenseAgent(Agent):
    """
    Implements Hobbes' concept of sense as the original source of all thoughts.
    "The original of them all is that which we call sense, (for there is no conception in a man's mind
    which hath not at first, totally or by parts, been begotten upon the organs of sense)."
    """

    async def process(self, input_text):
        """
        Process the "sensory input" (user's text) to extract its qualities and properties
        In Hobbes' terms, this creates the initial "appearance" or "fancy" in the mind
        """

        prompt = f"""
        You are emulating the process of sense perception as described by Thomas Hobbes in Leviathan.
        
        In Hobbes' philosophy, sense is "the original of all thoughts" and consists of appearances caused by
        external objects working on our sensory organs. For this AI system, the "external object" is the 
        following input text:
        
        "{input_text}"
        
        Analyze this input as if it were a sensory impression received by the mind, identifying:
        1. The key concepts or objects presented
        2. The qualities or properties of these concepts/objects
        3. The relationships between these elements
        4. Any emotional or value-laden aspects of the input
        
        Format your response as an analysis of the "sensory impression" received with the first 10 words that come to mind. This will serve as the foundation for all other thought processes. Use only 10 words, they do not have to form a coherent sentence, these can be sensations, feelings, images, ideas, etc.
        """

        sense_data = await self.llm.generate(prompt, temperature=0.7)

        # Store in memory
        await self.memory.add_memory(
            sense_data, "sense_impressions", {"input": input_text}
        )

        return sense_data
