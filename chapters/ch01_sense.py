# chapters/ch01_sense.py
from core.agent import Agent


class SenseAgent(Agent):
    """
    Implements Hobbes' concept of sense as the original source of all thoughts.
    "The original of them all is that which we call sense, (for there is no conception in a man's mind
    which hath not at first, totally or by parts, been begotten upon the organs of sense)."
    """
    
    def __init__(self, name, llm_client, memory_manager):
        # Configure memory for sense impressions which are fleeting (only keep 3)
        memory_config = {
            "sense_impressions": {
                "max_memories": 3,  # Sense impressions are fleeting, keep fewer
                "summary_prompt": """
                Summarize these fleeting sensory impressions as they begin to decay from the mind:
                
                {entries}
                
                Create a brief impression of what remains after these sensations have mostly faded.
                As Hobbes notes, sensations decay quickly as they transition to imagination.
                """
            }
        }
        super().__init__(name, llm_client, memory_manager, memory_config)

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
        
        Analyze this input as if it were a sensory impression received by the mind, as Hobbes puts it "The cause of sense is the external body, or object, which presseth the organ proper to each sense, either immediately, as in the taste and touch; or mediately, as in seeing, hearing, and smelling: which pressure, by the mediation of nerves and other strings and membranes of the body, continued inwards to the brain and heart, causeth there a resistance, or counter-pressure, or endeavour of the heart to deliver itself: which en- deavour, because outward, seemeth to be some matter without. And this seeming, or fancy, is that which men call sense; and consisteth, as to the eye, in a light, or colour figured; to the ear, in a sound; to the nostril, in an odour; to the tongue and palate, in a savour; and to the rest of the body, in heat, cold, hardness, softness, and such other qualities as we discern by feeling."

        Format your response as a "sensory impression received by the mind" with the first 20 words that come to mind. This will serve as the foundation for all other thought processes. Use only 20 words, they can be phrases, things, sensations, feelings, images, ideas, etc.
        """

        sense_data = await self.llm.generate(prompt, temperature=0.7)

        # Store in memory
        await self.memory.add_memory(
            sense_data, "sense_impressions", {"input": input_text}
        )

        return sense_data
