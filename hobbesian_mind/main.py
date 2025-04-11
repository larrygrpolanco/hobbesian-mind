# main.py
import asyncio
from core.memory import MemoryManager
from core.llm_interface import LLMClient

# Import agents from chapters
from chapters.ch01_sense import SenseAgent
from chapters.ch02_imagination import SimpleImaginationAgent, CompoundImaginationAgent
from chapters.ch03_train_of_thought import (
    UnguidedThoughtAgent,
    RegulatedThoughtAgent,
    CauseSeekingAgent,
    EffectSeekingAgent,
)


class HobbesianMind:
    """Main orchestrator for Hobbesian thought processes"""

    def __init__(self, model="gpt-4o"):
        # Core components
        self.memory = MemoryManager()
        self.llm = LLMClient(model=model)

        # Initialize agents for each chapter
        # Chapter I: Of Sense - processes raw input into sensory impressions
        self.sense_agent = SenseAgent("sense", self.llm, self.memory)

        # Chapter II: Of Imagination - transforms sensory data into imagination
        self.simple_imagination_agent = SimpleImaginationAgent(
            "simple_imagination", self.llm, self.memory
        )
        self.compound_imagination_agent = CompoundImaginationAgent(
            "compound_imagination", self.llm, self.memory
        )

        # Chapter III: Of the Consequence or Train of Imaginations - develops thought processes
        self.unguided_agent = UnguidedThoughtAgent("unguided", self.llm, self.memory)
        self.regulated_agent = RegulatedThoughtAgent("regulated", self.llm, self.memory)
        self.cause_agent = CauseSeekingAgent("cause_seeking", self.llm, self.memory)
        self.effect_agent = EffectSeekingAgent("effect_seeking", self.llm, self.memory)

    async def _extract_goal(self, user_input):
        """Extract a goal from user input for regulated thought"""
        prompt = f"""
        Given this user input: "{user_input}"
        
        Extract a clear goal or desire that would direct regulated thought in the Hobbesian sense.
        What would someone asking this question ultimately want to achieve or understand?
        
        Goal:
        """

        goal = await self.llm.generate(prompt, temperature=0.5)
        return goal.strip()

    async def _should_seek_causes(self, user_input):
        """Determine if the query is asking for causes rather than effects"""
        prompt = f"""
        Given this user input: "{user_input}"
        
        Determine if the user is more likely seeking:
        1. The CAUSES of something (why/how something happened or exists)
        2. The EFFECTS of something (what would result from or follow something)
        
        Answer with just "CAUSES" or "EFFECTS":
        """

        result = await self.llm.generate(prompt, temperature=0.3)
        return "CAUSE" in result.upper()

    async def _synthesize_response(self, user_input, results):
        """Create final response by integrating different thought processes"""
        # Create the list of thought processes that were used
        thought_processes = []

        for process_name, content in results.items():
            if process_name != "final_response" and process_name != "original_input":
                thought_processes.append(f"{process_name}: {content[:200]}...")

        prompt = f"""
        You are a philosophical AI system modeled after Thomas Hobbes' understanding of human cognition.
        You have processed the user's question through multiple Hobbesian thought processes:
        
        {' '.join(thought_processes)}
        
        Based on these thought processes, craft a thoughtful, philosophical response to:
        "{user_input}"
        
        Your response should integrate insights from the different thought processes,
        showing how the sequence from sense to imagination to trains of thought leads to understanding.
        Be philosophical yet accessible.
        """

        response = await self.llm.generate(prompt, temperature=0.7)
        return response

    async def process_query(self, user_input):
        """Process a user query through Hobbesian thought processes"""
        results = {"original_input": user_input}

        # Step 1: Sense perception (Chapter I)
        # This is the foundation of all thought in Hobbes' system - raw sensory data
        print("Processing sense perception...")
        results["sense_data"] = await self.sense_agent.process(user_input)

        # Step 2: Simple Imagination (Chapter II)
        # Takes sense data and creates "decaying sense" - how information persists in memory
        print("Processing simple imagination...")
        results["simple_imagination"] = await self.simple_imagination_agent.process(
            results["sense_data"], original_input=user_input
        )

        # Step 3: Compound Imagination (Chapter II)
        # Combines elements from different memories to create new mental constructs
        print("Processing compound imagination...")
        results["compound_imagination"] = await self.compound_imagination_agent.process(
            results["simple_imagination"], original_input=user_input
        )

        # Step 4a: Unguided Train of Thought (Chapter III)
        # Free-flowing, associative thought without direction
        print("Processing unguided train of thought...")
        results["unguided_thought"] = await self.unguided_agent.process(
            results["compound_imagination"]
        )

        # Step 4b: Regulated Train of Thought (Chapter III)
        # Goal-directed thought process aimed at achieving something specific
        print("Extracting goal...")
        goal = await self._extract_goal(user_input)
        results["goal"] = goal

        print("Processing regulated train of thought...")
        results["regulated_thought"] = await self.regulated_agent.process(
            results["compound_imagination"], goal
        )

        # Step 5: Causal/Effect Analysis (Chapter III)
        # Determines whether to analyze causes or effects based on the query
        should_seek_causes = await self._should_seek_causes(user_input)

        if should_seek_causes:
            print("Processing causal analysis...")
            results["causal_analysis"] = await self.cause_agent.process(user_input)
        else:
            print("Processing effect analysis...")
            results["effect_analysis"] = await self.effect_agent.process(user_input)

        # Final synthesis - combines all thought processes into a cohesive response
        print("Synthesizing final response...")
        results["final_response"] = await self._synthesize_response(user_input, results)

        return results


async def interactive_shell(mind):
    """Simple interactive shell for the Hobbesian mind with memory viewing"""
    print("\n=== Hobbesian Mind Simulator ===")
    print("Type a query to process through Hobbes' model of cognition.")
    print("Special commands:")
    print("  memory           - List all memory buckets")
    print("  memory <bucket>  - View memories in a specific bucket")
    print("  exit             - Exit the program")

    while True:
        user_input = input("\nQuery > ")

        # Handle special commands
        if user_input.lower() == "exit":
            break

        elif user_input.lower().startswith("memory"):
            parts = user_input.split()

            # Just "memory" command - list all buckets
            if len(parts) == 1:
                print("\nAvailable memory buckets:")
                for bucket in mind.memory.buckets:
                    count = len(mind.memory.buckets[bucket])
                    print(f"  {bucket} ({count} memories)")

            # "memory <bucket>" command - show content of specific bucket
            elif len(parts) > 1:
                bucket_name = parts[1]
                if bucket_name in mind.memory.buckets:
                    memories = mind.memory.buckets[bucket_name]
                    if not memories:
                        print(f"\nNo memories in '{bucket_name}' bucket.")
                    else:
                        print(f"\n=== Memories in '{bucket_name}' ===")
                        for i, memory in enumerate(memories):
                            print(f"\n--- Memory {i+1} ---")
                            print(f"Timestamp: {memory['timestamp']}")

                            # Print metadata if it exists
                            if memory["metadata"]:
                                print("Metadata:")
                                for key, value in memory["metadata"].items():
                                    print(f"  {key}: {value}")

                            # Print the actual content
                            print("\nContent:")
                            print(
                                memory["content"][:500] + "..."
                                if len(memory["content"]) > 500
                                else memory["content"]
                            )
                else:
                    print(f"Bucket '{bucket_name}' does not exist.")

            continue

        # Process normal query
        try:
            print("\nProcessing your query through Hobbesian thought processes...")
            results = await mind.process_query(user_input)

            print("\n=== FINAL RESPONSE ===")
            print(results["final_response"])
            print("\nType 'memory' to view the memories created during this process.")

        except Exception as e:
            print(f"Error processing query: {e}")


async def main():
    mind = HobbesianMind(model="gpt-4o")
    await interactive_shell(mind)


if __name__ == "__main__":
    asyncio.run(main())
