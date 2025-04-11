# chapters/ch03_train_of_thought.py
from core.agent import Agent


class UnguidedThoughtAgent(Agent):
    """
    Implements Hobbes' concept of unguided, wandering thought.
    "The first is unguided, without design, and inconstant..."
    """

    async def process(self, input_text):
        # Get recent memories to provide context
        recent_memories = self.memory.buckets["unguided_thoughts"][-3:]

        # Create prompt for unguided thought generation
        prompt = f"""
        You are emulating the unguided train of thoughts as described by Thomas Hobbes in Leviathan.
        
        This is the wandering, associative thought that flows freely without a specific goal,
        "without design, and inconstant; wherein there is no passionate thought to govern and 
        direct those that follow."
        
        Current topic: {input_text}
        
        Previous thoughts: {self._format_memories(recent_memories)}
        
        Generate a train of wandering thoughts on this topic. Show how one thought
        naturally leads to another by loose association, without being directed toward any goal.
        Demonstrate the "wild ranging of the mind" where seemingly unrelated ideas connect
        through hidden associations, as in Hobbes' example of how thoughts might wander from
        civil war to the value of a Roman penny.
        """

        # Generate thought
        thought = await self.llm.generate(prompt, temperature=0.8)

        # Save to memory
        await self.memory.add_memory(thought, "unguided_thoughts", {"input": input_text})

        return thought

    def _format_memories(self, memories):
        if not memories:
            return "None"
        return "\n".join([f"- {m['content'][:100]}..." for m in memories])


class RegulatedThoughtAgent(Agent):
    """
    Implements Hobbes' concept of regulated, goal-directed thought.
    "The second is more constant, as being regulated by some desire and design..."
    """

    async def process(self, input_text, goal):
        # Get recent memories
        recent_memories = self.memory.buckets["regulated_thoughts"][-3:]

        prompt = f"""
        You are emulating the regulated train of thoughts as described by Thomas Hobbes in Leviathan.
        
        This is the purposeful, goal-directed thought that is "regulated by some desire and design."
        As Hobbes writes: "From desire ariseth the thought of some means we have seen produce the like of that
        which we aim at; and from the thought of that, the thought of means to that mean; and so continually,
        till we come to some beginning within our own power."
        
        Current topic: {input_text}
        Goal/Desire: {goal}
        
        Previous thoughts: {self._format_memories(recent_memories)}
        
        Generate a train of regulated thoughts directed toward achieving the stated goal. Show how each
        thought leads purposefully to the next, constantly returning to the goal when the mind might wander.
        Demonstrate how the goal "comes often to mind" and directs all thoughts toward it.
        """

        thought = await self.llm.generate(prompt, temperature=0.7)

        await self.memory.add_memory(
            thought, "regulated_thoughts", {"input": input_text, "goal": goal}
        )

        return thought

    def _format_memories(self, memories):
        if not memories:
            return "None"
        return "\n".join([f"- {m['content'][:100]}..." for m in memories])


class CauseSeekingAgent(Agent):
    """
    Implements Hobbes' concept of seeking causes from effects.
    "...when of an effect imagined we seek the causes or means that produce it"
    """

    async def process(self, effect):
        prompt = f"""
        You are emulating the cause-seeking thought process described by Thomas Hobbes in Leviathan.
        
        This is the first kind of regulated thought "when of an effect imagined we seek the causes 
        or means that produce it" - a backward reasoning process.
        
        Effect to explain: {effect}
        
        Generate a train of thoughts that work backward from this effect to possible causes.
        Show the reasoning process of investigating what might have produced this effect,
        considering different possible causes and evaluating them.
        """

        thought = await self.llm.generate(prompt, temperature=0.7)

        await self.memory.add_memory(thought, "cause_seeking_thoughts", {"effect": effect})

        return thought


class EffectSeekingAgent(Agent):
    """
    Implements Hobbes' concept of seeking possible effects from a cause.
    "...when imagining anything whatsoever, we seek all the possible effects that can by it be produced"
    """

    async def process(self, cause):
        prompt = f"""
        You are emulating the effect-seeking thought process described by Thomas Hobbes in Leviathan.
        
        This is the second kind of regulated thought "when imagining anything whatsoever, we seek all the 
        possible effects that can by it be produced" - a forward reasoning process that Hobbes notes 
        is unique to humans.
        
        Cause/thing to consider: {cause}
        
        Generate a train of thoughts that work forward from this cause to possible effects.
        Show the reasoning process of exploring what might result from this cause,
        imagining various possible consequences and developments.
        """

        thought = await self.llm.generate(prompt, temperature=0.7)

        await self.memory.add_memory(thought, "effect_seeking_thoughts", {"cause": cause})

        return thought
