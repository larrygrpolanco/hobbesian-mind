# Hobbesian Mind Simulator

A philosophical AI system that models Thomas Hobbes' theory of human cognition from "Leviathan," simulating how thoughts develop from sensory perception through imagination and reasoning.

## Project Goals

This project aims to implement Hobbes' model of human cognition as described in his seminal work "Leviathan" (1651). Hobbes provides a materialist account of the mind, describing how all thoughts originate with sensory perception and develop through processes of imagination and various forms of reasoning.

The philosophical goal is to create a computational simulation of cognitive processes as Hobbes understood them, allowing us to:

1. Explore how a 17th-century philosophical model of mind might work when implemented with modern AI
2. Better understand Hobbes' theory by making it "executable"
3. Create an AI system whose responses are grounded in a coherent philosophical framework

As Hobbes states: "Concerning the thoughts of man, I will consider them first singly, and afterwards in train or dependence upon one another." This project follows this approach by implementing discrete cognitive functions and their connections.

## How It Works

The system processes input through a series of stages that mirror Hobbes' conception of mind:

1. **Sense Perception**: The input is processed as "sensory data," extracting key concepts, properties, and relationships (Chapter I of Leviathan)

2. **Simple Imagination**: The sensory data is transformed into "decaying sense" - how it persists in memory after initial perception (Chapter II)

3. **Compound Imagination**: Elements from simple imagination are creatively recombined to form new mental constructs (Chapter II)

4. **Trains of Thought**: From imagination, two paths emerge (Chapter III):
   - **Unguided Thought**: Free-flowing, associative thoughts without specific direction
   - **Regulated Thought**: Goal-directed thinking aimed at solving a problem

5. **Causal/Effect Analysis**: Based on the nature of the query, the system either:
   - Seeks causes for an observed effect
   - Explores possible effects of a given cause

6. **Response Synthesis**: All thought processes are integrated into a cohesive response

## Project Architecture

The system follows a modular, agent-based design pattern where each Hobbesian cognitive process is implemented as a separate agent:

```
hobbesian_mind/
├── core/                      # Core components
│   ├── agent.py               # Base agent class
│   ├── llm_interface.py       # LLM client (DeepSeek)
│   ├── memory.py              # Memory management
├── chapters/                  # Agents for each chapter
│   ├── ch01_sense.py          # Sense perception
│   ├── ch02_imagination.py    # Simple and compound imagination
│   ├── ch03_train_of_thought.py # Unguided and regulated thought
├── memory_stores/             # Persistent memory storage
├── main.py                    # Main orchestrator
```

The architecture follows Hobbes' conception of thought as a sequential process where each stage builds upon the previous, with information flowing from sense to imagination to reasoning.

## Key Components

### Core Components

1. **Agent (agent.py)**  
   Base class for all cognitive processes. Each agent processes input and stores results in memory.

2. **LLMClient (llm_interface.py)**  
   Provides an interface to the DeepSeek LLM API, handling prompt construction and response generation.

3. **MemoryManager (memory.py)**  
   Manages "buckets" of memories corresponding to different cognitive processes. Provides methods for storing and retrieving memories.

### Cognitive Agents

1. **SenseAgent (ch01_sense.py)**  
   Implements Hobbes' concept of sense perception as the foundation of all thought. It analyzes raw input to extract key concepts and relationships.

2. **SimpleImaginationAgent and CompoundImaginationAgent (ch02_imagination.py)**  
   * Simple Imagination: Models how sense impressions persist as "decaying sense"
   * Compound Imagination: Combines elements from different impressions to create new mental constructs

3. **UnguidedThoughtAgent and RegulatedThoughtAgent (ch03_train_of_thought.py)**  
   * Unguided Thought: Generates associative, wandering thoughts without a specific goal
   * Regulated Thought: Produces goal-directed thinking that consistently returns to a central purpose

4. **CauseSeekingAgent and EffectSeekingAgent (ch03_train_of_thought.py)**  
   * Cause-Seeking: Works backward from effects to possible causes
   * Effect-Seeking: Works forward from causes to possible effects

### HobbesianMind (main.py)

The main orchestrator that connects all cognitive agents in a pipeline. It manages the flow of information between agents, determining which thought processes to apply based on the nature of the input.

## Usage

The system provides a simple interactive shell where you can:

1. Enter queries to be processed through Hobbes' model of cognition
2. Use the `memory` command to view all memory buckets
3. Use `memory <bucket_name>` to view specific memories (e.g., `memory sense_impressions`)
4. Use `exit` to quit the program

To run the system:

```bash
# Set your DeepSeek API key
export DEEPSEEK_API_KEY="your-api-key-here"

# Run the Hobbesian Mind
python main.py
```

The system will walk through each cognitive process, showing its progress, and provide a final synthesized response that integrates insights from all stages of thought.

---