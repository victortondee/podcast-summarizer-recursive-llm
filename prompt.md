# Build a Recursive Language Model (RLM) System - AI Coder Prompt

## What You're Building

Create a Recursive Language Model (RLM) system that allows LLMs to handle extremely long contexts by:
1. **Managing their own context** - The LLM decides what to keep in memory vs. delegate
2. **Using Python as extended memory** - Variables and data structures act as persistent memory
3. **Spawning sub-LLMs** - Parallel "worker" LLMs process chunks, returning summaries to main LLM
4. **Iterative answer building** - Build answers progressively, not in one shot

**Key insight**: Instead of cramming everything into the LLM's context window, the LLM actively manages what it needs to remember using code and delegation.

## Core Concept: How RLM Achieves Long Context & Memory

### The Problem with Normal LLMs
- **Fixed context window**: 128k tokens = ~100k words maximum
- **Context rot**: Performance degrades as context fills up
- **One-shot answers**: Must produce final answer in one generation
- **No persistent memory**: Each conversation turn starts fresh

### How RLM Solves This

#### 1. **Context Compression via Delegation**
```
Instead of:
  Main LLM sees: [10,000 documents] → tries to process all → context overflow

RLM does:
  Main LLM sees: [Task description]
  Main LLM thinks: "I'll split this into 100 chunks"
  Main LLM code: llm_batch([doc1, doc2, ...doc100]) → gets back 100 summaries
  Main LLM sees: [100 summaries] → much smaller context!
```

**Result**: Main LLM never sees the full 10,000 documents. Sub-LLMs process them and return compressed results.

#### 2. **Python Variables as Persistent Memory**
```
Normal LLM:
  Turn 1: "The answer is probably X" [forgets this]
  Turn 2: "Let me reconsider..." [starts over]

RLM:
  Turn 1: answer["content"] = "Initial thoughts: probably X"
  Turn 2: print(answer["content"]) → sees previous work → builds on it
  Turn 3: answer["content"] += "\nAfter analysis: definitely Y"
```

**Result**: The `answer` dictionary acts as the LLM's external memory that persists across turns.

#### 3. **Iterative Refinement (Not One-Shot)**
```
Normal LLM:
  Input → [Think] → Output final answer → Done

RLM:
  Input → [Think + Code] → Store in answer["content"]
       → [Review own answer] → Spot error
       → [Fix with code] → answer["content"] = corrected_version
       → [Verify] → answer["ready"] = True
```

**Result**: The LLM can edit and improve its answer multiple times before committing.

#### 4. **Hierarchical Memory Structure**
```
Main LLM Memory (small context):
  - Current task
  - Available tools
  - Current iteration's REPL output only (truncated to 4096 chars)
  - answer variable state

Python REPL Memory (unlimited):
  - Full input data (could be gigabytes)
  - All variables from previous turns
  - Intermediate computation results
  - Sub-LLM response cache

Sub-LLM Memory (fresh context each):
  - Only sees the specific chunk they're assigned
  - Returns compressed result
  - Gets garbage collected after
```

**Result**: Three-tier memory system keeps main LLM context lean.

## System Architecture - Visual Overview

### The Complete Flow (Reference the diagram provided)

```
┌─────────────────────────┐         ┌──────────────────────────────────────┐
│   MODEL CONTEXT         │         │   PYTHON REPL                        │
│   (Limited Size)        │         │   (Unlimited Memory)                 │
├─────────────────────────┤         ├──────────────────────────────────────┤
│ • SYSTEM PROMPT         │         │ • INPUT DATA (full, not in LLM)     │
│ • USER PROMPT           │ ──────> │ • EXECUTE CODE                       │
│ • REASONING             │ ANALYZE │   - Access input_data                │
│ • CALL_PYTHON_REPL ────>│  DATA   │   - Call llm_batch()  ─────────────> │
│                         │         │   - Update answer variable           │
│                         │         │                                      │
│ • REPL OUTPUT    <──────│ PRINT   │ • PARALLEL SUB-LLMS:                │
│ • REASONING             │ OUTPUTS │   ├─> PROMPT 1 → REASONING → ANSWER 1│
│ • CALL_PYTHON_REPL ────>│ FINISH  │   ├─> PROMPT 2 → REASONING → ANSWER 2│
│                         │         │   └─> PROMPT 3 → REASONING → ANSWER 3│
│                         │         │                      ↓                │
│                         │         │ • EXECUTE CODE (collect results)     │
│                         │         │ • FROM VARIABLE ────> FINAL ANSWER   │
└─────────────────────────┘         └──────────────────────────────────────┘
```

### Key Observations from the Architecture:

**LEFT SIDE (Model Context)**: 
- Contains: System prompt, user prompt, reasoning, REPL outputs
- **Limited size** - must stay within context window
- Cycles through: Reasoning → Call REPL → Get Output → Reasoning → Repeat
- Never directly sees the input data!

**RIGHT SIDE (Python REPL)**:
- Contains: Full input data, code execution environment, sub-LLM spawning
- **Unlimited size** - can store gigabytes of data
- Main LLM sends "analyze data" command → REPL has the data → processes it
- Returns only truncated print outputs back to Model Context

**PARALLEL SUB-LLMS**:
- Spawned from within the REPL via `llm_batch()`
- Each has independent, fresh context (no context rot)
- Process tasks in parallel: PROMPT → REASONING → TOOL_CALL → REASONING → ANSWER
- Results collected in REPL, then summarized back to main model

**THE MEMORY TRICK**:
- Main model's context stays small (just prompts + truncated outputs)
- All heavy data lives in REPL (persistent across turns)
- Sub-LLMs process chunks in parallel (distributed memory)
- Final answer built in `answer` variable, retrieved at end

### Components to Build

#### 1. **Main RLM Controller**
**Purpose**: The "brain" that orchestrates everything

**What it manages**:
- The main LLM loop (think → code → execute → think → repeat)
- The special `answer` variable: `{"content": "", "ready": False}`
- Conversation history (but keeps it compressed)
- Iteration counting (stop after N iterations)
- Decides when task is complete

**Why this creates memory**:
- Maintains conversation thread across iterations
- Keeps answer variable persistent
- Decides what to remember vs. forget

#### 2. **Python REPL Executor**
**Purpose**: Acts as the LLM's "extended RAM"

**What it does**:
- Executes Python code the LLM generates
- Maintains a persistent namespace (variables survive across turns)
- Provides access to `input_data` (the big data LLM doesn't see directly)
- Injects `llm_batch()` function for sub-LLM calls
- Truncates output to prevent context overflow

**Why this creates memory**:
- Variables persist across turns (real memory)
- Can store gigabytes of data outside LLM context
- Acts like a computer's RAM for the LLM's "processor"

#### 3. **Sub-LLM Manager**
**Purpose**: Handles parallel delegation to "worker" LLMs

**What it does**:
- Takes list of prompts from main LLM
- Spawns multiple LLM calls in parallel
- Collects and returns results
- Tracks token usage separately

**Why this enables long context**:
- Each sub-LLM has fresh context (no rot)
- Process 100 documents with 100 sub-LLMs simultaneously
- Main LLM only sees compressed summaries, not raw data

#### 4. **Generic LLM Client**
**Purpose**: Talk to any LLM API (model-agnostic)

**Supports**:
- OpenRouter (Llama, Qwen, DeepSeek, etc.)
- Ollama (local models)
- OpenAI-compatible APIs
- Together AI, etc.

**Configuration**: User provides API endpoint, model name, API key

### The Critical Flow Pattern (From Your Diagram)

**ITERATION 1:**
```
Model Context:
  [SYSTEM PROMPT] → [USER PROMPT] → [REASONING]
  ↓
  Generates: [CALL_PYTHON_REPL] with code "analyze data"
  ↓
Python REPL:
  - Has [INPUT DATA] already loaded
  - [EXECUTE CODE] runs the analysis
  - Spawns [PARALLEL SUB-LLMS] if needed:
      PROMPT 1 → REASONING → TOOL_CALL → REASONING → ANSWER 1
      PROMPT 2 → REASONING → ANSWER 2
      PROMPT 3 → REASONING → TOOL_CALL → REASONING → ANSWER 3
  - Collects results
  ↓
  Returns: [PRINT OUTPUTS] (truncated to 4096 chars)
  ↓
Model Context:
  Receives [REPL OUTPUT]
  ↓
  [REASONING] about the results
```

**ITERATION 2:**
```
Model Context:
  [REASONING] continues: "Good, now I'll finalize"
  ↓
  Generates: [CALL_PYTHON_REPL] with code "answer['ready'] = True"
  ↓
Python REPL:
  - [EXECUTE CODE] sets ready flag
  - [FROM VARIABLE] extracts answer["content"]
  ↓
  Returns: [FINAL ANSWER]
  ↓
Task Complete!
```

### Why This Architecture Solves Long Context

**Problem 1: Context Overflow**
```
❌ Normal LLM: 
   Input (1M tokens) → Model Context → OVERFLOW!

✅ RLM:
   Input (1M tokens) → REPL Storage → Model never sees it
   Model sees: "input_data has 1M tokens" (5 tokens!)
```

**Problem 2: Context Rot**
```
❌ Normal LLM:
   Turn 1: [128k tokens context] → accuracy 90%
   Turn 5: [128k tokens context] → accuracy 60% (rot!)

✅ RLM:
   Turn 1: [8k tokens context] → REPL has rest → accuracy 90%
   Turn 5: [8k tokens context] → REPL has rest → accuracy 90% (no rot!)
```

**Problem 3: No Parallelization**
```
❌ Normal LLM:
   Process 1000 docs sequentially → 1000 LLM calls → slow

✅ RLM:
   llm_batch([1000 prompts]) → 1000 parallel calls → 100x faster
```

**Problem 4: No Persistent Memory**
```
❌ Normal LLM:
   Turn 1: "Hypothesis: X"
   Turn 2: [Forgets X] Starts over

✅ RLM:
   Turn 1: answer["hypothesis"] = "X"
   Turn 2: print(answer["hypothesis"]) → "X" → Builds on it
```

### Data Flow & Memory Management

```
┌─────────────────────────────────────────────────────────────┐
│  USER INPUT                                                  │
│  - Prompt: "Summarize these 1000 documents"                 │
│  - Input Data: [1000 documents, 50MB of text]              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  MAIN LLM (Small Context - 8k tokens)                       │
│  Sees:                                                       │
│  - Task prompt                                              │
│  - System instructions about RLM                            │
│  - Info that input_data has 1000 documents                 │
│  - Previous REPL output (truncated)                        │
│                                                             │
│  Thinks: "I'll split into 100 chunks, process parallel"   │
│  Generates Python code ───────────────────────┐            │
└───────────────────────────────────────────────┼────────────┘
                                                │
                                                ↓
┌───────────────────────────────────────────────┴────────────┐
│  PYTHON REPL (Unlimited Memory)                            │
│  Contains:                                                  │
│  - input_data = [all 1000 documents]  ← Full data here    │
│  - answer = {"content": "", "ready": False}                │
│  - llm_batch = <function>                                  │
│  - Any variables from previous turns                       │
│                                                             │
│  Executes:                                                  │
│    chunks = [input_data[i:i+10] for i in range(0,1000,10)]│
│    summaries = llm_batch([                                 │
│      f"Summarize: {chunk}" for chunk in chunks             │
│    ]) ────────────────────────────────┐                    │
│    answer["content"] = merge(summaries)│                   │
└────────────────────────────────────────┼───────────────────┘
                                         │
                    ┌────────────────────┴────────────┐
                    ↓                    ↓            ↓
            ┌───────────┐      ┌───────────┐   ┌───────────┐
            │ Sub-LLM 1 │      │ Sub-LLM 2 │...│ Sub-LLM 100│
            │ (Fresh)   │      │ (Fresh)   │   │ (Fresh)    │
            │ Sees:     │      │ Sees:     │   │ Sees:      │
            │ - Doc 1-10│      │ - Doc11-20│   │ - Doc991-  │
            │           │      │           │   │   1000     │
            │ Returns:  │      │ Returns:  │   │ Returns:   │
            │ "Summary1"│      │ "Summary2"│   │ "Summary100"│
            └─────┬─────┘      └─────┬─────┘   └──────┬─────┘
                  │                  │                 │
                  └──────────────────┴─────────────────┘
                                     ↓
                    [100 summaries returned to REPL]
                                     ↓
┌─────────────────────────────────────────────────────────────┐
│  REPL RESULT (Truncated to 4096 chars)                      │
│  "Successfully processed 1000 docs → 100 summaries          │
│   answer['content'] now contains merged summary"            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  MAIN LLM (Next Turn)                                        │
│  Sees: "Successfully processed... 100 summaries"            │
│  Thinks: "Good! Now I'll synthesize final answer"           │
│  Generates:                                                  │
│    print(answer["content"][:500])  # Review what I wrote   │
│    answer["ready"] = True          # Done!                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
                   [Task Complete]
              answer["content"] returned to user
```

### Why This Architecture Enables Long Context

**1. Main LLM never sees raw data**
- Input data lives in REPL only
- Main LLM gets metadata: "input_data is 50MB, 1000 documents"
- Prevents context window overflow

**2. Sub-LLMs are ephemeral**
- Created, used, destroyed
- No context accumulation
- Each has fresh, clean context

**3. Memory hierarchy**:
```
Fast, Small (Main LLM context):
  - Current reasoning
  - Last REPL output
  - answer state
  → Updated every turn

Medium, Persistent (REPL namespace):
  - All variables
  - Full input data
  - Computation results
  → Survives across turns

Slow, Distributed (Sub-LLMs):
  - Process chunks
  - Return summaries
  - Garbage collected
  → Created on demand
```

## Specific Instructions for Implementation

### 1. API Integration (Model Agnostic)

**Requirements**:
- Support any OpenAI-compatible API endpoint
- User configures: `api_base`, `model_name`, `api_key`
- Use standard `/chat/completions` endpoint
- Parse responses generically

**Example configurations to support**:
```
# OpenRouter
api_base: https://openrouter.ai/api/v1
model: meta-llama/llama-3.1-70b-instruct

# Ollama (local)
api_base: http://localhost:11434/v1
model: llama3.1:70b

# Together AI
api_base: https://api.together.xyz/v1
model: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
```

### 2. Python REPL Safety (No Docker Needed)

**Use simple approach**:
- Python's built-in `exec()` with restricted namespace
- Whitelist safe built-in functions
- Whitelist safe imports (json, re, math, etc.)
- Run with timeout
- This is a POC - full security not required

**Memory persistence**:
```python
# Namespace persists across turns
namespace = {
    "input_data": user_input_data,
    "answer": {"content": "", "ready": False},
    "llm_batch": sub_llm_function
}

# Turn 1
exec(code_from_llm_turn1, namespace)
# namespace now has any variables LLM created

# Turn 2  
exec(code_from_llm_turn2, namespace)
# Can access variables from turn 1!
```

### 3. The System Prompt (Critical!)

**Must explain to the LLM**:

```
You are a Recursive Language Model (RLM). You solve tasks by:

1. MANAGING YOUR OWN MEMORY
   - Your context window is limited
   - Input data is NOT in your context - it's in input_data variable
   - Use Python to access and process it

2. THE ANSWER VARIABLE (Your Persistent Memory)
   answer = {"content": "", "ready": False}
   
   - Build your answer in answer["content"]
   - Modify it as many times as needed
   - Only set answer["ready"] = True when completely done

3. SUB-LLM DELEGATION (Scale Your Thinking)
   results = llm_batch([
       "prompt 1",
       "prompt 2",
       ...
   ])
   
   - Use sub-LLMs to process data chunks in parallel
   - They see only what you send them
   - They return compressed summaries
   - Keeps your context clean

4. ITERATIVE REFINEMENT
   - You get multiple turns
   - Build answer incrementally
   - Review and improve your work
   - Use print() to see what you've written

5. REPL OUTPUT IS TRUNCATED
   - Only first 4096 chars shown to you
   - Design code to print concisely
   - Store full data in variables, not print output

YOUR TASK: {user_prompt}

INPUT DATA: Available in 'input_data' variable
  - Size: {len(input_data)} characters
  - Preview: {input_data[:200]}...

Begin by writing Python code to solve this task.
```

### 4. The Main Loop Logic

**Iteration cycle**:
```
Initialize:
  - answer = {"content": "", "ready": False}
  - namespace = {input_data, answer, llm_batch}
  - messages = []
  - iteration = 0

Loop while iteration < max_iterations and not answer["ready"]:
  1. Call main LLM with current messages + system prompt
  
  2. If LLM generated Python code:
     a. Extract code from markdown
     b. Execute in namespace (with timeout)
     c. Capture output (truncate to 4096 chars)
     d. Update answer from namespace
     e. Add LLM response to messages
     f. Add REPL output to messages as user message
  
  3. Else (no code, just reasoning):
     a. Add to messages
     b. Prompt: "Write Python code to make progress"
  
  4. iteration++

Return answer["content"]
```

### 5. Sub-LLM Implementation

**The llm_batch function injected into REPL**:
```
def llm_batch(prompts: list[str]) -> list[str]:
    """
    Process multiple prompts in parallel using sub-LLMs
    
    Args:
        prompts: List of prompts to send to fresh LLM instances
        
    Returns:
        List of responses in same order as prompts
    """
    # Create async tasks for each prompt
    # Each calls same LLM API with fresh context
    # Return results in order
```

**Why this enables scaling**:
- Main LLM: "I need to process 1000 docs"
- Generates: `summaries = llm_batch([f"Summarize: {doc}" for doc in docs])`
- System spawns 1000 parallel LLM calls
- Each sub-LLM has tiny, focused context
- Main LLM receives 1000 summaries (compressed)

### 6. Memory Management Strategy

**Context Budget Allocation**:
```
Main LLM Context (8k tokens total):
  - System prompt: ~1k tokens
  - Task description: ~500 tokens
  - Last REPL output: ~2k tokens (truncated)
  - Conversation history: ~3k tokens (keep recent only)
  - Buffer: ~1.5k tokens

REPL Memory (unlimited):
  - input_data: [any size]
  - answer: [grows incrementally]
  - All user variables: [any size]

Sub-LLM Context (4k tokens each):
  - Just their specific task
  - Fresh slate each time
```

**History Compression**:
- Keep last N turns only (N=5 for POC)
- Or keep only: first turn + last 3 turns
- Main LLM sees: "... [history compressed] ... [recent turns]"

## Example Use Cases to Test

### Test 1: Long Context Summarization
```
Prompt: "Summarize this article"
Input Data: [50 page article, 30k words]

Expected behavior:
  Turn 1: LLM chunks article into 10 parts
  Turn 1: Calls llm_batch with 10 chunks
  Turn 2: Receives 10 summaries
  Turn 2: Merges into final summary
  Turn 2: Sets answer["ready"] = True
```

### Test 2: Data Analysis
```
Prompt: "Count how many times 'AI' appears"
Input Data: [100 documents]

Expected behavior:
  Turn 1: LLM writes: count = sum(1 for doc in input_data if 'AI' in doc)
  Turn 1: Updates answer["content"] = f"Found {count} occurrences"
  Turn 1: Sets answer["ready"] = True
```

### Test 3: Iterative Refinement
```
Prompt: "Write a poem about space, then improve it"
Input Data: ""

Expected behavior:
  Turn 1: answer["content"] = "First draft: Stars shine bright..."
  Turn 2: print(answer["content"]) → reviews it
  Turn 2: answer["content"] = "Improved version: Cosmic lights..."
  Turn 3: answer["ready"] = True
```

## Success Criteria

Your implementation is successful when:

1. **Context stays small**: Main LLM context never exceeds reasonable limit, even with huge input data

2. **Memory persists**: Variables survive across turns - LLM can build on previous work

3. **Parallelization works**: Multiple sub-LLMs can be called simultaneously

4. **Iterative refinement**: LLM can review and improve its answer over multiple turns

5. **Model agnostic**: Works with any OpenAI-compatible API endpoint

6. **Observable**: Clear logging shows LLM thinking, REPL execution, sub-LLM calls

## Technical Constraints

- **No Docker required** - Use simple Python exec()
- **No specific model required** - Support any API
- **POC quality** - Focus on core concept, not production security
- **Async support** - For parallel sub-LLM calls
- **Simple web UI** - Basic interface to test the system

## Deliverables

Build:
1. **Backend**: Python FastAPI server with RLM controller
2. **Frontend**: React UI to input prompt/data and see execution
3. **README**: How to configure API, run examples
4. **Examples**: 3-4 demo cases showing different capabilities

The goal is to **prove the concept works** - that an LLM can manage its own context and memory through code and delegation.