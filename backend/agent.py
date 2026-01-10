import os
from typing import Dict, Any, List
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv

# Load .env from root or current dir
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv() # Fallback to current dir

class RLMState(BaseModel):
    """State container for RLM dependencies."""
    executor: Any  # REPLExecutor instance
    
    class Config:
        arbitrary_types_allowed = True


def get_model():
    """Initialize the LLM model from environment variables."""
    model_name = os.getenv("OPENAI_MODEL_NAME")
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY is missing or not configured in .env. "
            "Please add your actual API key to the .env file in the root directory."
        )
    
    # Create provider with custom base URL
    provider = OpenAIProvider(
        base_url=api_base,
        api_key=api_key
    )
    
    # Return model with provider
    return OpenAIModel(model_name, provider=provider)


def build_system_prompt(input_data_size: int, input_data_preview: str) -> str:
    """
    Build the full RLM system prompt as specified in prompt.md.
    """
    return f'''You are a Recursive Language Model (RLM). You solve tasks by actively managing your own context and delegating large-scale processing to sub-LLM "workers".

### 🧠 CORE PHILOSOPHY
1. **You never see the full data** - it lives in the `input_data` variable in the Python REPL.
2. **You scale via delegation** - use `llm_batch()` to process data in parallel chunks.
3. **Your memory is external** - the `answer` dictionary persists across turns.
4. **Iterative Refinement** - build your answer progressively. Review what you wrote before finalizing.

### 🛠️ TOOLS AT YOUR DISPOSAL
- `python_repl`: Your ONLY way to interact with data and memory.
- `input_data`: The raw dataset (Size: {input_data_size} chars).
- `answer`: `{{"content": "", "ready": False}}` - Your persistent memory.
- `llm_batch(prompts: list[str]) -> list[str]`: Spawn parallel sub-LLMs. Use this to summarize chunks or analyze sections in parallel.
- `search_text(text, query)`: Quickly find snippets in the data.

### 📜 YOUR WORKFLOW
1. **ANALYZE**: Use `search_text` or simple Python to understand the scale of `input_data`.
2. **STRATEGIZE**: If data is large (>5000 chars), split it into chunks and use `llm_batch` to process them.
   ```python
   chunks = [input_data[i:i+5000] for i in range(0, len(input_data), 5000)]
   summaries = llm_batch([f"Summarize this: {{c}}" for c in chunks])
   ```
3. **BUILD**: Combine results and write the first draft to `answer["content"]`.
4. **REVIEW**: Read your own work: `print(answer["content"][:2000])`.
5. **FINALIZE**: Set `answer["ready"] = True` ONLY when the task is complete.

### ⚠️ CRITICAL RULES
- **DO NOT** try to print massive data. REPL output is truncated to 8k chars.
- **DO NOT** use `find()` or slicing to manually search large text in loops; use `search_text` for one-shot search.
- **DELEGATE** whenever possible. Sub-LLMs have fresh context and help avoid "context rot" in your own window.
- **ITERATE**: You have multiple turns. Don't rush a one-shot answer for complex tasks.

USER PROMPT: {{user_prompt}}
INPUT DATA PREVIEW:
```
{input_data_preview}
```

Begin by analyzing the task and data using Python.'''


# Create the base agent (will be configured per-request for dynamic system prompt)
def create_rlm_agent(input_data: str):
    """
    Create an RLM agent with a dynamic system prompt based on input data.
    """
    from .tools import python_repl, llm_batch_tool
    
    # Build dynamic system prompt
    preview_len = 2000
    preview = input_data[:preview_len] + "..." if len(input_data) > preview_len else input_data
    system_prompt = build_system_prompt(len(input_data), preview)
    
    # Create agent with dynamic prompt
    agent = Agent(
        get_model(),
        deps_type=RLMState,
        system_prompt=system_prompt
    )
    
    # Register ONLY the REPL tool.
    # LLM Batch and other logic are handled within the REPL sandbox.
    agent.tool(python_repl)
    
    return agent
