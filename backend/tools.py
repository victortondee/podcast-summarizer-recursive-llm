import asyncio
import httpx
import os
from typing import List, Dict, Any
from pydantic_ai import RunContext
from dotenv import load_dotenv

# Load .env from root or current dir
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv() # Fallback to current dir

# We will inject the REPL executor into the agent's context
async def python_repl(ctx: RunContext[Any], code: str) -> str:
    """
    Execute Python code in a persistent REPL namespace.
    
    The namespace contains:
    - input_data: The full input data (you never see this directly in your context)
    - answer: Dict with 'content' (str) and 'ready' (bool) - your persistent memory
    - llm_batch: Function to call sub-LLMs in parallel
    - math, json, re: Safe standard library modules
    
    Use print() to see outputs. Output is truncated to 4096 chars.
    Store important data in variables, not in print output.
    
    Args:
        code: Python code to execute. Can include markdown fences.
        
    Returns:
        The printed output from code execution.
    """
    executor = ctx.deps.executor
    return executor.execute(code)


def create_llm_batch_function():
    """
    Factory function to create the llm_batch callable.
    This is called synchronously by the REPL but handles async internally.
    """
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL_NAME")
    
    async def _async_call_llm(prompt: str, client: httpx.AsyncClient) -> str:
        """Make a single LLM call."""
        try:
            response = await client.post(
                f"{api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,  # Sub-LLMs get smaller context
                },
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except httpx.HTTPStatusError as e:
            return f"[Sub-LLM HTTP Error: {e.response.status_code}]"
        except Exception as e:
            return f"[Sub-LLM Error: {str(e)}]"
    
    async def _async_batch(prompts: List[str]) -> List[str]:
        """Process all prompts in parallel."""
        async with httpx.AsyncClient() as client:
            tasks = [_async_call_llm(p, client) for p in prompts]
            return await asyncio.gather(*tasks)
    
    def llm_batch(prompts: List[str]) -> List[str]:
        """
        Process multiple prompts in parallel using fresh sub-LLM instances.
        
        Each sub-LLM has a clean context (no history), preventing context rot.
        Use this to:
        - Process large datasets in chunks
        - Get multiple perspectives on a problem
        - Parallelize independent subtasks
        
        Args:
            prompts: List of prompts to send to sub-LLMs
            
        Returns:
            List of responses in the same order as prompts
            
        Example:
            chunks = [input_data[i:i+1000] for i in range(0, len(input_data), 1000)]
            summaries = llm_batch([f"Summarize: {chunk}" for chunk in chunks])
        """
        # Handle the case where we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, need to run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _async_batch(prompts))
                return future.result()
        except RuntimeError:
            # No running loop, we can just run directly
            return asyncio.run(_async_batch(prompts))
    
    return llm_batch


async def llm_batch_tool(ctx: RunContext[Any], prompts: List[str]) -> List[str]:
    """
    Process multiple prompts in parallel using sub-LLMs.
    
    Each sub-LLM instance has a fresh, clean context - no history accumulation.
    This prevents context rot and enables horizontal scaling.
    
    Use this when you need to:
    - Process many chunks of data in parallel
    - Get multiple independent analyses
    - Scale beyond your context window
    
    Args:
        prompts: List of prompts to process in parallel
        
    Returns:
        List of responses in the same order as input prompts
    """
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL_NAME")

    if not api_key or api_key == "your_api_key_here":
        return ["Error: OPENAI_API_KEY not configured."]

    async def call_llm(prompt: str) -> str:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2048,
                    },
                    timeout=120.0
                )
                response.raise_for_status()
                data = response.json()
                return data['choices'][0]['message']['content']
            except Exception as e:
                return f"[Sub-LLM Error: {str(e)}]"

    tasks = [call_llm(p) for p in prompts]
    return await asyncio.gather(*tasks)
