import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
from .agent import create_rlm_agent, RLMState
from .repl_executor import REPLExecutor
from .tools import create_llm_batch_function
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Recursive Language Model (RLM) API",
    description="An LLM system that manages its own context through code and delegation",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    prompt: str
    input_data: Optional[str] = ""
    max_iterations: Optional[int] = 5


class ProcessResponse(BaseModel):
    content: str
    history: List[Dict[str, Any]]
    iterations: int
    token_usage: Optional[Dict[str, int]] = None


def compress_history(history: list, keep_first: int = 1, keep_last: int = 3) -> list:
    """
    Compress conversation history to prevent context rot.
    
    As per prompt.md, we keep:
    - First N turns (for context)
    - Last M turns (for recent work)
    - Middle is compressed/dropped
    """
    if len(history) <= keep_first + keep_last:
        return history
    
    first_part = history[:keep_first]
    last_part = history[-keep_last:]
    
    # Add a marker indicating compression
    compression_marker = {
        "role": "system",
        "content": f"[... {len(history) - keep_first - keep_last} earlier turns compressed ...]"
    }
    
    return first_part + [compression_marker] + last_part


@app.post("/process")
async def process_rlm(request: ProcessRequest):
    """
    Process a task using the RLM system with real-time streaming.
    """
    
    async def event_generator():
        # Create the llm_batch function for REPL use
        llm_batch_func = create_llm_batch_function()
        
        # Initialize REPL with all required variables
        executor = REPLExecutor(
            initial_namespace={
                "input_data": request.input_data,
                "answer": {"content": "", "ready": False},
            },
            llm_batch_func=llm_batch_func
        )
        
        # Create agent with dynamic system prompt
        agent = create_rlm_agent(request.input_data)
        deps = RLMState(executor=executor)
        
        history = []
        iterations = 0
        current_prompt = f"TASK: {request.prompt}"
        
        while iterations < request.max_iterations:
            iterations += 1
            
            # Yield iteration start event
            yield f"data: {json.dumps({'type': 'iteration_start', 'iteration': iterations})}\n\n"
            
            # Compress history if it's getting long
            compressed_history = compress_history(history) if len(history) > 6 else history
            
            try:
                # Run agent for one turn
                result = await agent.run(
                    current_prompt, 
                    deps=deps, 
                    message_history=compressed_history
                )
                
                # Extract new messages from this turn to find thoughts and tool calls
                new_messages = result.new_messages()
                for msg in new_messages:
                    if hasattr(msg, 'parts'):
                        for part in msg.parts:
                            kind = getattr(part, 'part_kind', getattr(part, 'kind', ''))
                            
                            if kind == 'text':
                                yield f"data: {json.dumps({'type': 'thought', 'content': part.content if hasattr(part, 'content') else part.text})}\n\n"
                            
                            elif kind == 'tool-call':
                                tool_name = part.tool_name if hasattr(part, 'tool_name') else part.name
                                args = part.args
                                yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'args': args})}\n\n"
                            
                            elif kind in ['tool-return', 'tool_return']:
                                content = part.content if hasattr(part, 'content') else (part.text if hasattr(part, 'text') else part.result)
                                yield f"data: {json.dumps({'type': 'tool_result', 'content': str(content)})}\n\n"

                # Update history
                history = list(result.all_messages())
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                break
            
            # Check if agent marked the answer as ready
            answer = executor.get_answer()
            if answer.get("ready"):
                yield f"data: {json.dumps({'type': 'final_result', 'content': answer.get('content', ''), 'iterations': iterations})}\n\n"
                return
            
            # Prepare prompt for next iteration
            if iterations < request.max_iterations:
                current_prompt = (
                    "The answer is not yet marked as 'ready'. "
                    "Continue processing, refine the answer, or set answer['ready'] = True if done."
                )

        # Max iterations reached
        answer = executor.get_answer()
        final_content = answer.get("content", "") or "[Max iterations reached]"
        yield f"data: {json.dumps({'type': 'final_result', 'content': final_content, 'iterations': iterations})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _serialize_history(history: list) -> List[Dict[str, Any]]:
    """Convert message objects to serializable dictionaries."""
    result = []
    for msg in history:
        if hasattr(msg, 'model_dump'):
            result.append(msg.model_dump())
        elif hasattr(msg, '__dict__'):
            result.append(msg.__dict__)
        elif isinstance(msg, dict):
            result.append(msg)
        else:
            result.append({"content": str(msg)})
    return result


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": os.getenv("OPENAI_MODEL_NAME"),
        "api_base": os.getenv("OPENAI_API_BASE")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
