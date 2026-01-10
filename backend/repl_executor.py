import io
import contextlib
import traceback
import sys
from typing import Any, Dict, Callable
from copy import deepcopy

# Import safe standard library modules
import math
import json
import re
import random
import string
import datetime
import collections
import itertools
import functools
import statistics
import hashlib
import base64
import urllib.parse
import textwrap
import difflib
import csv

# Whitelist of safe built-in functions
SAFE_BUILTINS = {
    # Type constructors
    'bool': bool, 'int': int, 'float': float, 'str': str,
    'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
    'frozenset': frozenset, 'bytes': bytes, 'bytearray': bytearray,
    
    # Math operations
    'abs': abs, 'divmod': divmod, 'pow': pow, 'round': round,
    'min': min, 'max': max, 'sum': sum,
    
    # Iteration/Sequence
    'len': len, 'range': range, 'enumerate': enumerate,
    'zip': zip, 'map': map, 'filter': filter,
    'sorted': sorted, 'reversed': reversed,
    'all': all, 'any': any, 'next': next, 'iter': iter,
    'slice': slice,
    
    # String/Repr
    'repr': repr, 'format': format, 'chr': chr, 'ord': ord,
    'bin': bin, 'hex': hex, 'oct': oct, 'ascii': ascii,
    
    # Object inspection
    'type': type, 'isinstance': isinstance, 'issubclass': issubclass,
    'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr,
    'dir': dir, 'vars': vars, 'id': id, 'hash': hash,
    'callable': callable,
    
    # I/O
    'print': print, 'input': lambda *args: "[input() disabled in sandbox]",
    
    # Constants
    'True': True, 'False': False, 'None': None,
    
    # Misc utilities
    'open': None,  # Disabled - no file access
    'exec': None,  # Disabled - no nested exec
    'eval': None,  # Disabled - no eval
    '__import__': None,  # Disabled - no imports
}

# Safe imports with useful modules for data processing
try:
    import numpy
    import scipy
    import sympy
except ImportError:
    numpy = None
    scipy = None
    sympy = None

SAFE_IMPORTS = {
    # Math & Statistics
    'math': math,
    'statistics': statistics,
    'random': random,
    
    # Scientific Stack
    'numpy': numpy,
    'scipy': scipy,
    'sympy': sympy,
    
    # Data formats
    'json': json,
    're': re,
    'csv': csv,
    'base64': base64,
    
    # Data structures
    'collections': collections,
    'itertools': itertools,
    'functools': functools,
    
    # String utilities
    'string': string,
    'textwrap': textwrap,
    'difflib': difflib,
    
    # Date/Time
    'datetime': datetime,
    
    # Encoding/Hashing
    'hashlib': hashlib,
    'urllib': urllib,  # urllib.parse for URL manipulation
}


class SandboxedREPLExecutor:
    """
    A sandboxed Python REPL executor with persistent namespace.
    
    Security features:
    - Restricted builtins (no file I/O, no exec/eval, no imports)
    - Whitelisted safe standard library modules
    - Isolated namespace per session
    - Output truncation to prevent memory issues
    
    This acts as the LLM's "extended RAM" - variables persist across turns,
    allowing the LLM to build and refine its answer incrementally.
    """
    
    def __init__(self, initial_namespace: Dict[str, Any] = None, llm_batch_func: Callable = None):
        """
        Initialize a new sandboxed REPL executor.
        
        Args:
            initial_namespace: Variables to inject (e.g., input_data, answer)
            llm_batch_func: The llm_batch function for parallel LLM calls
        """
        # Build the sandboxed namespace
        self.namespace = {
            '__builtins__': SAFE_BUILTINS.copy(),
            '__name__': '__rlm_sandbox__',
        }
        
        # Add safe imports
        self.namespace.update(SAFE_IMPORTS)
        
        # Add initial namespace (input_data, answer, etc.)
        if initial_namespace:
            self.namespace.update(initial_namespace)
        
        # Ensure answer variable exists
        if "answer" not in self.namespace:
            self.namespace["answer"] = {"content": "", "ready": False}
        
        # Inject llm_batch if provided
        if llm_batch_func:
            self.namespace["llm_batch"] = llm_batch_func
        
        # Add helper functions
        self._inject_helpers()

    def _inject_helpers(self):
        """Inject useful helper functions into the namespace."""
        
        def chunk_text(text: str, chunk_size: int = 1000) -> list:
            """Split text into chunks of approximately chunk_size characters."""
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        def chunk_list(lst: list, chunk_size: int = 10) -> list:
            """Split a list into chunks of chunk_size items."""
            return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
        
        def word_count(text: str) -> int:
            """Count words in text."""
            return len(text.split())
        
        def unique_words(text: str) -> set:
            """Get unique words from text (lowercase)."""
            return set(re.findall(r'\b\w+\b', text.lower()))
        
        def summarize_dict(d: dict, max_items: int = 5) -> str:
            """Get a summary of a dictionary structure."""
            keys = list(d.keys())[:max_items]
            return f"Dict with {len(d)} keys. First {max_items}: {keys}"
        
        def search_text(text: Any, query: Any = None, context: int = 200, start: int = 0, **kwargs) -> str:
            """
            Search for all occurrences of a query in text starting from 'start' index.
            Returns snippets with 'context' characters on each side.
            Available globally - DO NOT IMPORT.
            """
            if text is None:
                return "Error: input text is None"
            if query is None:
                return "Error: search query is None. Please provide a string to search for."
            
            text = str(text)
            query = str(query)
            
            # Handle aliases for parameters
            ctx = kwargs.get('context_length', context)
            idx = kwargs.get('start_index', start)
            
            results = []
            search_area = text[idx:]
            try:
                for match in re.finditer(re.escape(query), search_area, re.IGNORECASE):
                    abs_start = max(0, (match.start() + idx) - ctx)
                    abs_end = min(len(text), (match.end() + idx) + ctx)
                    snippet = text[abs_start:abs_end].replace('\n', ' ')
                    results.append(f"Match at index {match.start() + idx}: ...{snippet}...")
            except Exception as e:
                return f"Error during search: {str(e)}"
            
            if not results:
                return f"No matches found for '{query}'"
            
            return f"Found {len(results)} matches for '{query}':\n\n" + "\n\n".join(results[:10])
        
        self.namespace.update({
            'chunk_text': chunk_text,
            'chunk_list': chunk_list,
            'word_count': word_count,
            'unique_words': unique_words,
            'summarize_dict': summarize_dict,
            'search_text': search_text,
        })

    def execute(self, code: str, output_limit: int = 8192, timeout: int = 120) -> str:
        """
        Execute Python code in the isolated, persistent namespace.
        
        Args:
            code: Python code to execute (may include markdown fences)
            output_limit: Maximum characters to return (default 8192 as per Prime Intellect spec)
            timeout: Maximum seconds for code execution (default 120 as per Prime Intellect spec)
            
        Returns:
            Captured stdout output, truncated if necessary
        """
        import concurrent.futures
        
        f = io.StringIO()
        clean_code = self._clean_code(code)
        
        # Compile first to catch syntax errors before executing
        try:
            compiled = compile(clean_code, '<rlm_sandbox>', 'exec')
        except SyntaxError as e:
            return f"SyntaxError: {e}"
        
        # Define the execution function
        def run_exec():
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                try:
                    exec(compiled, self.namespace)
                except NameError as e:
                    print(f"NameError: {e}")
                except TypeError as e:
                    print(f"TypeError: {e}")
                except Exception as e:
                    print(f"Error ({type(e).__name__}): {e}")
                    print(traceback.format_exc())
        
        # Execute with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            try:
                future = pool.submit(run_exec)
                future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                f.write(f"\n\n[TIMEOUT] Code execution exceeded {timeout}s limit.")
        
        output = f.getvalue()
        if not output:
            output = "[Code executed successfully - no print output]"
        
        # Truncate output to prevent context overflow (critical for RLM)
        if len(output) > output_limit:
            truncated_msg = f"\n\n... [OUTPUT TRUNCATED - showing {output_limit}/{len(output)} chars]"
            output = output[:output_limit - len(truncated_msg)] + truncated_msg
            
        return output
    
    def _clean_code(self, code: str) -> str:
        """Remove markdown code fences if present."""
        clean = code.strip()
        
        # Handle ```python ... ``` blocks
        if clean.startswith("```python"):
            clean = clean[9:]
        elif clean.startswith("```py"):
            clean = clean[5:]
        elif clean.startswith("```"):
            clean = clean[3:]
            
        if clean.endswith("```"):
            clean = clean[:-3]
            
        return clean.strip()

    def get_answer(self) -> Dict[str, Any]:
        """Get the current state of the answer variable."""
        return self.namespace.get("answer", {"content": "", "ready": False})
    
    def get_variable(self, name: str) -> Any:
        """Get any variable from the namespace."""
        return self.namespace.get(name)
    
    def set_variable(self, name: str, value: Any):
        """Set a variable in the namespace."""
        self.namespace[name] = value
    
    def get_namespace_info(self) -> Dict[str, Any]:
        """Get info about what's in the namespace (for debugging)."""
        user_vars = {}
        for k, v in self.namespace.items():
            if not k.startswith('__') and k not in SAFE_IMPORTS and k not in SAFE_BUILTINS:
                user_vars[k] = type(v).__name__
        return {
            "user_variables": user_vars,
            "available_modules": list(SAFE_IMPORTS.keys()),
            "helper_functions": ['chunk_text', 'chunk_list', 'word_count', 'unique_words', 'summarize_dict', 'search_text', 'llm_batch']
        }


# Backwards compatibility alias
REPLExecutor = SandboxedREPLExecutor
