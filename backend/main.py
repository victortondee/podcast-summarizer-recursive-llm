import os
import secrets
import sqlite3
import hashlib
import hmac
import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from fastapi import Cookie, Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
from .agent import create_rlm_agent, RLMState
from .repl_executor import REPLExecutor
from .tools import create_llm_batch_function
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DB_PATH = Path(os.getenv("RLM_DB_PATH", str(Path(__file__).resolve().parent / "rlm_auth.db")))

load_dotenv(BASE_DIR / ".env")

app = FastAPI(
    title="Recursive Language Model (RLM) API",
    description="An LLM system that manages its own context through code and delegation",
    version="1.0.0"
)

SESSION_COOKIE_NAME = "rlm_session"
SESSION_DURATION_DAYS = 14
INITIAL_CREDITS = 10
DEFAULT_PURCHASE_CREDITS = 10

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
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


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class PurchaseCreditsRequest(BaseModel):
    credits: Optional[int] = DEFAULT_PURCHASE_CREDITS


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                password_hash TEXT,
                picture TEXT,
                google_sub TEXT UNIQUE,
                credits INTEGER NOT NULL DEFAULT 10,
                questions_used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_login_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)"
        )
        _ensure_user_columns(conn)


def _ensure_user_columns(conn: sqlite3.Connection) -> None:
    cols = {
        row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()
    }
    if "password_hash" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    if "name" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN name TEXT")
    if "picture" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN picture TEXT")
    if "google_sub" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN google_sub TEXT")
    if "credits" not in cols:
        conn.execute(f"ALTER TABLE users ADD COLUMN credits INTEGER NOT NULL DEFAULT {INITIAL_CREDITS}")
    if "questions_used" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN questions_used INTEGER NOT NULL DEFAULT 0")
    conn.execute(
        f"UPDATE users SET credits = {INITIAL_CREDITS} WHERE credits IS NULL"
    )
    conn.execute(
        "UPDATE users SET questions_used = 0 WHERE questions_used IS NULL"
    )


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return f"{base64.b64encode(salt).decode()}:{base64.b64encode(digest).decode()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_b64, digest_b64 = stored_hash.split(":")
        salt = base64.b64decode(salt_b64.encode())
        expected = base64.b64decode(digest_b64.encode())
    except Exception:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return hmac.compare_digest(actual, expected)


def _touch_user_login(user_id: int) -> None:
    now = _to_iso(_utcnow())
    with _get_db() as conn:
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, user_id))


def _consume_credit_or_raise(user_id: int) -> int:
    with _get_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT credits FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="User account not found")

        credits = int(row["credits"] or 0)
        if credits <= 0:
            raise HTTPException(
                status_code=403,
                detail={"message": "No credits remaining", "credits_remaining": 0},
            )

        remaining = credits - 1
        conn.execute(
            """
            UPDATE users
            SET credits = ?, questions_used = COALESCE(questions_used, 0) + 1
            WHERE id = ?
            """,
            (remaining, user_id),
        )
        return remaining


def _add_credits(user_id: int, credits_to_add: int) -> Dict[str, Any]:
    if credits_to_add <= 0:
        raise HTTPException(status_code=400, detail="credits must be greater than 0")
    if credits_to_add > 1000:
        raise HTTPException(status_code=400, detail="credits purchase too large")

    with _get_db() as conn:
        conn.execute(
            """
            UPDATE users
            SET credits = COALESCE(credits, 0) + ?
            WHERE id = ?
            """,
            (credits_to_add, user_id),
        )
        row = conn.execute(
            "SELECT id, email, name, picture, credits, questions_used FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return dict(row)


def _register_user(email: str, password: str, name: Optional[str]) -> Dict[str, Any]:
    normalized_email = email.strip().lower()
    if not normalized_email or "@" not in normalized_email:
        raise HTTPException(status_code=400, detail="Valid email is required")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    now = _to_iso(_utcnow())
    pwd_hash = _hash_password(password)
    with _get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE lower(email) = lower(?)", (normalized_email,)
        ).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="Account already exists")

        conn.execute(
            """
            INSERT INTO users (email, name, password_hash, google_sub, credits, questions_used, created_at, last_login_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (normalized_email, name, pwd_hash, f"local:{normalized_email}", INITIAL_CREDITS, 0, now, now),
        )
        row = conn.execute(
            "SELECT id, email, name, picture, credits, questions_used FROM users WHERE lower(email) = lower(?)",
            (normalized_email,),
        ).fetchone()
        return dict(row) if row else {}


def _authenticate_user(email: str, password: str) -> Dict[str, Any]:
    normalized_email = email.strip().lower()
    if not normalized_email or "@" not in normalized_email:
        raise HTTPException(status_code=400, detail="Valid email is required")
    with _get_db() as conn:
        row = conn.execute(
            """
            SELECT id, email, name, picture, credits, questions_used, password_hash
            FROM users
            WHERE lower(email) = lower(?)
            """,
            (normalized_email,),
        ).fetchone()

    if not row or not row["password_hash"] or not _verify_password(password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    _touch_user_login(row["id"])
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "picture": row["picture"],
        "credits": row["credits"],
        "questions_used": row["questions_used"],
    }


def _create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(48)
    now = _utcnow()
    expires = now + timedelta(days=SESSION_DURATION_DAYS)
    with _get_db() as conn:
        conn.execute(
            """
            INSERT INTO sessions (token, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (token, user_id, _to_iso(now), _to_iso(expires)),
        )
    return token


def _delete_session(token: str) -> None:
    with _get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def _get_user_by_session(token: str) -> Optional[Dict[str, Any]]:
    now_iso = _to_iso(_utcnow())
    with _get_db() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.email, u.name, u.picture, u.credits, u.questions_used
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ? AND s.expires_at > ?
            """,
            (token, now_iso),
        ).fetchone()
        return dict(row) if row else None


def require_user(
    session_token: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> Dict[str, Any]:
    if not session_token:
        raise HTTPException(status_code=401, detail="Authentication required")
    user = _get_user_by_session(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


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
async def process_rlm(request: ProcessRequest, user: Dict[str, Any] = Depends(require_user)):
    """
    Process a task using the RLM system with real-time streaming.
    """
    
    remaining_credits = _consume_credit_or_raise(user["id"])
    user["credits"] = remaining_credits
    user["questions_used"] = int(user.get("questions_used") or 0) + 1

    async def event_generator():
        # Create the llm_batch function for REPL use
        llm_batch_func = create_llm_batch_function()
        
        # Initialize REPL with all required variables
        executor = REPLExecutor(
            initial_namespace={
                "input_data": request.input_data,
                "answer": {"content": "", "ready": False},
                "current_user": user,
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

    response = StreamingResponse(event_generator(), media_type="text/event-stream")
    response.headers["X-Credits-Remaining"] = str(remaining_credits)
    return response


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


@app.on_event("startup")
def on_startup():
    _init_db()


@app.post("/auth/register")
async def register(payload: RegisterRequest, response: Response):
    user = _register_user(payload.email, payload.password, payload.name)
    session_token = _create_session(user["id"])
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        httponly=True,
        max_age=SESSION_DURATION_DAYS * 24 * 60 * 60,
        samesite="lax",
        secure=False,
    )
    return {"authenticated": True, "user": user}


@app.post("/auth/login")
async def login(payload: LoginRequest, response: Response):
    user = _authenticate_user(payload.email, payload.password)
    session_token = _create_session(user["id"])
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        httponly=True,
        max_age=SESSION_DURATION_DAYS * 24 * 60 * 60,
        samesite="lax",
        secure=False,
    )
    return {"authenticated": True, "user": user}


@app.get("/auth/me")
async def auth_me(
    session_token: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    if not session_token:
        return {"authenticated": False, "user": None}
    user = _get_user_by_session(session_token)
    if not user:
        return {"authenticated": False, "user": None}
    return {"authenticated": True, "user": user}


@app.post("/auth/logout")
async def logout(
    response: Response,
    session_token: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    if session_token:
        _delete_session(session_token)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@app.post("/credits/purchase")
async def purchase_credits(
    payload: PurchaseCreditsRequest,
    user: Dict[str, Any] = Depends(require_user),
):
    purchased = payload.credits if payload.credits is not None else DEFAULT_PURCHASE_CREDITS
    updated_user = _add_credits(user["id"], int(purchased))
    return {
        "ok": True,
        "purchased_credits": int(purchased),
        "user": updated_user,
    }


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/app.js")
async def app_js():
    return FileResponse(FRONTEND_DIR / "app.js")


@app.get("/style.css")
async def style_css():
    return FileResponse(FRONTEND_DIR / "style.css")


@app.get("/buy-credits")
async def buy_credits_page():
    return FileResponse(FRONTEND_DIR / "buy-credits.html")


@app.get("/buy-credits.js")
async def buy_credits_js():
    return FileResponse(FRONTEND_DIR / "buy-credits.js")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
