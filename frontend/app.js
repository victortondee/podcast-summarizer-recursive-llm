const promptInput = document.getElementById('prompt');
const dataInput = document.getElementById('inputData');
const maxIterInput = document.getElementById('maxIterations');
const runBtn = document.getElementById('runBtn');
const resetBtn = document.getElementById('resetBtn');
const executionLog = document.getElementById('executionLog');
const logContent = document.getElementById('logContent');
const resultSection = document.getElementById('resultSection');
const finalContent = document.getElementById('finalContent');
const iterCount = document.getElementById('iterCount');
const creditsBanner = document.getElementById('creditsBanner');
const buyCreditsBtn = document.getElementById('buyCreditsBtn');

const loginSection = document.getElementById('loginSection');
const appSection = document.getElementById('appSection');
const userBar = document.getElementById('userBar');
const userEmail = document.getElementById('userEmail');
const userCredits = document.getElementById('userCredits');
const logoutBtn = document.getElementById('logoutBtn');
const authMessage = document.getElementById('authMessage');
const authForm = document.getElementById('authForm');
const authName = document.getElementById('authName');
const authEmail = document.getElementById('authEmail');
const authPassword = document.getElementById('authPassword');
const authSubmitBtn = document.getElementById('authSubmitBtn');
const authToggleBtn = document.getElementById('authToggleBtn');
const pageHeader = document.querySelector('header');

const API_URL = '/process';

let currentUser = null;
let authMode = 'login';

initApp();

async function initApp() {
    await restoreSession();
    setAuthMode('login');
}

async function restoreSession() {
    try {
        const response = await fetch('/auth/me', { credentials: 'include' });
        if (!response.ok) {
            showLogin();
            return;
        }

        const payload = await response.json();
        if (payload.authenticated && payload.user) {
            setAuthenticatedUser(payload.user);
        } else {
            showLogin();
        }
    } catch (_) {
        showLogin('Unable to connect to auth service.');
    }
}

function setAuthMode(mode) {
    authMode = mode;
    const isRegister = mode === 'register';

    authSubmitBtn.textContent = isRegister ? 'Create Account' : 'Sign In';
    authToggleBtn.textContent = isRegister ? 'Already have an account?' : 'Create Account';
    authName.closest('.field').classList.toggle('hidden', !isRegister);
    authMessage.classList.add('hidden');
}

function setAuthenticatedUser(user) {
    currentUser = user;
    userEmail.textContent = user.email;
    userCredits.textContent = `Credits: ${user.credits ?? 0}/10`;
    syncCreditsState();
    loginSection.classList.add('hidden');
    appSection.classList.remove('hidden');
    userBar.classList.remove('hidden');
    if (pageHeader) pageHeader.classList.add('hidden');
    authMessage.classList.add('hidden');
}

function showLogin(message = '') {
    currentUser = null;
    loginSection.classList.remove('hidden');
    appSection.classList.add('hidden');
    creditsBanner.classList.add('hidden');
    userBar.classList.add('hidden');
    if (pageHeader) pageHeader.classList.remove('hidden');

    if (message) {
        authMessage.textContent = message;
        authMessage.classList.remove('hidden');
    } else {
        authMessage.classList.add('hidden');
    }
}

function syncCreditsState() {
    const credits = Number(currentUser?.credits ?? 0);
    userCredits.textContent = `Credits: ${credits}/10`;
    if (credits <= 0) {
        creditsBanner.classList.remove('hidden');
        runBtn.disabled = true;
        runBtn.textContent = 'No Credits Remaining';
    } else {
        creditsBanner.classList.add('hidden');
        if (runBtn.textContent === 'No Credits Remaining') {
            runBtn.textContent = 'Initialize RLM Loop';
            runBtn.disabled = false;
        }
    }
}

authToggleBtn.addEventListener('click', () => {
    setAuthMode(authMode === 'login' ? 'register' : 'login');
});

authForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const email = authEmail.value.trim();
    const password = authPassword.value;
    const name = authName.value.trim();

    if (!email || !password) {
        showLogin('Email and password are required.');
        return;
    }

    if (authMode === 'register' && password.length < 8) {
        showLogin('Password must be at least 8 characters.');
        return;
    }

    authSubmitBtn.disabled = true;

    try {
        const endpoint = authMode === 'register' ? '/auth/register' : '/auth/login';
        const payload = authMode === 'register'
            ? { email, password, name: name || null }
            : { email, password };

        const response = await fetch(endpoint, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });

        const data = await response.json();
        if (!response.ok || !data.authenticated) {
            throw new Error(data.detail || 'Authentication failed.');
        }

        authPassword.value = '';
        setAuthenticatedUser(data.user);
    } catch (error) {
        showLogin(error.message || 'Authentication failed.');
    } finally {
        authSubmitBtn.disabled = false;
    }
});

logoutBtn.addEventListener('click', async () => {
    try {
        await fetch('/auth/logout', {
            method: 'POST',
            credentials: 'include',
        });
    } catch (_) {
        // Continue to reset UI even if request fails.
    }

    authPassword.value = '';
    showLogin();
    logContent.innerHTML = '';
    executionLog.classList.add('hidden');
    resultSection.classList.add('hidden');
});

buyCreditsBtn.addEventListener('click', () => {
    window.location.href = '/buy-credits';
});

runBtn.addEventListener('click', async () => {
    if (!currentUser) {
        showLogin('Please sign in to run tasks.');
        return;
    }

    const prompt = promptInput.value.trim();
    const inputData = dataInput.value.trim();
    const maxIterations = parseInt(maxIterInput.value, 10);

    if (!prompt) {
        alert('Please enter a task prompt.');
        return;
    }

    runBtn.disabled = true;
    runBtn.textContent = 'Agent Thinking...';
    executionLog.classList.remove('hidden');
    resultSection.classList.add('hidden');
    logContent.innerHTML = '';

    addLogEntry('SYSTEM', `Signed in as ${currentUser.email}`, 'system', 'ACCOUNT');
    addLogEntry('SYSTEM', 'Initializing RLM context...', 'system', 'SYSTEM');
    addLogEntry('SYSTEM', `Input data size: ${inputData.length} characters`, 'system', 'SYSTEM');

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt,
                input_data: inputData,
                max_iterations: maxIterations,
            }),
        });

        if (response.status === 401) {
            showLogin('Session expired. Please sign in again.');
            throw new Error('Authentication required. Please sign in again.');
        }

        if (!response.ok) {
            let detail = `Server error ${response.status}`;
            try {
                const errorJson = await response.json();
                if (typeof errorJson.detail === 'string') {
                    detail = errorJson.detail;
                } else if (errorJson.detail && errorJson.detail.message) {
                    detail = errorJson.detail.message;
                    if (typeof errorJson.detail.credits_remaining === 'number') {
                        currentUser.credits = errorJson.detail.credits_remaining;
                        syncCreditsState();
                    }
                }
            } catch (_) {
                const errorText = await response.text();
                if (errorText) detail = errorText;
            }
            throw new Error(detail);
        }

        const remainingCredits = parseInt(response.headers.get('X-Credits-Remaining') || '', 10);
        if (Number.isFinite(remainingCredits)) {
            currentUser.credits = remainingCredits;
            syncCreditsState();
        } else if (typeof currentUser.credits === 'number') {
            currentUser.credits = Math.max(0, currentUser.credits - 1);
            syncCreditsState();
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const event = JSON.parse(line.substring(6));
                        handleStreamEvent(event);
                    } catch (err) {
                        console.error('Error parsing event:', err, line);
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error:', error);
        addLogEntry('ERROR', error.message, 'error', 'ERROR');
        runBtn.textContent = 'Execution Failed';
        runBtn.style.background = 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)';
    } finally {
        runBtn.disabled = false;
    }
});

function handleStreamEvent(event) {
    switch (event.type) {
        case 'iteration_start':
            addLogEntry('ITERATION', `Starting Iteration ${event.iteration}...`, 'system', `LOOP ${event.iteration}`);
            break;

        case 'thought':
            addLogEntry('THOUGHT', event.content, 'assistant', 'THINKING');
            break;

        case 'tool_call':
            if (event.tool === 'python_repl') {
                addLogEntry('TOOL CALL', event.args.code || event.args, 'tool', 'EXECUTING CODE');
            } else if (event.tool === 'llm_batch_tool') {
                addLogEntry('TOOL CALL', `Delegating to ${event.args.prompts.length} sub-LLM instances...`, 'tool', 'DELEGATING');
            } else {
                addLogEntry('TOOL CALL', JSON.stringify(event.args), 'tool', `TOOL: ${event.tool}`);
            }
            break;

        case 'tool_result': {
            let content = event.content;
            if (content.length > 5000) content = `${content.substring(0, 5000)}\n... [truncated]`;
            addLogEntry('TOOL RESULT', content, 'system', 'OUTPUT');
            break;
        }

        case 'error':
            addLogEntry('ERROR', event.content, 'error', 'ERROR');
            break;

        case 'final_result':
            resultSection.classList.remove('hidden');
            finalContent.textContent = event.content;
            iterCount.textContent = `Completed in ${event.iterations} iteration${event.iterations > 1 ? 's' : ''}`;
            runBtn.textContent = 'Task Complete';
            runBtn.style.background = 'linear-gradient(135deg, #10B981 0%, #059669 100%)';
            resultSection.scrollIntoView({ behavior: 'smooth' });
            break;

        default:
            break;
    }
}

resetBtn.addEventListener('click', () => {
    promptInput.value = '';
    dataInput.value = '';
    executionLog.classList.add('hidden');
    resultSection.classList.add('hidden');
    runBtn.textContent = 'Initialize RLM Loop';
    runBtn.style.background = '';
    runBtn.disabled = false;
    logContent.innerHTML = '';
});

function addLogEntry(role, content, type = 'default', title = null) {
    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;

    const displayTitle = title || role;

    entry.innerHTML = `
        <div class="log-header">
            <span class="log-role">${displayTitle}</span>
            <span class="log-time">${new Date().toLocaleTimeString()}</span>
        </div>
        <div class="log-text">${formatContent(content, type)}</div>
    `;
    logContent.appendChild(entry);
    logContent.scrollTop = logContent.scrollHeight;
}

function formatContent(content, type) {
    if (type === 'tool') {
        return `<pre class="code-block">${escapeHtml(content)}</pre>`;
    }

    return escapeHtml(content)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/`(.*?)`/g, '<code>$1</code>');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter' && !runBtn.disabled) {
        runBtn.click();
    }
});
