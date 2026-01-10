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

const API_URL = 'http://localhost:8000/process';

runBtn.addEventListener('click', async () => {
    const prompt = promptInput.value.trim();
    const inputData = dataInput.value.trim();
    const maxIterations = parseInt(maxIterInput.value);

    if (!prompt) {
        alert('Please enter a task prompt.');
        return;
    }

    // UI Feedback
    runBtn.disabled = true;
    runBtn.textContent = '🧠 Agent Thinking...';
    executionLog.classList.remove('hidden');
    resultSection.classList.add('hidden');
    logContent.innerHTML = '';

    addLogEntry('SYSTEM', 'Initializing RLM context...', 'system', '⚙️ SYSTEM');
    addLogEntry('SYSTEM', `Input data size: ${inputData.length} characters`, 'system', '⚙️ SYSTEM');

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt,
                input_data: inputData,
                max_iterations: maxIterations
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error ${response.status}: ${errorText}`);
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
                    } catch (e) {
                        console.error('Error parsing event:', e, line);
                    }
                }
            }
        }

    } catch (error) {
        console.error('Error:', error);
        addLogEntry('ERROR', error.message, 'error', '❌ ERROR');
        runBtn.textContent = '✗ Execution Failed';
        runBtn.style.background = 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)';
    } finally {
        runBtn.disabled = false;
    }
});

function handleStreamEvent(event) {
    switch (event.type) {
        case 'iteration_start':
            addLogEntry('ITERATION', `Starting Iteration ${event.iteration}...`, 'system', `🔄 LOOP ${event.iteration}`);
            break;

        case 'thought':
            addLogEntry('THOUGHT', event.content, 'assistant', '🧠 THINKING');
            break;

        case 'tool_call':
            if (event.tool === 'python_repl') {
                addLogEntry('TOOL CALL', event.args.code || event.args, 'tool', '💻 EXECUTING CODE');
            } else if (event.tool === 'llm_batch_tool') {
                addLogEntry('TOOL CALL', `Delegating to ${event.args.prompts.length} sub-LLM instances...`, 'tool', '📡 DELEGATING');
            } else {
                addLogEntry('TOOL CALL', JSON.stringify(event.args), 'tool', `🛠️ TOOL: ${event.tool}`);
            }
            break;

        case 'tool_result':
            let content = event.content;
            if (content.length > 5000) content = content.substring(0, 5000) + "\n... [truncated]";
            addLogEntry('TOOL RESULT', content, 'system', '📥 OUTPUT');
            break;

        case 'error':
            addLogEntry('ERROR', event.content, 'error', '❌ ERROR');
            break;

        case 'final_result':
            resultSection.classList.remove('hidden');
            finalContent.textContent = event.content;
            iterCount.textContent = `✅ Completed in ${event.iterations} iteration${event.iterations > 1 ? 's' : ''}`;
            runBtn.textContent = '✓ Task Complete';
            runBtn.style.background = 'linear-gradient(135deg, #10B981 0%, #059669 100%)';
            // Scroll to final result
            resultSection.scrollIntoView({ behavior: 'smooth' });
            break;
    }
}

resetBtn.addEventListener('click', () => {
    promptInput.value = '';
    dataInput.value = '';
    executionLog.classList.add('hidden');
    resultSection.classList.add('hidden');
    runBtn.textContent = '▶ Initialize RLM Loop';
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
        // Special formatting for code/tool calls
        return `<pre class="code-block">${escapeHtml(content)}</pre>`;
    }
    // Handle markdown-ish bold/code in reasoning
    return escapeHtml(content)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/`(.*?)`/g, '<code>$1</code>');
}

function displayLogs(history) {
    if (!history || history.length === 0) {
        addLogEntry('SYSTEM', 'No execution history available', 'system');
        return;
    }

    logContent.innerHTML = '';

    history.forEach((msg) => {
        if (!msg.parts || !Array.isArray(msg.parts)) return;

        msg.parts.forEach(part => {
            const pk = part.part_kind || part.kind;

            if (pk === 'user-prompt' || pk === 'user-request' || pk === 'user') {
                const content = part.content || part.text || (msg.parts && msg.parts[0].text);
                addLogEntry('USER', content, 'user', '👤 USER');
            }
            else if (pk === 'system-prompt' || pk === 'system') {
                addLogEntry('SYSTEM', 'RLM Instructions Applied', 'system', '⚙️ SYSTEM');
            }
            else if (pk === 'text') {
                addLogEntry('ASSISTANT', part.content || part.text, 'assistant', '🧠 THINKING');
            }
            else if (pk === 'tool-call') {
                const toolName = part.tool_name || part.name;
                const args = part.args;
                let displayArgs = typeof args === 'string' ? args : JSON.stringify(args, null, 2);

                if (toolName === 'python_repl') {
                    const code = (typeof args === 'object' && args.code) ? args.code :
                        (typeof args === 'string' && args.includes('code')) ? JSON.parse(args).code : displayArgs;
                    addLogEntry('TOOL CALL', code, 'tool', '💻 EXECUTING CODE');
                } else if (toolName === 'llm_batch_tool') {
                    addLogEntry('TOOL CALL', `Calling sub-LLMs in parallel...`, 'tool', '📡 DELEGATING');
                } else {
                    addLogEntry('TOOL CALL', `${toolName}: ${displayArgs}`, 'tool', `🛠️ TOOL: ${toolName}`);
                }
            }
            else if (pk === 'tool-return' || pk === 'tool_return') {
                let content = part.content || part.text || part.result;
                if (typeof content !== 'string') content = JSON.stringify(content, null, 2);

                if (content.length > 5000) {
                    content = content.substring(0, 5000) + "\n... [truncated in log]";
                }
                addLogEntry('TOOL RESULT', content, 'system', '� OUTPUT');
            }
        });
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add keyboard shortcut (Ctrl+Enter to run)
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter' && !runBtn.disabled) {
        runBtn.click();
    }
});
