/**
 * Logic Answering System - Frontend Application
 *
 * Handles:
 * - WebSocket connection for real-time updates
 * - Tree view rendering and interaction
 * - Audio recording and transcription
 * - User query handling
 */

class LogicAnsweringApp {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.ws = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;

        this.init();
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    init() {
        // Get DOM elements
        this.elements = {
            questionInput: document.getElementById('question-input'),
            submitBtn: document.getElementById('submit-btn'),
            recordBtn: document.getElementById('record-btn'),
            audioStatus: document.getElementById('audio-status'),
            statusBar: document.getElementById('status-bar'),
            statusText: document.getElementById('status-text'),
            costDisplay: document.getElementById('cost-display'),
            treeSection: document.getElementById('tree-section'),
            treeContainer: document.getElementById('tree-container'),
            expandAll: document.getElementById('expand-all'),
            collapseAll: document.getElementById('collapse-all'),
            answerSection: document.getElementById('answer-section'),
            answerContent: document.getElementById('answer-content'),
            feedbackSection: document.getElementById('feedback-section'),
            feedbackInput: document.getElementById('feedback-input'),
            feedbackBtn: document.getElementById('feedback-btn'),
            queryModal: document.getElementById('query-modal'),
            queryTitle: document.getElementById('query-title'),
            queryText: document.getElementById('query-text'),
            queryResponse: document.getElementById('query-response'),
            querySubmit: document.getElementById('query-submit'),
            costSection: document.getElementById('cost-section'),
            costDetails: document.getElementById('cost-details'),
            warnings: document.getElementById('warnings'),
        };

        // Bind event listeners
        this.elements.submitBtn.addEventListener('click', () => this.submitQuestion());
        this.elements.recordBtn.addEventListener('click', () => this.toggleRecording());
        this.elements.expandAll.addEventListener('click', () => this.expandAllNodes());
        this.elements.collapseAll.addEventListener('click', () => this.collapseAllNodes());
        this.elements.feedbackBtn.addEventListener('click', () => this.submitFeedback());
        this.elements.querySubmit.addEventListener('click', () => this.submitQueryResponse());

        // Enter key to submit
        this.elements.questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.submitQuestion();
            }
        });
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showError('Connection error. Please refresh the page.');
        };

        this.ws.onclose = () => {
            console.log('WebSocket closed');
        };
    }

    handleMessage(data) {
        switch (data.type) {
            case 'state_update':
                this.updateState(data.data);
                break;
            case 'user_query':
                this.showQueryModal(data.query_type, data.question);
                break;
            case 'error':
                this.showError(data.message);
                break;
        }
    }

    updateState(state) {
        // Update status
        this.elements.statusText.textContent = this.formatStatus(state.status);

        // Update cost display
        if (state.cost_summary) {
            this.elements.costDisplay.innerHTML =
                `<span>Cost: $${state.cost_summary.total_cost_usd.toFixed(4)}</span>`;
        }

        // Update tree view
        if (state.tree_view) {
            this.renderTreeView(state.tree_view);
            this.elements.treeSection.classList.remove('hidden');
        }

        // Handle completion
        if (state.is_complete) {
            this.elements.statusBar.querySelector('.spinner').style.display = 'none';

            if (state.final_answer) {
                this.showAnswer(state.final_answer);
            } else if (state.termination_reason) {
                this.showAnswer(state.termination_reason);
            }

            // Show cost summary
            if (state.cost_summary) {
                this.renderCostSummary(state.cost_summary);
            }
        }
    }

    formatStatus(status) {
        const statusMap = {
            'initializing': 'Initializing...',
            'disambiguating': 'Lowering Ambiguity...',
            'reasoning': 'Reasoning...',
            'completed': 'Complete',
            'unanswerable': 'Proven Unanswerable',
            'max_iterations': 'Maximum Iterations Reached',
            'error': 'Error',
        };
        return statusMap[status] || status;
    }

    async submitQuestion() {
        const question = this.elements.questionInput.value.trim();
        if (!question) return;

        // Disable input
        this.elements.submitBtn.disabled = true;
        this.elements.questionInput.disabled = true;

        // Show status bar
        this.elements.statusBar.classList.remove('hidden');
        this.elements.statusBar.querySelector('.spinner').style.display = 'block';

        // Connect WebSocket and start processing
        this.connectWebSocket();

        // Wait for connection then send
        await new Promise(resolve => {
            const checkConnection = setInterval(() => {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    clearInterval(checkConnection);
                    resolve();
                }
            }, 100);
        });

        this.ws.send(JSON.stringify({
            type: 'start_processing',
            question: question
        }));
    }

    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            await this.startRecording();
        }
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                await this.transcribeAudio(audioBlob);
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            this.elements.recordBtn.innerHTML = '<span class="icon">⏹</span>';
            this.elements.audioStatus.textContent = 'Recording...';
            this.elements.audioStatus.classList.remove('hidden');
            this.elements.audioStatus.classList.add('recording');
        } catch (error) {
            console.error('Recording error:', error);
            this.elements.audioStatus.textContent = 'Microphone access denied';
            this.elements.audioStatus.classList.remove('hidden');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            this.isRecording = false;
            this.elements.recordBtn.innerHTML = '<span class="icon">🎤</span>';
            this.elements.audioStatus.textContent = 'Transcribing...';
            this.elements.audioStatus.classList.remove('recording');
        }
    }

    async transcribeAudio(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');

        try {
            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.text) {
                this.elements.questionInput.value = data.text;
                this.elements.audioStatus.textContent = `Transcribed (${data.language})`;
            } else if (data.error) {
                this.elements.audioStatus.textContent = `Error: ${data.error}`;
            }
        } catch (error) {
            console.error('Transcription error:', error);
            this.elements.audioStatus.textContent = 'Transcription failed';
        }
    }

    renderTreeView(treeData) {
        this.elements.treeContainer.innerHTML = '';

        // Question header
        const header = document.createElement('div');
        header.className = 'tree-question';
        header.innerHTML = `
            <div style="margin-bottom: 1rem;">
                <strong>Original:</strong> ${this.escapeHtml(treeData.original_question)}
            </div>
            ${treeData.refined_question && treeData.refined_question !== treeData.original_question ? `
            <div class="diff-container">
                <div class="diff-from">
                    <div class="diff-label">Original</div>
                    ${this.escapeHtml(treeData.original_question)}
                </div>
                <div class="diff-to">
                    <div class="diff-label">Refined</div>
                    ${this.escapeHtml(treeData.refined_question)}
                </div>
            </div>
            ` : ''}
        `;
        this.elements.treeContainer.appendChild(header);

        // Render phases
        if (treeData.phases) {
            treeData.phases.forEach(phase => {
                const phaseNode = this.createPhaseNode(phase);
                this.elements.treeContainer.appendChild(phaseNode);
            });
        }
    }

    createPhaseNode(phase) {
        const node = document.createElement('div');
        node.className = 'tree-node';

        const statusClass = phase.status === 'complete' ? 'complete' : 'in-progress';
        node.innerHTML = `
            <div class="phase-header ${statusClass}">
                ${this.escapeHtml(phase.name)}
                ${phase.iterations ? ` (${phase.iterations} iterations)` : ''}
            </div>
        `;

        if (phase.children && phase.children.length > 0) {
            const childContainer = document.createElement('div');
            childContainer.className = 'node-children expanded';

            phase.children.forEach(child => {
                const childNode = this.createNode(child);
                childContainer.appendChild(childNode);
            });

            node.appendChild(childContainer);
        }

        return node;
    }

    createNode(nodeData) {
        const node = document.createElement('div');
        node.className = 'tree-node';

        const hasChildren = nodeData.children && nodeData.children.length > 0;
        const hasContent = nodeData.full_text || nodeData.text || nodeData.explanation || nodeData.full_explanation;

        // Determine icon
        let icon = '📁';
        if (nodeData.type === 'reasoning') icon = '💭';
        else if (nodeData.type === 'claim') icon = '📝';
        else if (nodeData.type === 'verification' || nodeData.type === 'verification_node') icon = '✓';
        else if (nodeData.type === 'cohesive') icon = '🔄';
        else if (nodeData.type === 'iteration') icon = '🔁';

        // Verdict badge
        let verdictBadge = '';
        if (nodeData.verdict) {
            const verdictClass = `verdict-${nodeData.verdict.toLowerCase()}`;
            verdictBadge = `<span class="node-badge ${verdictClass}">${nodeData.verdict}</span>`;
        }

        // Category badge
        let categoryBadge = '';
        if (nodeData.category) {
            categoryBadge = `<span class="node-badge category-badge">${nodeData.category}</span>`;
        }

        node.innerHTML = `
            <div class="node-header">
                ${hasChildren || hasContent ?
                    '<span class="node-toggle">▶</span>' :
                    '<span class="node-toggle" style="visibility: hidden;">▶</span>'}
                <span class="node-icon">${icon}</span>
                <span class="node-name">${this.escapeHtml(nodeData.name)}</span>
                ${verdictBadge}
                ${categoryBadge}
            </div>
        `;

        const header = node.querySelector('.node-header');
        const toggle = node.querySelector('.node-toggle');

        // Content
        if (hasContent) {
            const content = document.createElement('div');
            content.className = 'node-content';
            content.style.display = 'none';
            content.textContent = nodeData.full_text || nodeData.text || nodeData.full_explanation || nodeData.explanation;
            node.appendChild(content);
        }

        // Diff display for cohesive updates
        if (nodeData.type === 'cohesive' && nodeData.from && nodeData.to) {
            const diffContent = document.createElement('div');
            diffContent.className = 'node-content';
            diffContent.style.display = 'none';
            diffContent.innerHTML = `
                <div class="diff-container">
                    <div class="diff-from">
                        <div class="diff-label">Before</div>
                        ${this.escapeHtml(nodeData.from)}
                    </div>
                    <div class="diff-to">
                        <div class="diff-label">After</div>
                        ${this.escapeHtml(nodeData.to)}
                    </div>
                </div>
            `;
            node.appendChild(diffContent);
        }

        // Children container
        if (hasChildren) {
            const childContainer = document.createElement('div');
            childContainer.className = 'node-children';

            nodeData.children.forEach(child => {
                const childNode = this.createNode(child);
                childContainer.appendChild(childNode);
            });

            node.appendChild(childContainer);
        }

        // Toggle behavior
        if (hasChildren || hasContent) {
            header.addEventListener('click', () => {
                const isExpanded = toggle.classList.contains('expanded');
                toggle.classList.toggle('expanded');

                // Toggle content
                const content = node.querySelector(':scope > .node-content');
                if (content) {
                    content.style.display = isExpanded ? 'none' : 'block';
                }

                // Toggle children
                const children = node.querySelector(':scope > .node-children');
                if (children) {
                    children.classList.toggle('expanded');
                }
            });
        }

        return node;
    }

    expandAllNodes() {
        document.querySelectorAll('.node-toggle').forEach(toggle => {
            toggle.classList.add('expanded');
        });
        document.querySelectorAll('.node-children').forEach(children => {
            children.classList.add('expanded');
        });
        document.querySelectorAll('.node-content').forEach(content => {
            content.style.display = 'block';
        });
    }

    collapseAllNodes() {
        document.querySelectorAll('.node-toggle').forEach(toggle => {
            toggle.classList.remove('expanded');
        });
        document.querySelectorAll('.node-children').forEach(children => {
            children.classList.remove('expanded');
        });
        document.querySelectorAll('.node-content').forEach(content => {
            content.style.display = 'none';
        });
    }

    showAnswer(answer) {
        this.elements.answerContent.textContent = answer;
        this.elements.answerSection.classList.remove('hidden');
        this.elements.feedbackSection.classList.remove('hidden');
    }

    submitFeedback() {
        const feedback = this.elements.feedbackInput.value.trim();
        if (!feedback || !this.ws) return;

        this.ws.send(JSON.stringify({
            type: 'continue_feedback',
            feedback: feedback
        }));

        this.elements.feedbackInput.value = '';
        this.elements.statusBar.querySelector('.spinner').style.display = 'block';
        this.elements.statusText.textContent = 'Continuing with feedback...';
    }

    showQueryModal(queryType, question) {
        const titles = {
            'clarification': 'Clarification Needed',
            'empirical_data': 'Data Required',
            'axiom_choice': 'Axiom Choice',
        };

        this.elements.queryTitle.textContent = titles[queryType] || 'Input Required';
        this.elements.queryText.textContent = question;
        this.elements.queryResponse.value = '';
        this.elements.queryModal.classList.remove('hidden');
    }

    submitQueryResponse() {
        const response = this.elements.queryResponse.value.trim();
        if (!response || !this.ws) return;

        this.ws.send(JSON.stringify({
            type: 'query_response',
            response: response
        }));

        this.elements.queryModal.classList.add('hidden');
    }

    renderCostSummary(summary) {
        this.elements.costDetails.innerHTML = `
            <div class="cost-item">
                <div class="cost-item-label">Total Cost</div>
                <div class="cost-item-value">$${summary.total_cost_usd.toFixed(4)}</div>
            </div>
            <div class="cost-item">
                <div class="cost-item-label">Total Calls</div>
                <div class="cost-item-value">${summary.total_calls}</div>
            </div>
            <div class="cost-item">
                <div class="cost-item-label">Opus (Smart)</div>
                <div class="cost-item-value">$${summary.opus_cost_usd.toFixed(4)} (${summary.opus_calls} calls)</div>
            </div>
            <div class="cost-item">
                <div class="cost-item-label">Flash (Fast)</div>
                <div class="cost-item-value">$${summary.flash_cost_usd.toFixed(4)} (${summary.flash_calls} calls)</div>
            </div>
            <div class="cost-item">
                <div class="cost-item-label">Disambiguation Iterations</div>
                <div class="cost-item-value">${summary.disambiguation_iterations}</div>
            </div>
            <div class="cost-item">
                <div class="cost-item-label">Max Verification Depth</div>
                <div class="cost-item-value">${summary.max_verification_depth}</div>
            </div>
        `;

        if (summary.warnings && summary.warnings.length > 0) {
            this.elements.warnings.innerHTML = summary.warnings
                .map(w => `<div class="warning-item">⚠️ ${this.escapeHtml(w)}</div>`)
                .join('');
        }

        this.elements.costSection.classList.remove('hidden');
    }

    showError(message) {
        this.elements.statusText.textContent = `Error: ${message}`;
        this.elements.statusBar.querySelector('.spinner').style.display = 'none';
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.app = new LogicAnsweringApp();
});
