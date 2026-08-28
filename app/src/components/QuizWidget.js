/**
 * Summary:
 *   QuizWidget.js manages live comprehension quizzes and cultural reinforcement
 *   widgets in the EchoSphere Tandem Co-Teacher client.
 *   When the AI Co-Teacher streams a quiz payload over Agora SD-RTN Data Streams,
 *   this component renders an interactive macOS card with clickable options,
 *   instant tactile feedback, and pedagogical explanations.
 *
 * Key Class:
 *   - QuizWidget: DOM controller for quiz cards, answer evaluation, and explanations.
 */

export class QuizWidget {
  /**
   * Initialize QuizWidget controller.
   * 
   * Algorithm:
   * 1. Store target container element.
   * 2. Initialize active state.
   * 
   * @param {HTMLElement|string} container - Target container element or CSS selector
   */
  constructor(container) {
    this.container = typeof container === 'string'
      ? document.querySelector(container)
      : container;
  }

  /**
   * Renders an interactive quiz card.
   * 
   * Algorithm:
   * 1. Extract question, options, correct_index, and explanation.
   * 2. Create quiz container element with options list.
   * 3. Attach click event listener to evaluate selection and highlight correct/incorrect answers.
   * 4. Display explanation card upon selection.
   * 5. Prepend or display in container.
   * 
   * @param {Object} payload - Quiz payload { question, options, correct_index, explanation }
   */
  renderQuiz(payload) {
    if (!this.container) return;

    const question = payload.question || 'Quick Comprehension Check';
    const options = payload.options || ['Option A', 'Option B'];
    const correctIndex = typeof payload.correct_index === 'number' ? payload.correct_index : 0;
    const explanation = payload.explanation || '';

    const quizEl = document.createElement('div');
    quizEl.className = 'quiz-container';

    const optionsHtml = options.map((opt, idx) => `
      <button class="quiz-opt-btn" data-index="${idx}">
        <span>${String.fromCharCode(65 + idx)}.</span>
        <span style="margin-left: 8px;">${this.escapeHtml(opt)}</span>
      </button>
    `).join('');

    quizEl.innerHTML = `
      <div class="quiz-header">
        <span class="quiz-tag">🧠 Quick Check</span>
        <button class="traffic-light traffic-close" style="width: 10px; height: 10px; border: none; outline: none; cursor: pointer;" title="Dismiss"></button>
      </div>
      <div class="quiz-question">${this.escapeHtml(question)}</div>
      <div class="quiz-options">
        ${optionsHtml}
      </div>
      <div class="quiz-explanation" style="display: none;"></div>
    `;

    // Step 3: Attach option listeners
    const optButtons = quizEl.querySelectorAll('.quiz-opt-btn');
    const explanationBox = quizEl.querySelector('.quiz-explanation');

    optButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const selectedIdx = parseInt(btn.getAttribute('data-index'), 10);
        
        // Disable all buttons
        optButtons.forEach((b, i) => {
          b.disabled = true;
          b.style.cursor = 'default';
          if (i === correctIndex) {
            b.classList.add('correct');
          } else if (i === selectedIdx) {
            b.classList.add('incorrect');
          }
        });

        // Show explanation
        if (explanationBox && explanation) {
          explanationBox.style.display = 'block';
          explanationBox.textContent = `💡 ${explanation}`;
        }
      });
    });

    // Dismiss handler
    const closeBtn = quizEl.querySelector('.traffic-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => {
        quizEl.style.transition = 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)';
        quizEl.style.opacity = '0';
        quizEl.style.transform = 'scale(0.92)';
        setTimeout(() => {
          if (quizEl.parentNode) {
            quizEl.parentNode.removeChild(quizEl);
          }
        }, 200);
      });
    }

    // Insert at beginning of scaffolding container
    this.container.prepend(quizEl);
  }

  /**
   * Clears quizzes.
   */
  clear() {
    if (this.container) {
      this.container.innerHTML = '';
    }
  }

  /**
   * Escape HTML helper.
   */
  escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
}
