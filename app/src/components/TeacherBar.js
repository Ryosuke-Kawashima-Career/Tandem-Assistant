/**
 * Summary:
 *   TeacherBar.js manages the Human-in-the-Loop Teacher Oversight bar
 *   in the EchoSphere Tandem Co-Teacher client.
 *   It provides live alert banners for classroom instructors and quick-action
 *   pedagogical triggers (such as silence breakers, turn nudges, instant quizzes,
 *   and positive reinforcement) without interrupting natural peer flow.
 *
 * Key Class:
 *   - TeacherBar: DOM controller managing teacher alerts, action buttons, and mode visibility.
 */

export class TeacherBar {
  /**
   * Initialize TeacherBar controller.
   * 
   * Algorithm:
   * 1. Store DOM references for teacher dock and alert elements.
   * 2. Initialize action callbacks dictionary.
   * 3. Attach click handlers to teacher action buttons.
   * 
   * @param {Object} options - Configuration and DOM selectors
   */
  constructor(options = {}) {
    this.dockEl = typeof options.dock === 'string'
      ? document.querySelector(options.dock)
      : options.dock;

    this.alertPillEl = typeof options.alertPill === 'string'
      ? document.querySelector(options.alertPill)
      : options.alertPill;

    this.alertTextEl = typeof options.alertText === 'string'
      ? document.querySelector(options.alertText)
      : options.alertText;

    this.actionCallbacks = {};

    this.initActionButtons();
  }

  /**
   * Initializes event listeners on teacher intervention buttons.
   * 
   * Algorithm:
   * 1. Query all buttons with [data-teacher-action] attribute inside dock.
   * 2. Attach click handlers dispatching to registered callbacks.
   */
  initActionButtons() {
    if (!this.dockEl) return;

    const buttons = this.dockEl.querySelectorAll('[data-teacher-action]');
    buttons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const action = btn.getAttribute('data-teacher-action');
        this.dispatchAction(action, e);
      });
    });
  }

  /**
   * Registers a callback for a specific teacher action.
   * 
   * @param {string} action - 'break_silence', 'nudge_turn', 'quiz', 'praise'
   * @param {Function} callback - Handler function
   */
  onAction(action, callback) {
    if (!this.actionCallbacks[action]) {
      this.actionCallbacks[action] = [];
    }
    this.actionCallbacks[action].push(callback);
  }

  /**
   * Dispatches teacher action event to registered subscribers.
   */
  dispatchAction(action, event) {
    const callbacks = this.actionCallbacks[action] || [];
    callbacks.forEach(cb => {
      try {
        cb(action, event);
      } catch (err) {
        console.error(`Error in teacher action callback for ${action}:`, err);
      }
    });
  }

  /**
   * Displays a real-time teacher oversight alert.
   * 
   * Algorithm:
   * 1. Extract alert message and severity ('info', 'warning', 'critical').
   * 2. Update pill styling and message text.
   * 3. Animate alert pulse.
   * 
   * @param {Object} alertData - { message, severity }
   */
  showAlert(alertData = {}) {
    const message = alertData.message || 'Dialogue progressing normally.';
    const severity = (alertData.severity || 'info').toLowerCase();

    if (this.alertPillEl) {
      this.alertPillEl.className = `teacher-alert-pill ${severity}`;
      this.alertPillEl.textContent = severity.toUpperCase();
    }

    if (this.alertTextEl) {
      this.alertTextEl.textContent = message;
    }
  }

  /**
   * Toggles the visibility of the Teacher Oversight Dock.
   * 
   * @param {boolean} visible - Whether teacher dock is visible
   */
  setVisible(visible) {
    if (!this.dockEl) return;
    if (visible) {
      this.dockEl.classList.remove('hidden');
    } else {
      this.dockEl.classList.add('hidden');
    }
  }
}
