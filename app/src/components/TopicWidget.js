/**
 * Summary:
 *   TopicWidget.js manages the active conversation prompt and real-time speaking
 *   balance gauge in the EchoSphere Tandem Co-Teacher client.
 *   It visually indicates conversational turns, topics (e.g. Indo-Japanese culture/festivals),
 *   and dynamically calculates dialogue parity to ensure equitable peer engagement.
 *
 * Key Class:
 *   - TopicWidget: DOM controller managing topic title, prompt, and speaking progress bars.
 */

export class TopicWidget {
  /**
   * Initialize TopicWidget controller.
   * 
   * Algorithm:
   * 1. Store DOM references for topic container and balance elements.
   * 2. Initialize default state values.
   * 
   * @param {Object} options - Options containing container elements or selectors
   */
  constructor(options = {}) {
    this.topicTitleEl = typeof options.topicTitle === 'string'
      ? document.querySelector(options.topicTitle)
      : options.topicTitle;
    
    this.topicPromptEl = typeof options.topicPrompt === 'string'
      ? document.querySelector(options.topicPrompt)
      : options.topicPrompt;

    this.segAEl = typeof options.segA === 'string'
      ? document.querySelector(options.segA)
      : options.segA;

    this.segBEl = typeof options.segB === 'string'
      ? document.querySelector(options.segB)
      : options.segB;

    this.labelAEl = typeof options.labelA === 'string'
      ? document.querySelector(options.labelA)
      : options.labelA;

    this.labelBEl = typeof options.labelB === 'string'
      ? document.querySelector(options.labelB)
      : options.labelB;

    this.statusEl = typeof options.statusText === 'string'
      ? document.querySelector(options.statusText)
      : options.statusText;
  }

  /**
   * Updates the conversational topic card.
   * 
   * Algorithm:
   * 1. Update topic title element text.
   * 2. Update discussion prompt element text.
   * 
   * @param {Object} data - { topic_title, prompt }
   */
  updateTopic(data) {
    if (this.topicTitleEl && data.topic_title) {
      this.topicTitleEl.textContent = data.topic_title;
    }
    if (this.topicPromptEl && data.prompt) {
      this.topicPromptEl.textContent = data.prompt;
    }
  }

  /**
   * Updates the peer speaking balance gauge.
   * 
   * Algorithm:
   * 1. Parse speaker percentages dictionary.
   * 2. Calculate distribution across primary speakers (e.g. Kenji vs Aarav).
   * 3. Update width of progress bar segments with CSS transitions.
   * 4. Update legend labels and parity status indicator.
   * 
   * @param {Object} speakerStats - Mapping of speaker names to percentage integers
   */
  updateSpeakingBalance(speakerStats = {}) {
    const keys = Object.keys(speakerStats);
    if (keys.length < 2) return;

    const speakerA = keys[0];
    const speakerB = keys[1];
    const pctA = Math.max(5, Math.min(95, speakerStats[speakerA] || 50));
    const pctB = 100 - pctA;

    // Step 3: Update bar widths
    if (this.segAEl) {
      this.segAEl.style.width = `${pctA}%`;
    }
    if (this.segBEl) {
      this.segBEl.style.width = `${pctB}%`;
    }

    // Step 4: Update legends
    if (this.labelAEl) {
      this.labelAEl.textContent = `${speakerA}: ${pctA}%`;
    }
    if (this.labelBEl) {
      this.labelBEl.textContent = `${speakerB}: ${pctB}%`;
    }

    // Update status evaluation
    if (this.statusEl) {
      const diff = Math.abs(pctA - pctB);
      if (diff <= 15) {
        this.statusEl.textContent = '⚖️ Optimal Peer Balance';
        this.statusEl.style.color = 'var(--macos-accent-green)';
      } else if (diff <= 35) {
        this.statusEl.textContent = '📊 Moderate Speaking Gap';
        this.statusEl.style.color = 'var(--macos-accent-orange)';
      } else {
        this.statusEl.textContent = '⚠️ Severe Imbalance Detected';
        this.statusEl.style.color = 'var(--macos-accent-red)';
      }
    }
  }
}
