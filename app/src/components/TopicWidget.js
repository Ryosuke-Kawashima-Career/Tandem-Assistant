/**
 * Summary:
 *   TopicWidget.js manages the active conversation prompt and real-time speaking
 *   balance gauge in the EchoSphere Tandem Co-Teacher client.
 *   It visually indicates conversational turns, topics (e.g. Indo-Japanese culture/festivals),
 *   and dynamically calculates dialogue parity to ensure equitable peer engagement.
 *
 * Key Class:
 *   - TopicWidget: DOM controller managing the topic title, the prompt, and one
 *     speaking-balance row per participant.
 */

// One tone per row, cycled by position. Four covers every session this product runs
// today, and the cycle degrades quietly rather than running out.
const TONES = [
  { glyph: '🧑‍🎓', avatar: 'rgba(0, 122, 255, 0.12)', fill: 'linear-gradient(90deg, #007aff, #30b0c7)' },
  { glyph: '👨‍🎓', avatar: 'rgba(255, 149, 0, 0.14)', fill: 'linear-gradient(90deg, #ff9500, #ff3b30)' },
  { glyph: '👩‍🎓', avatar: 'rgba(52, 199, 89, 0.14)', fill: 'linear-gradient(90deg, #34c759, #30b0c7)' },
  { glyph: '🧑‍🏫', avatar: 'rgba(175, 82, 222, 0.14)', fill: 'linear-gradient(90deg, #af52de, #007aff)' }
];

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

    this.rowsEl = typeof options.rows === 'string'
      ? document.querySelector(options.rows)
      : options.rows;

    this.emptyEl = typeof options.emptyHint === 'string'
      ? document.querySelector(options.emptyHint)
      : options.emptyHint;

    this.countEl = typeof options.peerCount === 'string'
      ? document.querySelector(options.peerCount)
      : options.peerCount;

    // speaker -> the row's rendered share (17.4). Held here rather than read back off
    // the element, for the same reason NotesPanel keeps its own signatures: a value
    // recovered from the DOM has been through a formatting round-trip and no longer
    // compares reliably against the one that produced it.
    this.renderedShares = new Map();

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
   * Updates the speaking balance gauge (REQ-23).
   *
   * Algorithm:
   * 1. Show the empty hint and stop when nobody has been measured yet - "no data" and
   *    "perfectly balanced" are different statements, and `percentages()` deliberately
   *    returns nothing rather than an even split before anyone has spoken.
   * 2. Order the speakers by share, loudest first, so the row order states the finding.
   * 3. Diff those rows against what is on screen, keyed by speaker: reuse a row whose
   *    share has not moved, update in place one that changed, drop departed speakers.
   * 4. Restate parity from the spread between the loudest and quietest share.
   *
   * 17.4: this used to write two fixed segments of one bar, so a third participant was
   * invisible and the second speaker's share was drawn as `100 - first` rather than as
   * their own measurement - and a two-speaker session was clamped into 5..95, so a
   * genuinely lopsided 97/3 was reported as 95/5. The shares arrive already summing to
   * exactly 100 for any number of speakers (largest-remainder rounding, REQ-23), so
   * nothing here recomputes or normalizes them; it only draws what the backend measured.
   *
   * @param {Object} speakerStats - Mapping of speaker name to integer percentage
   */
  updateSpeakingBalance(speakerStats = {}) {
    if (!this.rowsEl) return;

    const entries = Object.entries(speakerStats)
      .filter(([speaker]) => speaker)
      .sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));

    if (this.countEl) {
      this.countEl.textContent = `${entries.length} Active Peer${entries.length === 1 ? '' : 's'}`;
    }
    if (this.emptyEl) this.emptyEl.classList.toggle('hidden', entries.length > 0);

    // Step 1: nobody measured yet.
    if (entries.length === 0) {
      this.rowsEl.replaceChildren();
      this.renderedShares.clear();
      if (this.statusEl) {
        this.statusEl.textContent = '⏳ Awaiting Speech';
        this.statusEl.style.color = 'var(--macos-text-tertiary)';
      }
      return;
    }

    // Step 3: keyed diff, the discipline 15.6 established for the notes list. A row
    // whose share is unchanged keeps its element, and a changed one is updated in place
    // rather than replaced - the width transition IS the animation here, so a fresh
    // element would start from the CSS default instead of from where the bar actually
    // was, and every recomputation would read as a reset.
    const existing = new Map();
    this.rowsEl.querySelectorAll('[data-speaker]').forEach((el) => {
      existing.set(el.getAttribute('data-speaker'), el);
    });

    const rows = entries.map(([speaker, share], index) => {
      const pct = this.clampShare(share);
      const key = String(speaker);
      const current = existing.get(key);
      if (current) {
        existing.delete(key);
        if (this.renderedShares.get(key) !== pct) {
          const fill = current.querySelector('.balance-row-fill');
          const value = current.querySelector('.balance-row-pct');
          if (fill) fill.style.width = `${pct}%`;
          if (value) value.textContent = `${pct}%`;
        }
        return current;
      }
      return this.buildRow(key, pct, index);
    }).filter(Boolean);

    existing.forEach((el) => el.remove());
    // `replaceChildren` with elements already inside moves them into the given order
    // rather than cloning, so a reused row survives the reorder intact.
    this.rowsEl.replaceChildren(...rows);
    this.renderedShares = new Map(
      entries.map(([speaker, share]) => [String(speaker), this.clampShare(share)])
    );

    // Step 4: parity is the spread across everyone, not the gap between two people, so
    // it still means the same thing once a third participant joins.
    if (this.statusEl) {
      const shares = entries.map(([, share]) => this.clampShare(share));
      const spread = Math.max(...shares) - Math.min(...shares);
      if (spread <= 15) {
        this.statusEl.textContent = '⚖️ Optimal Peer Balance';
        this.statusEl.style.color = 'var(--macos-accent-green)';
      } else if (spread <= 35) {
        this.statusEl.textContent = '📊 Moderate Speaking Gap';
        this.statusEl.style.color = 'var(--macos-accent-orange)';
      } else {
        this.statusEl.textContent = '⚠️ Severe Imbalance Detected';
        this.statusEl.style.color = 'var(--macos-accent-red)';
      }
    }
  }

  /**
   * Reads one speaker's share as a percentage.
   *
   * Bounded to 0..100 only against a malformed payload - not narrowed to a readable
   * range the way the old two-segment gauge was. A share of 0 is a real, and pointed,
   * measurement: someone has not spoken at all.
   *
   * @param {*} share - The percentage as it arrived
   * @returns {number} The share, clamped into 0..100
   */
  clampShare(share) {
    return Math.max(0, Math.min(100, Number(share) || 0));
  }

  /**
   * Builds one participant's balance row.
   *
   * The avatar and bar colour are picked by position in the ordered list rather than
   * from anything about the person: this panel knows a speaker's name and their share
   * and nothing else, so any per-person styling would be invented - which is exactly
   * what the two hardcoded peer cards 17.3 removed were doing.
   *
   * @param {string} speaker - Speaker name, as it arrives in the payload
   * @param {number} pct - Their integer share of the measured speaking time
   * @param {number} index - Position in the ordered list, used only to pick a tone
   * @returns {?HTMLElement} The row element
   */
  buildRow(speaker, pct, index) {
    const tone = TONES[index % TONES.length];
    const template = document.createElement('template');
    template.innerHTML = `
      <div class="balance-row" data-speaker="${this.escape(speaker)}">
        <div class="balance-row-label">
          <span class="balance-row-avatar" style="background: ${tone.avatar};">${tone.glyph}</span>
          <span class="balance-row-name">${this.escape(speaker)}</span>
          <span class="balance-row-pct">${pct}%</span>
        </div>
        <div class="balance-row-track">
          <div class="balance-row-fill" style="width: ${pct}%; background: ${tone.fill};"></div>
        </div>
      </div>
    `.trim();
    return template.content.firstElementChild;
  }

  /**
   * Escapes text before it is inserted as HTML.
   *
   * A speaker name arrives from the session payload, so it is never trusted as markup.
   *
   * @param {string} value - Raw text
   * @returns {string} HTML-safe text
   */
  escape(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[char]);
  }
}
