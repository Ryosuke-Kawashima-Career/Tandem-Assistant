/**
 * Summary:
 *   Subtitles.js manages the real-time tri-lingual subtitle stream in the EchoSphere
 *   Tandem Co-Teacher client.
 *   It receives speech-to-text transcriptions and multi-lingual translations
 *   (Hindi / Japanese / English) along with Romaji / Latin transliterations,
 *   rendering them into frosted glass speech cards with smooth spring transitions.
 *
 * Key Class:
 *   - Subtitles: DOM controller managing subtitle entries, speaker badges,
 *     transliteration pills, and auto-scroll.
 */

export class Subtitles {
  /**
   * Initialize Subtitles controller.
   * 
   * Algorithm:
   * 1. Store reference to the container DOM element.
   * 2. Initialize max entries limit and history cache.
   * 
   * @param {HTMLElement|string} container - Target container element or CSS selector
   */
  constructor(container) {
    // Step 1: Resolve container
    this.container = typeof container === 'string' 
      ? document.querySelector(container) 
      : container;
    
    // Step 2: History cache and limit
    this.history = [];
    this.maxEntries = 50;
  }

  /**
   * Formats speaker metadata with visual badges and flags.
   * 
   * Algorithm:
   * 1. Inspect speaker string.
   * 2. Return appropriate avatar icon and display badge.
   * 
   * @param {string} speaker - Speaker name or identifier
   * @returns {Object} { displayName, avatar, isAI }
   */
  formatSpeaker(speaker) {
    const s = (speaker || 'Unknown').trim();
    if (s.toLowerCase().includes('bot') || s.toLowerCase().includes('ai') || s.toLowerCase().includes('teacher')) {
      return { displayName: 'EchoSphere AI Co-Teacher', avatar: '🤖', isAI: true };
    }
    if (s.toLowerCase().includes('kenji') || s.toLowerCase().includes('japanese')) {
      return { displayName: `${s} 🇯🇵`, avatar: '🧑‍🎓', isAI: false };
    }
    if (s.toLowerCase().includes('aarav') || s.toLowerCase().includes('priya') || s.toLowerCase().includes('hindi')) {
      return { displayName: `${s} 🇮🇳`, avatar: '👨‍🎓', isAI: false };
    }
    if (s.toLowerCase().includes('sarah') || s.toLowerCase().includes('english')) {
      return { displayName: `${s} 🇬🇧`, avatar: '👩‍🎓', isAI: false };
    }
    return { displayName: s, avatar: '👤', isAI: false };
  }

  /**
   * Adds a new real-time subtitle card to the stream.
   * 
   * Algorithm:
   * 1. Extract and sanitize speaker, text, transliteration, and tri-lingual translations.
   * 2. Format speaker metadata and timestamp.
   * 3. Construct macOS frosted acrylic subtitle card DOM structure.
   * 4. Append to container and prune oldest items if exceeding maxEntries.
   * 5. Smoothly scroll container to the bottom.
   * 
   * @param {Object} data - Subtitle event payload
   */
  addSubtitle(data) {
    if (!this.container) return;

    // Step 1: Extract payload fields
    const speaker = data.speaker || 'Anonymous';
    const originalText = data.original_text || data.text || '';
    const transliteration = data.transliteration || data.romaji || '';
    const translationEn = data.translation_en || '';
    const translationJa = data.translation_ja || '';
    const translationHi = data.translation_hi || '';
    const timestampStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    if (!originalText) return;

    // Step 2: Speaker metadata
    const speakerInfo = this.formatSpeaker(speaker);

    // Step 3: Build card element
    const entryEl = document.createElement('div');
    entryEl.className = `subtitle-entry ${speakerInfo.isAI ? 'ai-mediator' : ''}`;

    let transliterationHtml = '';
    if (transliteration && transliteration !== originalText) {
      transliterationHtml = `
        <div class="subtitle-transliteration-pill">
          <span>🔤</span>
          <span>${this.escapeHtml(transliteration)}</span>
        </div>
      `;
    }

    let translationsHtml = '';
    const translationLines = [];
    if (translationEn && translationEn !== originalText) {
      translationLines.push(`<div class="translation-line"><span class="lang-flag-pill">EN</span><span>${this.escapeHtml(translationEn)}</span></div>`);
    }
    if (translationJa && translationJa !== originalText) {
      translationLines.push(`<div class="translation-line"><span class="lang-flag-pill">JA</span><span>${this.escapeHtml(translationJa)}</span></div>`);
    }
    if (translationHi && translationHi !== originalText) {
      translationLines.push(`<div class="translation-line"><span class="lang-flag-pill">HI</span><span>${this.escapeHtml(translationHi)}</span></div>`);
    }

    if (translationLines.length > 0) {
      translationsHtml = `
        <div class="subtitle-translations-box">
          ${translationLines.join('')}
        </div>
      `;
    }

    entryEl.innerHTML = `
      <div class="subtitle-header">
        <div class="subtitle-speaker-badge">
          <span>${speakerInfo.avatar}</span>
          <span>${this.escapeHtml(speakerInfo.displayName)}</span>
        </div>
        <div class="subtitle-timestamp">${timestampStr}</div>
      </div>
      <div class="subtitle-original-text">${this.escapeHtml(originalText)}</div>
      ${transliterationHtml}
      ${translationsHtml}
    `;

    // Step 4: Append & prune
    this.container.appendChild(entryEl);
    this.history.push(data);

    while (this.container.children.length > this.maxEntries) {
      this.container.removeChild(this.container.firstChild);
    }

    // Step 5: Auto-scroll
    this.container.scrollTop = this.container.scrollHeight;
  }

  /**
   * Clears all subtitle cards from the stream.
   */
  clear() {
    if (this.container) {
      this.container.innerHTML = '';
    }
    this.history = [];
  }

  /**
   * Helper to escape HTML characters.
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
