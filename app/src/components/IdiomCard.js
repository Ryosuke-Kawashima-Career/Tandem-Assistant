/**
 * Summary:
 *   IdiomCard.js manages real-time cultural idiom cards, slang annotations,
 *   and etiquette scaffolding in the EchoSphere Tandem Co-Teacher client.
 *   When the AI Co-Teacher detects nuanced phrases (such as Japanese Yojijukugo or
 *   Hindi honorific registers), it renders a luminous macOS acrylic card
 *   with Romaji transliteration, literal meaning, and cultural context.
 *
 * Key Class:
 *   - IdiomCard: DOM controller managing idiom popups, animated entries, and dismiss actions.
 */

export class IdiomCard {
  /**
   * Initialize IdiomCard controller.
   * 
   * Algorithm:
   * 1. Store container DOM element.
   * 2. Initialize active cards registry.
   * 
   * @param {HTMLElement|string} container - Target DOM container or selector
   */
  constructor(container) {
    this.container = typeof container === 'string'
      ? document.querySelector(container)
      : container;
    
    this.maxCards = 10;
  }

  /**
   * Renders a new cultural idiom / scaffolding card.
   * 
   * Algorithm:
   * 1. Extract phrase, romaji transliteration, meaning, and cultural note from payload.
   * 2. Build macOS styled card with close trigger and badge.
   * 3. Prepend card to container with pop-in spring animation.
   * 4. Enforce maxCards limit by removing excess trailing cards.
   * 
   * @param {Object} payload - Idiom card data { phrase, romaji, meaning, cultural_note }
   */
  renderIdiom(payload) {
    if (!this.container) return;

    // Step 1: Extract fields
    const phrase = payload.phrase || 'Cultural Annotation';
    const romaji = payload.romaji || payload.transliteration || '';
    const meaning = payload.meaning || '';
    const culturalNote = payload.cultural_note || '';

    // Step 2: Build card DOM element
    const cardEl = document.createElement('div');
    cardEl.className = 'idiom-bubble';

    let romajiHtml = '';
    if (romaji) {
      romajiHtml = `<div class="idiom-romaji">🔤 ${this.escapeHtml(romaji)}</div>`;
    }

    let noteHtml = '';
    if (culturalNote) {
      noteHtml = `<div class="idiom-cultural-note">💡 ${this.escapeHtml(culturalNote)}</div>`;
    }

    cardEl.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <span class="idiom-badge">✨ Cultural Context</span>
        <button class="traffic-light traffic-close" style="width: 10px; height: 10px; border: none; outline: none; cursor: pointer;" title="Dismiss"></button>
      </div>
      <div class="idiom-phrase">${this.escapeHtml(phrase)}</div>
      ${romajiHtml}
      <div class="idiom-meaning">${this.escapeHtml(meaning)}</div>
      ${noteHtml}
    `;

    // Step 3: Attach dismiss handler
    const closeBtn = cardEl.querySelector('.traffic-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => {
        cardEl.style.transition = 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)';
        cardEl.style.opacity = '0';
        cardEl.style.transform = 'scale(0.92) translateY(-6px)';
        setTimeout(() => {
          if (cardEl.parentNode) {
            cardEl.parentNode.removeChild(cardEl);
          }
        }, 200);
      });
    }

    // Step 4: Insert at beginning
    this.container.prepend(cardEl);

    // Step 5: Prune excess cards
    while (this.container.children.length > this.maxCards) {
      this.container.removeChild(this.container.lastChild);
    }
  }

  /**
   * Clears all idiom cards.
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
