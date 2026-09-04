/**
 * Summary:
 *   ReferenceCard.js renders what the agent's tools produced (REQ-18–20): a researched
 *   reference or task-material card, a scheduled follow-up meeting, and the notice a
 *   tool leaves behind when it is unavailable or failed.
 *
 *   All three share one surface deliberately. They are answers to something a
 *   participant asked for - "what does this mean", "book the follow-up" - and a person
 *   who clicked a button looks in one place for the result, including when the result is
 *   that the server cannot do it (REQ-20 forbids a silent no-op).
 *
 * Key Class:
 *   - ReferenceCard: DOM controller for reference, meeting, and tool-status cards.
 */

export class ReferenceCard {
  /**
   * Initialize the ReferenceCard controller.
   *
   * @param {HTMLElement|string} container - Target DOM container or selector
   */
  constructor(container) {
    this.container = typeof container === 'string'
      ? document.querySelector(container)
      : container;

    this.maxCards = 8;
  }

  /**
   * Renders one researched reference or material card (REQ-18).
   *
   * Algorithm:
   * 1. Read the topic and the results out of the enveloped payload.
   * 2. Render each result as a title, a snippet, and a link to the source.
   * 3. Show the thumbnail when the result carries one - a work-mode material card is
   *    about a document, and a picture of it says more than its filename.
   *
   * Every link opens in a new tab with `noopener`: these URLs come from a search engine,
   * so the page behind one is not this application's and must not be handed a reference
   * back to it.
   *
   * @param {Object} payload - `reference.card` event payload
   */
  renderReference(payload) {
    const card = payload?.card;
    if (!this.container || !card) return;

    const results = Array.isArray(card.results) ? card.results : [];
    const badge = card.materials ? '📎 Task Material' : '🔎 Reference';
    const requester = card.requested_by === 'assistant'
      ? 'looked up by the AI co-teacher'
      : '';

    const body = results.length
      ? results.map((result) => this.renderResult(result)).join('')
      : `<div class="idiom-meaning">No results found for this lookup.</div>`;

    this.prepend(`
      <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <span class="idiom-badge">${badge}</span>
        <button class="traffic-light traffic-close" style="width: 10px; height: 10px; border: none; outline: none; cursor: pointer;" title="Dismiss"></button>
      </div>
      <div class="idiom-phrase">${this.escapeHtml(card.query || '')}</div>
      ${requester ? `<div class="idiom-romaji">🤖 ${requester}</div>` : ''}
      ${body}
    `);
  }

  /**
   * Renders one search result inside a reference card.
   */
  renderResult(result) {
    const title = this.escapeHtml(result?.title || 'Untitled result');
    const snippet = this.escapeHtml(result?.snippet || '');
    const url = this.safeUrl(result?.url);
    const image = this.safeUrl(result?.image_url);

    return `
      <div class="reference-result">
        ${image ? `<img class="reference-thumb" src="${image}" alt="" loading="lazy" />` : ''}
        <div class="idiom-meaning">
          ${url
            ? `<a href="${url}" target="_blank" rel="noopener noreferrer">${title}</a>`
            : title}
        </div>
        ${snippet ? `<div class="idiom-cultural-note">${snippet}</div>` : ''}
      </div>
    `;
  }

  /**
   * Renders a scheduled follow-up meeting (REQ-20).
   */
  renderMeeting(payload) {
    const meeting = payload?.meeting;
    if (!this.container || !meeting) return;

    const attendees = Array.isArray(meeting.attendees) ? meeting.attendees : [];
    const link = this.safeUrl(meeting.html_link);

    this.prepend(`
      <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <span class="idiom-badge">📅 Meeting Scheduled</span>
        <button class="traffic-light traffic-close" style="width: 10px; height: 10px; border: none; outline: none; cursor: pointer;" title="Dismiss"></button>
      </div>
      <div class="idiom-phrase">${this.escapeHtml(meeting.summary || 'Follow-up session')}</div>
      <div class="idiom-meaning">${this.escapeHtml(meeting.start_time || '')} → ${this.escapeHtml(meeting.end_time || '')}</div>
      <div class="idiom-cultural-note">
        ${attendees.length
          ? `Invitations sent to ${this.escapeHtml(attendees.join(', '))}`
          : 'No attendee addresses were supplied, so nobody was invited.'}
      </div>
      ${link ? `<div class="idiom-meaning"><a href="${link}" target="_blank" rel="noopener noreferrer">Open in Google Calendar</a></div>` : ''}
    `);
  }

  /**
   * Renders an Anki export receipt (REQ-19).
   */
  renderExport(payload) {
    const exported = payload?.export;
    if (!this.container || !exported) return;

    const terms = Array.isArray(exported.terms) ? exported.terms : [];

    this.prepend(`
      <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <span class="idiom-badge">🗂️ Anki Export</span>
        <button class="traffic-light traffic-close" style="width: 10px; height: 10px; border: none; outline: none; cursor: pointer;" title="Dismiss"></button>
      </div>
      <div class="idiom-phrase">${exported.exported || 0} card(s) → ${this.escapeHtml(exported.deck || 'default deck')}</div>
      ${terms.length ? `<div class="idiom-cultural-note">${this.escapeHtml(terms.join(' · '))}</div>` : ''}
    `);
  }

  /**
   * Renders a tool that could not run (REQ-18–20).
   *
   * Shown rather than logged: someone pressed a button, and an unavailable tool that
   * says nothing is indistinguishable from one that worked.
   */
  renderStatus(payload) {
    if (!this.container || !payload || payload.state === 'ok') return;

    const label = payload.state === 'unavailable' ? 'Not configured' : 'Failed';

    this.prepend(`
      <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <span class="idiom-badge">⚠️ ${this.escapeHtml(payload.tool || 'tool')}: ${label}</span>
        <button class="traffic-light traffic-close" style="width: 10px; height: 10px; border: none; outline: none; cursor: pointer;" title="Dismiss"></button>
      </div>
      <div class="idiom-meaning">${this.escapeHtml(payload.reason || 'No reason was given.')}</div>
    `);
  }

  /**
   * Inserts one card at the top of the container and prunes the overflow.
   */
  prepend(innerHtml) {
    const cardEl = document.createElement('div');
    cardEl.className = 'idiom-bubble reference-bubble';
    cardEl.innerHTML = innerHtml;

    const closeBtn = cardEl.querySelector('.traffic-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => {
        if (cardEl.parentNode) cardEl.parentNode.removeChild(cardEl);
      });
    }

    this.container.prepend(cardEl);
    while (this.container.children.length > this.maxCards) {
      this.container.removeChild(this.container.lastChild);
    }
  }

  /**
   * Clears every card.
   */
  clear() {
    if (this.container) this.container.innerHTML = '';
  }

  /**
   * Returns a URL only when it is one a browser should follow.
   *
   * `javascript:` and `data:` URLs reaching an `href` from remote content is the
   * classic injection route, and these URLs arrive from a search engine's index.
   */
  safeUrl(value) {
    const url = String(value || '').trim();
    if (!/^https?:\/\//i.test(url)) return '';
    return this.escapeHtml(url);
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
