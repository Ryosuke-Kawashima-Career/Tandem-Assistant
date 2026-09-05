/**
 * Summary:
 *   NotesPanel.js renders the session notes captured from the conversation (REQ-14).
 *   It consumes the `note.upserted` and `note.deleted` RTC data events, keyed by the
 *   note id so a revised note replaces its earlier version in place rather than
 *   appending a duplicate - the backend deduplicates on the same key, and the panel
 *   must agree with it or the two drift apart on every regeneration.
 *
 *   Note types differ by session mode: vocabulary/correction/grammar/culture/example/goal
 *   in `language_learning`, term/decision/action/risk/open_question/glossary in
 *   `international_work`. The panel does not police the vocabulary; the server already
 *   dropped anything typed for the other mode.
 *
 * Key Class:
 *   - NotesPanel: renders, revises, and removes note cards.
 */

const TYPE_ICONS = {
  vocabulary: '🔤',
  correction: '✏️',
  grammar: '📐',
  culture: '🌏',
  example: '💬',
  goal: '🎯',
  term: '📘',
  decision: '✅',
  action: '⚡',
  risk: '⚠️',
  open_question: '❓',
  glossary: '📖'
};

export class NotesPanel {
  /**
   * Initialize the notes panel.
   *
   * Algorithm:
   * 1. Resolve the container, empty-state hint, and count badge elements.
   * 2. Initialize the id -> note map that backs the rendered list.
   *
   * @param {string} containerSelector - Selector for the notes list container
   * @param {Object} options - Optional `emptyHint` and `countBadge` selectors
   */
  constructor(containerSelector, options = {}) {
    this.container = document.querySelector(containerSelector);
    this.emptyHint = document.querySelector(options.emptyHint || '#notes-empty-hint');
    this.countBadge = document.querySelector(options.countBadge || '#notes-count');
    this.notes = new Map();
    this.onDelete = null;

    // Delegated rather than bound per card: the list is redrawn on every event, and
    // per-card listeners would be re-attached (and leak) on each redraw.
    if (this.container) {
      this.container.addEventListener('click', (event) => {
        const button = event.target.closest('[data-note-delete]');
        if (button && this.onDelete) {
          this.onDelete(button.getAttribute('data-note-delete'));
        }
      });
    }
  }

  /**
   * Inserts or replaces one note (REQ-14).
   *
   * @param {Object} payload - `note.upserted` event payload, enveloped by the server
   */
  upsert(payload) {
    const note = payload?.note;
    if (!note?.id) return;

    this.notes.set(note.id, note);
    this.render();
  }

  /**
   * Removes a deleted note from the panel.
   *
   * @param {Object} payload - `note.deleted` event payload
   */
  remove(payload) {
    const note = payload?.note;
    if (!note?.id) return;

    this.notes.delete(note.id);
    this.render();
  }

  /**
   * Drops every rendered note, e.g. when the session ends or the feed is cleared.
   */
  clear() {
    this.notes.clear();
    this.render();
  }

  /**
   * Redraws the list from the current note map.
   *
   * Algorithm:
   * 1. Toggle the empty-state hint and the count badge.
   * 2. Diff the desired note order against the cards already in the DOM, keyed by
   *    `data-note-id`: build a card only for a note that is new or whose rendered
   *    markup changed, reuse the existing element otherwise, and drop cards whose
   *    note is gone.
   * 3. Reorder the surviving elements in place to match newest-first.
   *
   * Keyed rather than a blanket `innerHTML` rebuild: the panel is redrawn on every
   * artifact poll (~5s) as well as on every note event, and a rebuild re-creates every
   * card, so each poll replayed the `macos-pop-in` entrance on notes that had not
   * changed - the whole list flashing every few seconds. Reusing an unchanged note's
   * element leaves its animation finished and untouched, so the entrance now marks what
   * it is supposed to mark: a note that genuinely just arrived.
   */
  render() {
    if (!this.container) return;

    if (this.countBadge) this.countBadge.textContent = String(this.notes.size);
    if (this.emptyHint) this.emptyHint.classList.toggle('hidden', this.notes.size > 0);

    const ordered = [...this.notes.values()]
      .sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));

    // Step 2: index what is already on screen, then claim or rebuild one card per note.
    const existing = new Map();
    this.container.querySelectorAll('[data-note-id]').forEach((el) => {
      existing.set(el.getAttribute('data-note-id'), el);
    });

    const elements = ordered.map((note) => {
      const html = this.renderCard(note);
      const current = existing.get(String(note.id));
      if (current) {
        existing.delete(String(note.id));
        // Compared as markup rather than by a revision field: the note contract has no
        // version, and re-rendering a card whose text is identical is exactly the
        // needless entrance replay this method exists to avoid.
        if (current.outerHTML.trim() === html.trim()) return current;
        const replacement = this.buildCard(html);
        if (replacement) {
          current.replaceWith(replacement);
          return replacement;
        }
        return current;
      }
      return this.buildCard(html);
    }).filter(Boolean);

    // Anything left in `existing` belongs to a note that was deleted.
    existing.forEach((el) => el.remove());

    // Step 3: put the cards in order. `append` moves an element already in the
    // container rather than cloning it, so a reused card keeps its finished animation.
    elements.forEach((el) => this.container.append(el));
  }

  /**
   * Turns one card's markup into a detached element.
   *
   * @param {string} html - Markup from `renderCard`
   * @returns {?HTMLElement} The card element, or `null` if the markup produced none
   */
  buildCard(html) {
    const template = document.createElement('template');
    template.innerHTML = html.trim();
    return template.content.firstElementChild;
  }

  /**
   * Builds the markup for one note card.
   *
   * @param {Object} note - Serialized NoteItem
   * @returns {string} HTML for the card
   */
  renderCard(note) {
    const icon = TYPE_ICONS[note.type] || '📝';
    const unconfirmed = note.status === 'needs_confirmation';
    const edited = note.status === 'edited';

    const meta = [
      note.owner ? `👤 ${this.escape(note.owner)}` : '',
      note.due_at ? `📅 ${this.escape(note.due_at)}` : '',
      edited ? '✍️ edited' : ''
    ].filter(Boolean).join(' · ');

    return `
      <div class="note-card${unconfirmed ? ' needs-confirmation' : ''}" data-note-id="${this.escape(note.id)}">
        <div class="note-card-header">
          <span class="note-type-badge">${icon} ${this.escape(note.type)}</span>
          <span class="note-card-actions">
            ${unconfirmed ? '<span class="note-status-pill">needs confirmation</span>' : ''}
            <button class="note-delete-btn" data-note-delete="${this.escape(note.id)}" title="Delete this note">✕</button>
          </span>
        </div>
        <div class="note-text">${this.escape(note.text)}</div>
        ${meta ? `<div class="note-meta">${meta}</div>` : ''}
      </div>
    `;
  }

  /**
   * Escapes text before it is inserted as HTML.
   *
   * Note text is model-generated from live speech, so it is never trusted as markup.
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
