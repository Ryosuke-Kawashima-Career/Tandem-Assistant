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
   * 2. Render one card per note, newest first, flagging unconfirmed ones.
   *
   * A full redraw rather than incremental DOM patching: the list is short, and a redraw
   * cannot drift out of step with the map the way targeted edits can.
   */
  render() {
    if (!this.container) return;

    if (this.countBadge) this.countBadge.textContent = String(this.notes.size);
    if (this.emptyHint) this.emptyHint.classList.toggle('hidden', this.notes.size > 0);

    const cards = [...this.notes.values()]
      .sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0))
      .map((note) => this.renderCard(note))
      .join('');

    const hint = this.emptyHint ? this.emptyHint.outerHTML : '';
    this.container.innerHTML = hint + cards;
    this.emptyHint = this.container.querySelector('#notes-empty-hint');
    if (this.emptyHint) this.emptyHint.classList.toggle('hidden', this.notes.size > 0);
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
          ${unconfirmed ? '<span class="note-status-pill">needs confirmation</span>' : ''}
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
