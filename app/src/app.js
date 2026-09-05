/**
 * Summary:
 *   app.js is the main client entry point for the EchoSphere Tandem Co-Teacher Web Client.
 *   It coordinates:
 *     1. Agora RTC Web SDK connection (microphone capture, SD-RTN voice channel join/leave).
 *     2. Agora RTC Data Stream listener via AgoraStreamManager.
 *     3. UI Components: Subtitles, IdiomCard, TopicWidget, TeacherBar, QuizWidget.
 *     4. Session mode switcher (Language Learning vs International Work - REQ-12).
 *     5. Direct spoken conversation with the Convo AI co-teacher (REQ-09).
 *     6. Interactive multi-turn demo simulation covering Japanese, Hindi, and English exchange.
 *
 * Key Controllers:
 *   - TandemApp: Manages overall state, event dispatching, and device lifecycles.
 */

import AgoraRTC from 'agora-rtc-sdk-ng';
import { AgoraStreamManager } from './services/agoraStream.js';
import { ConvoAIService } from './services/convoai.js';
import { requestJson } from './services/http.js';
import { ConvoAITranscriptService } from './services/convoaiTranscript.js';
import { SessionEventService } from './services/sessionEvents.js';
import { Subtitles } from './components/Subtitles.js';
import { IdiomCard } from './components/IdiomCard.js';
import { TopicWidget } from './components/TopicWidget.js';
import { TeacherBar } from './components/TeacherBar.js';
import { QuizWidget } from './components/QuizWidget.js';
import { NotesPanel } from './components/NotesPanel.js';
import { ReferenceCard } from './components/ReferenceCard.js';

// How often stored notes and quizzes are re-read while a session is live. Chosen to be
// fast enough that "automatic" still feels automatic, and slow enough that a session
// left open for an hour costs a few hundred cheap reads rather than thousands.
const ARTIFACT_POLL_INTERVAL_MS = 5000;

// Video-input labels that routinely answer getUserMedia successfully while showing
// nothing (D-UIUX-4). An infrared Windows Hello sensor returns a near-black frame under
// ordinary room light; a virtual camera returns black whenever the software behind it is
// not producing frames. Both open cleanly, report a real frame size, and even yield a
// plausible vision description - so nothing this app can measure distinguishes them from
// a working camera. The device label is the only signal that does.
const DECOY_CAMERA_PATTERN = /\b(ir|infrared|depth|virtual|mirametrix|obs|snap|droidcam)\b/i;

// How often a frame is pushed for the agent to look at while Camera Assist is open
// (REQ-CAM-01). Frequent enough that what the agent sees is what the participant is
// currently holding up, infrequent enough to stay well clear of the live audio the call
// depends on (REQ-LAT-01). It also has to stay comfortably inside the server's own
// freshness window (CAMERA_FRAME_TTL_SECONDS, 15s), or a camera left open would keep
// going briefly blind between pushes.
const CAMERA_STREAM_INTERVAL_MS = 3000;

// The longer edge of a pushed frame, in pixels. Deliberately a fraction of
// captureCameraFrame()'s full-resolution capture: that one is a participant asking for
// the best possible reading of a page, this one is a background push every few seconds
// on a connection already carrying a voice call (Risk 5). It is still large enough for a
// vision model to name an object and read reasonably sized text.
const CAMERA_STREAM_MAX_EDGE = 640;

// JPEG quality for those pushes, below the on-demand capture's 0.85 for the same reason.
const CAMERA_STREAM_QUALITY = 0.7;

class TandemApp {
  /**
   * Initialize Tandem Application.
   * 
   * Algorithm:
   * 1. Initialize client state (channel, session mode, joined status, mic status).
   * 2. Instantiate UI component controllers.
   * 3. Attach AgoraStreamManager and wire stream event listeners.
   * 4. Bind DOM UI event listeners.
   */
  constructor() {
    // Step 1: Client State
    this.appId = 'mock_agora_app_id';
    this.channelName = 'tokyo-mumbai-101';
    // Session mode (REQ-12). Immutable once a session starts: the backend refuses a
    // mid-session change, so the switcher locks itself while a conversation is live.
    this.currentMode = 'language_learning';
    this.isJoined = false;
    this.isMuted = false;
    this.localAudioTrack = null;
    this.agoraClient = null;
    this.localUid = Math.floor(Math.random() * 100000) + 1000;

    // Identity the backend authorizes artifact access against (REQ-16). It is also the
    // speaker id sent with a Convo AI session, so the participant recorded on the
    // session and the actor asking for its notes are the same person.
    this.speakerId = 'Learner';

    // Convo AI state (REQ-09)
    this.convoai = new ConvoAIService(this.channelName);
    this.isAiActive = false;
    this.isAiPending = false;
    this.remoteAudioTracks = new Map();

    // True only when a real RTC join published a live microphone track. A spoken AI
    // conversation is impossible without this, so it gates startConvoAI().
    this.hasLiveAudio = false;

    // Gemini Live translated audio (REQ-17). Defaults per mode - on for international
    // work, off for language learning - and is re-derived whenever the mode changes,
    // until the participant makes a choice of their own. The server owns the real gate;
    // this only mirrors it, because the router decides who is published to.
    this.translatedAudioEnabled = false;
    this.translatedAudioChosen = false;
    this.translationLegs = {};

    // Camera assist (REQ-22). The stream lives only while the panel is open; a capture is
    // taken on demand and uploaded as one frame, so nothing is recorded or held.
    this.cameraStream = null;
    this.isCameraOn = false;
    // The periodic push that lets the co-teacher answer "what is this?" mid-conversation
    // (REQ-CAM-01). It exists only between toggleCamera() on and stopCamera(): the
    // interval is created there and cleared there, so a closed panel captures nothing.
    this.cameraStreamHandle = null;
    this.cameraPushInFlight = false;
    // The camera actually being shown (D-UIUX-4). Remembered across open/close so a
    // participant who had to correct the browser's choice only corrects it once.
    this.cameraDeviceId = null;

    // Session timer (REQ-23). Client-side and approximate on purpose: it is a motivation
    // cue, and the authoritative duration is the stored session's own timestamps.
    this.sessionStartedAt = null;
    this.sessionTimerHandle = null;

    // Measured local speaking time (REQ-23). Sampled from this participant's own
    // microphone track, which is the only place a real duration exists client-side, and
    // reported to the server in batches - the server does the accumulating and the
    // publishing, so every participant sees one agreed balance.
    this.speechSampleHandle = null;
    this.unreportedSpeechMs = 0;

    // Session lifecycle (REQ-12 / D-UIUX-1). Joining the channel is what creates the
    // backend session every session-governed endpoint resolves against; without it,
    // search, direct query, camera assist, and export all answer "No such session."
    this.sessionId = null;

    // REST fallback for generated artifacts (D-UIUX-2). Notes and quizzes are generated
    // server-side and stored, but the RTC data-stream that is supposed to announce them
    // does not reach a real browser yet, so they are also polled while a session is live.
    // Quiz ids are tracked because QuizWidget appends a card per call and has no id map
    // of its own; NotesPanel already dedupes by note id.
    this.artifactPollHandle = null;
    this.renderedQuizIds = new Set();

    // Live transcripts and agent state, delivered over RTM by the Convo AI Engine
    this.transcriptService = new ConvoAITranscriptService();
    this.rtcCredentials = null;
    this.setupTranscriptListeners();

    // Step 2: Initialize UI Components
    this.subtitles = new Subtitles('#subtitles-container');
    this.idiomCard = new IdiomCard('#scaffolding-container');
    // Agent tools (REQ-18-20): reference/material cards, meeting receipts, and the
    // notice an unavailable tool leaves. Shares the scaffolding column, because all of
    // it answers the same question - what did the assistant just produce.
    this.referenceCard = new ReferenceCard('#scaffolding-container');
    // Which tools this server actually holds credentials for, so a control is not
    // offered when its only possible outcome is a 503.
    this.availableTools = {};
    this.quizWidget = new QuizWidget('#scaffolding-container');
    this.notesPanel = new NotesPanel('#notes-container', {
      emptyHint: '#notes-empty-hint',
      countBadge: '#notes-count'
    });
    this.notesPanel.onDelete = (noteId) => this.deleteNote(noteId);
    this.topicWidget = new TopicWidget({
      topicTitle: '#topic-title',
      topicPrompt: '#topic-prompt',
      segA: '#balance-seg-a',
      segB: '#balance-seg-b',
      labelA: '#balance-label-a',
      labelB: '#balance-label-b',
      statusText: '#balance-status'
    });
    this.teacherBar = new TeacherBar({
      dock: '#teacher-oversight-dock',
      alertPill: '#teacher-alert-pill',
      alertText: '#teacher-alert-text'
    });

    // Step 3: Stream Manager
    this.streamManager = new AgoraStreamManager();
    this.setupStreamListeners();

    // Live session events (D-UIUX-2). Feeds the same stream manager the RTC data
    // stream was always supposed to, so every widget above is unaffected by which
    // transport actually carried the event.
    this.sessionEvents = new SessionEventService(this.streamManager);

    // Step 4: UI Event Handlers
    this.bindDomEvents();
    this.setupTeacherBarActions();
    this.checkBackendConnection();
    this.refreshToolAvailability();

    console.log('🚀 EchoSphere Tandem Client initialized with macOS Light Theme.');
  }

  /**
   * Pings the EchoSphere backend server to verify live status.
   */
  async checkBackendConnection() {
    try {
      const data = await requestJson('/health');
      const connStatus = document.getElementById('connection-status');
      const connDot = document.getElementById('connection-dot');
      if (connStatus) connStatus.textContent = `Backend Live (v${data.version})`;
      if (connDot) connDot.style.backgroundColor = 'var(--macos-accent-green)';
      console.log('✅ Connected to EchoSphere Backend API:', data);
    } catch (err) {
      console.log('ℹ️ Running in standalone offline mode:', err.message);
    }
  }

  /**
   * Configures event listeners for incoming RTC Data Stream events.
   * 
   * Algorithm:
   * 1. Listen for 'subtitles' event and route payload to Subtitles component.
   * 2. Listen for 'idiom_card' event and route payload to IdiomCard component.
   * 3. Listen for 'topic_prompt' event and route payload to TopicWidget.
   * 4. Listen for 'speaking_balance' event and update balance progress bar.
   * 5. Listen for 'quiz' event and render interactive quiz widget.
   * 6. Listen for 'teacher_alert' event and update teacher alert banner.
   */
  setupStreamListeners() {
    // Subtitles
    this.streamManager.on('subtitles', (payload) => {
      this.subtitles.addSubtitle(payload);
    });

    // Idiom Card
    this.streamManager.on('idiom_card', (payload) => {
      this.idiomCard.renderIdiom(payload);
    });

    // Topic Prompt
    this.streamManager.on('topic_prompt', (payload) => {
      this.topicWidget.updateTopic(payload);
    });

    // Speaking Balance
    this.streamManager.on('speaking_balance', (payload) => {
      if (payload.speaker_percentages) {
        this.topicWidget.updateSpeakingBalance(payload.speaker_percentages);
      }
    });

    // Quiz Widget
    this.streamManager.on('quiz', (payload) => {
      this.quizWidget.renderQuiz(payload);
    });

    // Teacher Alert
    this.streamManager.on('teacher_alert', (payload) => {
      this.teacherBar.showAlert(payload);
    });

    // Session artifacts (REQ-13 / REQ-14). These carry an enveloped entity rather than
    // a widget payload: {schema_version, event_id, session_id, mode, quiz|note}.
    this.streamManager.on('quiz.created', (payload) => {
      if (payload?.quiz) this.quizWidget.renderQuiz(this.quizFromArtifact(payload.quiz));
    });

    this.streamManager.on('note.upserted', (payload) => {
      this.notesPanel.upsert(payload);
    });

    this.streamManager.on('note.deleted', (payload) => {
      this.notesPanel.remove(payload);
    });

    // Gemini Live Translate (REQ-17). Status carries either a leg's state or one
    // participant's audio-gate state; transcripts arrive whatever the gate says, which
    // is what keeps them usable as an on-demand subtitle.
    this.streamManager.on('translation.status', (payload) => {
      this.handleTranslationStatus(payload);
    });

    this.streamManager.on('translation.output_transcript', (payload) => {
      this.handleTranslationTranscript(payload);
    });

    // Agent tools (REQ-18-20). Enveloped like the artifact events:
    // {schema_version, event_id, session_id, mode, tool, card|export|meeting}.
    this.streamManager.on('reference.card', (payload) => {
      this.referenceCard.renderReference(payload);
    });

    this.streamManager.on('meeting.scheduled', (payload) => {
      this.referenceCard.renderMeeting(payload);
    });

    this.streamManager.on('anki.exported', (payload) => {
      this.referenceCard.renderExport(payload);
    });

    this.streamManager.on('tool.status', (payload) => {
      this.referenceCard.renderStatus(payload);
    });
  }

  /**
   * Asks the backend which agent tools it can actually run (REQ-18-20).
   *
   * A control whose tool has no credentials is disabled rather than hidden: the feature
   * exists, this deployment just has not configured it, and a disabled button with a
   * tooltip says that where a missing one says nothing.
   */
  async refreshToolAvailability() {
    try {
      const body = await requestJson('/api/tools/status');
      this.availableTools = body?.tools || {};
    } catch (err) {
      console.log('ℹ️ Agent tool status unavailable:', err.message);
      this.availableTools = {};
    }
    this.updateToolControls();
  }

  /**
   * Enables or disables each agent-tool control from the reported availability.
   */
  updateToolControls() {
    const controls = [
      ['btn-research', 'search', 'Google Search is not configured on this server.'],
      ['btn-schedule', 'calendar', 'Google Calendar is not configured on this server.'],
      ['btn-camera', 'vision', 'Camera assist needs GEMINI_API_KEY on this server.'],
      // Anki lives in the export selector rather than in its own toolbar button, but
      // the availability rule is the same: an option nothing backs is disabled, not
      // hidden, so the feature is visible as unconfigured rather than absent.
      ['export-option-anki', 'anki', 'No Anki MCP server is configured on this server.'],
    ];

    controls.forEach(([id, tool, unavailableHint]) => {
      const btn = document.getElementById(id);
      if (!btn) return;
      const available = Boolean(this.availableTools?.[tool]);
      btn.disabled = !available;
      if (!available) btn.title = unavailableHint;
    });
  }

  /**
   * Shows the inline prompt dialog and resolves with what was entered, or `null` if
   * cancelled (D-UIUX-6).
   *
   * Not `window.prompt()`: several browsers and embedding contexts suppress native
   * dialogs silently - no dialog shown, an immediate `null` return - which made a
   * suppressed prompt and a cancelled one look identical to a completely broken button,
   * with no request ever sent and nothing in a server log to tell the two apart.
   *
   * @param {string} message - What is being asked for
   * @param {string} [defaultValue] - Pre-filled input value
   * @returns {Promise<string|null>}
   */
  promptModal(message, defaultValue = '') {
    return new Promise((resolve) => {
      const overlay = document.getElementById('prompt-modal-overlay');
      const messageEl = document.getElementById('prompt-modal-message');
      const input = document.getElementById('prompt-modal-input');
      const okBtn = document.getElementById('prompt-modal-ok');
      const cancelBtn = document.getElementById('prompt-modal-cancel');
      if (!overlay || !messageEl || !input || !okBtn || !cancelBtn) {
        resolve(null);
        return;
      }

      messageEl.textContent = message;
      input.value = defaultValue;
      overlay.classList.remove('hidden');
      input.focus();

      const cleanup = (result) => {
        overlay.classList.add('hidden');
        okBtn.removeEventListener('click', onOk);
        cancelBtn.removeEventListener('click', onCancel);
        input.removeEventListener('keydown', onKeydown);
        resolve(result);
      };
      const onOk = () => cleanup(input.value);
      const onCancel = () => cleanup(null);
      const onKeydown = (event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          onOk();
        } else if (event.key === 'Escape') {
          event.preventDefault();
          onCancel();
        }
      };

      okBtn.addEventListener('click', onOk);
      cancelBtn.addEventListener('click', onCancel);
      input.addEventListener('keydown', onKeydown);
    });
  }

  /**
   * Researches a topic for the current session and publishes the card (REQ-18).
   *
   * The query is asked for rather than inferred: a participant pressing "Research" has
   * something specific in mind, and guessing it from the last subtitle produces a card
   * about whatever happened to be said most recently.
   */
  async researchTopic() {
    const query = await this.promptModal('What should the AI co-teacher look up?');
    if (!query || !query.trim()) return;

    try {
      const body = await requestJson('/api/tools/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          channel: this.channelName,
          actor: this.speakerId,
          query: query.trim()
        })
      });
      // The card also arrives over the RTC data stream, but that transport is not
      // real yet (D-UIUX-2) - rendering the REST response directly, the same way
      // captureCameraFrame() already does for vision, is what makes a lookup that
      // clearly succeeded on the server actually show up on screen.
      this.referenceCard.renderReference({ card: body?.card });
    } catch (err) {
      this.referenceCard.renderStatus({
        tool: 'search',
        state: err.status === 503 ? 'unavailable' : 'failed',
        reason: err.message
      });
      console.warn('🔎 Reference lookup failed:', err.message);
    }
  }

  /**
   * Exports this session's vocabulary and terminology to Anki (REQ-19).
   */
  async exportToAnki() {
    try {
      const body = await requestJson('/api/tools/anki/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel: this.channelName, actor: this.speakerId })
      });
      console.log('🗂️ Anki export:', body);
      // The receipt normally arrives as an `anki.exported` data event; rendering it here
      // too is what makes the export visible in a session with no live data stream.
      this.referenceCard.renderExport(body);
    } catch (err) {
      // Shown, not only logged: somebody pressed Export, and a silent failure is
      // indistinguishable from a button that does nothing.
      this.referenceCard.renderNotice('Anki Export', err.message, false);
      console.warn('🗂️ Anki export failed:', err.message);
    }
  }

  /**
   * Books a follow-up meeting for this session (REQ-20).
   *
   * Attendee addresses are collected here because a session records participants by
   * display name; this deployment has no registry mapping those to email addresses.
   */
  async scheduleFollowUp() {
    const startTime = await this.promptModal(
      'Start time for the follow-up (ISO-8601, e.g. 2026-09-10T09:00:00Z):'
    );
    if (!startTime || !startTime.trim()) return;

    const attendeeList = (await this.promptModal('Attendee email addresses (comma separated):')) || '';
    const attendees = attendeeList
      .split(',')
      .map((address) => address.trim())
      .filter(Boolean);

    try {
      const body = await requestJson('/api/tools/calendar/schedule', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          channel: this.channelName,
          actor: this.speakerId,
          start_time: startTime.trim(),
          duration_minutes: 30,
          attendees
        })
      });
      console.log('📅 Meeting scheduled:', body);
    } catch (err) {
      console.warn('📅 Scheduling failed:', err.message);
    }
  }

  /**
   * Asks the AI co-teacher one question of the participant's own (REQ-21).
   *
   * The answer is rendered from this response rather than awaited over the data stream,
   * because it is deliberately not broadcast: a private "what does this word mean" is
   * not part of the conversation both peers are in.
   */
  async askDirectQuery() {
    const input = document.getElementById('query-input');
    const button = document.getElementById('btn-ask');
    const question = (input?.value || '').trim();
    if (!question) return;

    // Locked while in flight so an impatient second Enter does not ask twice, and
    // cleared immediately so the box is ready for the next question either way.
    if (input) { input.value = ''; input.disabled = true; }
    if (button) button.disabled = true;

    try {
      const body = await requestJson('/api/agent/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          channel: this.channelName,
          actor: this.speakerId,
          speaker_id: this.speakerId,
          text: question
        })
      });
      this.referenceCard.renderAnswer(body?.answer);
    } catch (err) {
      // Shown, not only logged: somebody typed a question and pressed a button, and an
      // answer that never appears is indistinguishable from one that was never asked.
      this.referenceCard.renderStatus({
        tool: 'co-teacher',
        state: 'failed',
        reason: `Could not answer "${question}": ${err.message}`
      });
    } finally {
      if (input) { input.disabled = false; input.focus(); }
      if (button) button.disabled = false;
    }
  }

  /**
   * Turns the camera assist panel on or off (REQ-22).
   *
   * The stream is requested when the panel opens and stopped when it closes, rather than
   * held for the session: the camera exists for the moment the participant chose to show
   * something, and a preview left running is a camera nobody remembers is on.
   */
  async toggleCamera() {
    if (this.isCameraOn) {
      this.stopCamera();
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      this.referenceCard.renderStatus({
        tool: 'vision', state: 'unavailable',
        reason: 'This browser exposes no camera API. A secure origin (https or localhost) is required.'
      });
      return;
    }

    try {
      this.cameraStream = await this.openCameraStream(this.cameraDeviceId);
    } catch (err) {
      this.referenceCard.renderStatus({
        tool: 'vision', state: 'unavailable',
        reason: `Camera unavailable: ${err.message}`
      });
      return;
    }

    // Correct one specific silent failure before anything is shown: a facing-mode
    // preference can be satisfied by any front-facing device, and on a machine carrying
    // an IR sensor or a virtual camera beside the real webcam, that choice can land on
    // one that shows nothing. Skipped once the participant has picked a camera
    // themselves - their choice outranks this guess.
    if (!this.cameraDeviceId) {
      const better = await this.findVisibleLightCamera();
      if (better) {
        this.releaseCameraStream();
        try {
          this.cameraStream = await this.openCameraStream(better.deviceId);
          this.cameraDeviceId = better.deviceId;
          console.log(`📷 Using '${better.label}': the camera first offered shows no visible-light image.`);
        } catch (err) {
          // Reopening failed, so fall back to the browser's own pick: a preview the
          // participant can correct with the picker beats no preview at all.
          console.warn('📷 Could not switch camera:', err.message);
          this.cameraStream = await this.openCameraStream(null).catch(() => null);
        }
      }
    }

    if (!this.cameraStream) {
      this.referenceCard.renderStatus({
        tool: 'vision', state: 'unavailable',
        reason: 'No camera could be opened on this machine.'
      });
      return;
    }

    await this.showCameraStream();
    this.isCameraOn = true;
    this.startCameraStreaming();
    await this.populateCameraDevices();
    this.updateCameraUi();
  }

  /**
   * Begins pushing reduced-resolution frames while the panel is open (REQ-CAM-01).
   *
   * This is what lets the co-teacher answer "what is this?" in the middle of a spoken
   * turn: by the time the agent decides to look, a frame is already on the server, so
   * looking costs no round trip back to this browser on the one path the learner waits
   * on in silence (REQ-LAT-02).
   *
   * Strictly scoped to the participant's own opt-in: the interval starts here, when they
   * turned Camera Assist on, and stopCamera() clears it. Nothing is captured, uploaded,
   * or buffered outside that window.
   */
  startCameraStreaming() {
    this.stopCameraStreaming(false);
    this.pushCameraFrame();
    this.cameraStreamHandle = setInterval(
      () => this.pushCameraFrame(), CAMERA_STREAM_INTERVAL_MS
    );
  }

  /**
   * Stops the periodic push and, by default, tells the server to forget the last frame.
   *
   * The explicit forget matters: without it the buffered frame would stay describable
   * until its own TTL ran out, so closing the panel would not immediately stop the agent
   * from seeing - which is exactly what closing the panel means.
   *
   * @param {boolean} clearServer - Whether to ask the server to drop the buffered frame
   */
  stopCameraStreaming(clearServer = true) {
    if (this.cameraStreamHandle) {
      clearInterval(this.cameraStreamHandle);
      this.cameraStreamHandle = null;
    }
    if (!clearServer) return;

    const params = new URLSearchParams({ channel: this.channelName, actor: this.speakerId });
    requestJson(`/api/session/camera/stream?${params.toString()}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        channel: this.channelName, actor: this.speakerId, active: false
      })
    }).catch(() => {
      // A failed clear is not worth a message: the frame expires on its own within
      // seconds, and the panel is already closed.
    });
  }

  /**
   * Uploads one reduced-resolution frame for the agent to look at (REQ-CAM-01).
   *
   * Deliberately smaller than captureCameraFrame()'s full-resolution capture: this runs
   * every few seconds on a connection already carrying a live voice call (Risk 5), while
   * that one runs once, when a participant asked for the best possible reading of a page.
   *
   * Failures are silent by design. This is a background push nobody asked for; surfacing
   * an error card for it would put a failure on screen for something the participant did
   * not do, and the next push is three seconds away.
   */
  async pushCameraFrame() {
    const video = document.getElementById('camera-preview');
    if (!this.isCameraOn || !video?.videoWidth || this.cameraPushInFlight) return;

    const scale = Math.min(
      1, CAMERA_STREAM_MAX_EDGE / Math.max(video.videoWidth, video.videoHeight)
    );
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise(
      (resolve) => canvas.toBlob(resolve, 'image/jpeg', CAMERA_STREAM_QUALITY)
    );
    if (!blob || !this.isCameraOn) return;

    const form = new FormData();
    form.append('image', blob, 'frame.jpg');

    this.cameraPushInFlight = true;
    try {
      const params = new URLSearchParams({ channel: this.channelName, actor: this.speakerId });
      await requestJson(`/api/session/camera/stream?${params.toString()}`, {
        method: 'POST',
        body: form
      });
    } catch (err) {
      console.debug('📷 Camera frame push skipped:', err.message);
    } finally {
      this.cameraPushInFlight = false;
    }
  }

  /**
   * Opens a camera stream: a named device, or the browser's own choice.
   *
   * `facingMode: 'user'` rather than `'environment'` - this is a laptop client, and the
   * feature is holding a page or a whiteboard up to the built-in webcam, not a phone's
   * rear camera. It is only a preference either way; `deviceId` is what actually decides,
   * which is why the caller supplies one as soon as it knows a good one.
   *
   * @param {string|null} deviceId - Exact device to open, or null to let the browser pick
   * @returns {Promise<MediaStream>}
   */
  openCameraStream(deviceId = null) {
    const video = deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1280 } }
      : { facingMode: 'user', width: { ideal: 1280 } };
    return navigator.mediaDevices.getUserMedia({ video, audio: false });
  }

  /**
   * Names a real visible-light camera to switch to, when the open one is not one.
   *
   * Device labels are blank until a camera permission has been granted, which is why this
   * runs after the first stream opens rather than instead of it.
   *
   * @returns {Promise<Object|null>} A better `MediaDeviceInfo`, or null to keep this one
   */
  async findVisibleLightCamera() {
    const track = this.cameraStream?.getVideoTracks?.()[0];
    if (!track || !DECOY_CAMERA_PATTERN.test(track.label || '')) return null;

    const activeId = track.getSettings?.().deviceId || '';
    const devices = await this.listVideoInputs();
    return devices.find((device) => (
      device.deviceId
      && device.deviceId !== activeId
      && !DECOY_CAMERA_PATTERN.test(device.label || '')
    )) || null;
  }

  /**
   * Lists the video inputs this browser will admit to, or an empty list.
   */
  async listVideoInputs() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.filter((device) => device.kind === 'videoinput');
    } catch (err) {
      console.warn('📷 Could not enumerate cameras:', err.message);
      return [];
    }
  }

  /**
   * Fills the camera picker with the real device labels (D-UIUX-4).
   *
   * Shown only when there is a choice to make: on a one-camera machine this would be a
   * dropdown with a single entry, which says nothing and takes up room. Options are built
   * as DOM nodes rather than markup - a device label is operating-system text, and there
   * is no reason to interpolate text into HTML when a node will do.
   */
  async populateCameraDevices() {
    const select = document.getElementById('camera-device');
    if (!select) return;

    const devices = await this.listVideoInputs();
    const activeId = this.cameraStream?.getVideoTracks?.()[0]?.getSettings?.().deviceId || '';

    select.replaceChildren(...devices.map((device, index) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.textContent = device.label || `Camera ${index + 1}`;
      option.selected = device.deviceId === activeId;
      return option;
    }));
    select.hidden = devices.length < 2;
  }

  /**
   * Switches the preview to a camera the participant picked (D-UIUX-4).
   *
   * @param {string} deviceId - `deviceId` of the chosen video input
   */
  async selectCameraDevice(deviceId) {
    if (!deviceId || !this.isCameraOn) return;

    let stream;
    try {
      stream = await this.openCameraStream(deviceId);
    } catch (err) {
      this.referenceCard.renderStatus({
        tool: 'vision', state: 'failed',
        reason: `Could not switch to that camera: ${err.message}`
      });
      // Put the picker back on the camera actually being shown, so the control never
      // claims a device that failed to open.
      await this.populateCameraDevices();
      return;
    }

    // The previous stream is released only once its replacement is open: stopping first
    // blanks the preview for as long as the new device takes to start, and leaves it
    // blank for good if it never does.
    this.releaseCameraStream();
    this.cameraStream = stream;
    this.cameraDeviceId = deviceId;
    await this.showCameraStream();
  }

  /**
   * Plays the open stream in the preview and reveals the panel.
   */
  async showCameraStream() {
    const panel = document.getElementById('camera-panel');
    const video = document.getElementById('camera-preview');

    if (video) {
      video.srcObject = this.cameraStream;
      await video.play().catch(() => {});
    }
    if (panel) panel.classList.remove('hidden');
  }

  /**
   * Stops every track on the open stream and forgets it.
   */
  releaseCameraStream() {
    if (!this.cameraStream) return;
    this.cameraStream.getTracks().forEach((track) => track.stop());
    this.cameraStream = null;
  }

  /**
   * Releases the camera and hides its panel.
   */
  stopCamera() {
    const panel = document.getElementById('camera-panel');
    const video = document.getElementById('camera-preview');

    this.isCameraOn = false;
    this.stopCameraStreaming();
    this.releaseCameraStream();
    if (video) video.srcObject = null;
    if (panel) panel.classList.add('hidden');
    this.updateCameraUi();
  }

  /**
   * Reflects camera state in the toolbar control.
   */
  updateCameraUi() {
    const btn = document.getElementById('btn-camera');
    const text = document.getElementById('btn-camera-text');
    if (btn) {
      btn.classList.toggle('danger', this.isCameraOn);
      btn.setAttribute('aria-pressed', String(this.isCameraOn));
    }
    if (text) text.textContent = this.isCameraOn ? 'Stop Camera' : 'Camera Assist';
  }

  /**
   * Captures one frame and asks the co-teacher to explain it (REQ-22).
   *
   * Algorithm:
   * 1. Draw the current video frame onto an offscreen canvas at its own resolution.
   * 2. Encode it as JPEG and upload it as a multipart file - which is what the canvas
   *    produces natively, and a third smaller than the base64 alternative.
   * 3. Render the returned material card.
   *
   * The blob is never stored: it exists between the capture and the upload, and the
   * server discards it after describing it (REQ-22).
   */
  async captureCameraFrame() {
    const video = document.getElementById('camera-preview');
    const button = document.getElementById('btn-capture');
    const captureText = document.getElementById('btn-capture-text');

    if (!video?.videoWidth) {
      this.referenceCard.renderStatus({
        tool: 'vision', state: 'failed',
        reason: 'The camera has not produced a frame yet. Give it a moment and try again.'
      });
      return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.85));
    if (!blob) return;

    const question = (document.getElementById('query-input')?.value || '').trim();
    const form = new FormData();
    form.append('image', blob, 'capture.jpg');
    form.append('question', question || 'What is in this image, and what does it say?');

    if (button) button.disabled = true;
    if (captureText) captureText.textContent = 'Looking…';

    try {
      const params = new URLSearchParams({ channel: this.channelName, actor: this.speakerId });
      const body = await requestJson(`/api/tools/vision?${params.toString()}`, {
        method: 'POST',
        body: form
      });
      this.referenceCard.renderReference({ card: body?.card });
    } catch (err) {
      this.referenceCard.renderStatus({
        tool: 'vision', state: 'failed',
        reason: `Could not explain the captured frame: ${err.message}`
      });
    } finally {
      if (button) button.disabled = false;
      if (captureText) captureText.textContent = "Explain what I'm showing";
    }
  }

  /**
   * Starts the session elapsed-time display (REQ-23).
   *
   * Local and approximate by design. It answers "how long have I been practising today",
   * which is a motivation question - not "how long was this session", which is a matter
   * of record and belongs to the stored artifact's own timestamps (REQ-16).
   */
  startSessionTimer() {
    this.sessionStartedAt = Date.now();
    this.renderSessionTimer();
    if (this.sessionTimerHandle) clearInterval(this.sessionTimerHandle);
    this.sessionTimerHandle = setInterval(() => this.renderSessionTimer(), 1000);
  }

  /**
   * Stops the elapsed-time display and freezes it at the final duration.
   */
  stopSessionTimer() {
    if (this.sessionTimerHandle) {
      clearInterval(this.sessionTimerHandle);
      this.sessionTimerHandle = null;
    }
    this.renderSessionTimer();
    this.sessionStartedAt = null;
  }

  /**
   * Renders the elapsed time as mm:ss, or hh:mm:ss once a session runs past an hour.
   */
  renderSessionTimer() {
    const pill = document.getElementById('session-timer');
    if (!pill) return;

    if (!this.sessionStartedAt) {
      pill.textContent = '⏱️ 00:00';
      return;
    }

    const elapsed = Math.max(0, Math.floor((Date.now() - this.sessionStartedAt) / 1000));
    const hours = Math.floor(elapsed / 3600);
    const minutes = Math.floor((elapsed % 3600) / 60);
    const seconds = elapsed % 60;
    const pad = (value) => String(value).padStart(2, '0');

    pill.textContent = hours
      ? `⏱️ ${pad(hours)}:${pad(minutes)}:${pad(seconds)}`
      : `⏱️ ${pad(minutes)}:${pad(seconds)}`;
  }

  /**
   * Starts measuring this participant's own speaking time (REQ-23).
   *
   * Algorithm:
   * 1. Sample the local microphone track's volume level on a fixed interval.
   * 2. Count a sample above the speech threshold as that interval of speech.
   * 3. Report the accumulated milliseconds to the server in batches.
   *
   * The browser is the measurer because it is the only place a real per-participant
   * duration exists today: the backend has no raw per-speaker PCM (TASK-11.9 is
   * deferred), and REQ-23 refuses an invented number. The server still owns the
   * accumulation and the publication, so every client renders one agreed balance rather
   * than its own local guess.
   */
  startSpeechMeasurement() {
    if (!this.localAudioTrack || this.speechSampleHandle) return;

    const SAMPLE_MS = 250;
    // Above room noise but below normal speech. Agora reports 0..1; a quiet room floats
    // around 0.01-0.03, so counting anything audible would report silence as speech.
    const SPEECH_LEVEL = 0.08;
    const REPORT_EVERY_MS = 3000;

    this.unreportedSpeechMs = 0;
    this.speechSampleHandle = setInterval(() => {
      if (!this.localAudioTrack) return;

      let level = 0;
      try {
        level = this.localAudioTrack.getVolumeLevel();
      } catch (err) {
        return;
      }

      if (level >= SPEECH_LEVEL) this.unreportedSpeechMs += SAMPLE_MS;
      if (this.unreportedSpeechMs >= REPORT_EVERY_MS) {
        const measured = this.unreportedSpeechMs;
        this.unreportedSpeechMs = 0;
        this.reportSpeakingTime(measured);
      }
    }, SAMPLE_MS);
  }

  /**
   * Stops sampling and reports whatever was measured but not yet sent.
   */
  stopSpeechMeasurement() {
    if (this.speechSampleHandle) {
      clearInterval(this.speechSampleHandle);
      this.speechSampleHandle = null;
    }
    if (this.unreportedSpeechMs > 0) {
      const measured = this.unreportedSpeechMs;
      this.unreportedSpeechMs = 0;
      this.reportSpeakingTime(measured);
    }
  }

  /**
   * Sends one measured speech batch to the server (REQ-23).
   *
   * Failures are logged and dropped rather than retried: the balance is a live cue, and
   * a queue of stale segments replayed later would describe a conversation that has
   * already moved on.
   */
  async reportSpeakingTime(durationMs) {
    try {
      await requestJson('/api/session/speaking', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          channel: this.channelName,
          actor: this.speakerId,
          speaker_id: this.speakerId,
          duration_ms: Math.round(durationMs)
        })
      });
    } catch (err) {
      console.log('ℹ️ Speaking time not reported:', err.message);
    }
  }

  /**
   * Reflects a translation status event in the toolbar (REQ-17).
   *
   * Two shapes share the event: a leg state (`active`, `degraded`, `unavailable`) and
   * this participant's audio gate. Only the gate for *this* participant may move the
   * toggle - another participant's choice is theirs, and mirroring it here is how two
   * clients end up fighting over one control.
   */
  handleTranslationStatus(payload) {
    if (!payload) return;

    if (payload.state === 'audio_gate') {
      if (payload.participant_id === this.speakerId) {
        this.translatedAudioEnabled = Boolean(payload.translated_audio_enabled);
        this.updateTranslatedAudioUi();
      }
      return;
    }

    if (payload.state === 'unavailable') {
      console.warn(
        `🌐 Translation leg ${payload.leg_id} unavailable: ${payload.reason || 'no reason given'}. `
        + 'The call continues without translated audio.'
      );
    }
    this.translationLegs = { ...(this.translationLegs || {}), [payload.leg_id]: payload.state };
  }

  /**
   * Renders a translated utterance as a subtitle line (REQ-17 / REQ-06).
   *
   * Plain source/target text only. Transliteration and register-aware phrasing stay with
   * the ASR -> TeachingAgent subtitle pipeline, so the two never disagree on screen.
   */
  handleTranslationTranscript(payload) {
    if (!payload?.text || !payload.is_final) return;
    this.subtitles.addSubtitle({
      speaker: `${payload.speaker_id} → ${payload.target_language}`,
      original_text: payload.text,
      transliteration: '',
      translation_en: '',
      translation_ja: '',
      translation_hi: ''
    });
  }

  /**
   * Exports this channel's stored session artifact to the chosen destination (REQ-15).
   *
   * The actor travels with the request because the backend authorizes artifact access
   * against the session's participants (REQ-16).
   *
   * Algorithm:
   * 1. Read the destination from the export selector; Anki has its own endpoint.
   * 2. Markdown is a document, so it is opened in a tab - the browser renders and saves
   *    it better than this UI could.
   * 3. Notion answers JSON, so it is fetched and reported as a card. Opening that JSON
   *    in a tab is what made an unconfigured Notion look like a button that did nothing.
   */
  async exportSession() {
    const select = document.getElementById('export-format');
    const destination = select?.value || 'markdown';

    if (destination === 'anki') {
      await this.exportToAnki();
      return;
    }

    const params = new URLSearchParams({
      channel: this.channelName,
      actor: this.speakerId
    });

    if (destination === 'markdown') {
      params.set('format', 'markdown');
      window.open(`/api/session/artifact/export?${params.toString()}`, '_blank', 'noopener');
      return;
    }

    params.set('format', 'notion');
    params.set('target', destination === 'notion-database' ? 'database' : 'page');

    try {
      const body = await requestJson(`/api/session/artifact/export?${params.toString()}`);
      const notion = body?.notion || {};
      this.referenceCard.renderNotice(
        'Notion Export',
        `This session was exported to your Notion ${notion.target || 'page'}.`,
        true,
        notion.url
      );
    } catch (err) {
      this.referenceCard.renderNotice('Notion Export', err.message, false);
      console.warn('📤 Notion export failed:', err.message);
    }
  }

  /**
   * Deletes one stored note (REQ-14 / REQ-16).
   *
   * The panel is not updated optimistically: the server answers with `note.deleted` over
   * the data stream, and letting that single path drive the UI keeps every viewer of the
   * session in agreement about what was removed.
   */
  async deleteNote(noteId) {
    try {
      const url = `/api/session/notes/${encodeURIComponent(noteId)}`
        + `?channel=${encodeURIComponent(this.channelName)}`
        + `&actor=${encodeURIComponent(this.speakerId)}`;
      const body = await requestJson(url, { method: 'DELETE' });
      // Offline and simulated sessions never receive the RTC event, so reflect it here.
      this.notesPanel.remove(body);
    } catch (err) {
      console.warn('Could not delete the note:', err.message);
    }
  }

  /**
   * Applies a session mode to the client UI (REQ-12).
   *
   * Algorithm:
   * 1. Record the mode that subsequent session-creating calls will send.
   * 2. Retitle the assistance panel for what that mode actually produces.
   * 3. Show the teacher oversight dock only in language learning - there is no
   *    instructor overseeing a work call, and its actions (nudge a quieter learner,
   *    send a quiz) are meaningless there.
   */
  setSessionMode(mode) {
    this.currentMode = mode === 'international_work' ? 'international_work' : 'language_learning';
    const isLearning = this.currentMode === 'language_learning';

    const scaffoldingTitle = document.getElementById('scaffolding-title');
    if (scaffoldingTitle) {
      scaffoldingTitle.textContent = isLearning
        ? '✨ AI Pedagogical Insights'
        : '✨ Terms, Intent & Clarifications';
    }

    const notesTitle = document.getElementById('notes-title');
    if (notesTitle) {
      notesTitle.textContent = isLearning ? '📝 Learning Notes' : '📝 Decisions & Actions';
    }

    this.teacherBar.setVisible(isLearning && this.isAiActive);
    document.body.dataset.sessionMode = this.currentMode;

    // REQ-17: the mode supplies the default until the participant chooses for
    // themselves. Overriding an explicit choice on a mode switch would silently undo it.
    if (!this.translatedAudioChosen) {
      this.translatedAudioEnabled = !isLearning;
      this.updateTranslatedAudioUi();
    }
  }

  /**
   * Turns this participant's translated audio on or off (REQ-17, REQ-06 controls).
   *
   * Algorithm:
   * 1. Record that the choice is now explicit, so a later mode switch leaves it alone.
   * 2. Flip the local mirror and repaint immediately - the control must feel instant.
   * 3. Ask the server to move the real gate; on failure, roll the mirror back rather
   *    than leaving the button claiming a state the router does not have.
   */
  async toggleTranslatedAudio() {
    const next = !this.translatedAudioEnabled;
    this.translatedAudioChosen = true;
    this.translatedAudioEnabled = next;
    this.updateTranslatedAudioUi();

    try {
      const body = await requestJson('/api/translation/audio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          channel: this.channelName,
          participant_id: this.speakerId,
          enabled: next
        })
      });
      this.translatedAudioEnabled = Boolean(body.translated_audio_enabled);
    } catch (err) {
      console.warn('🌐 Could not change the translated-audio gate:', err.message);
      this.translatedAudioEnabled = !next;
    }
    this.updateTranslatedAudioUi();
  }

  /**
   * Starts this channel's Gemini Live Translate legs (REQ-17).
   *
   * Algorithm:
   * 1. Describe the room: this participant in the selected language, plus every remote
   *    participant already publishing audio.
   * 2. Ask the backend to plan and connect the legs for the session's mode.
   * 3. Adopt the gate state the server reports for this participant, since the mode
   *    default lives there rather than here.
   *
   * A failure is logged, not surfaced: translation being unavailable is a degraded call,
   * not a broken one, and the voice conversation is already running by this point.
   */
  async startTranslationLegs() {
    const language = document.getElementById('convoai-language')?.value || 'en';
    const participants = [{ participant_id: this.speakerId, language }];
    this.remoteAudioTracks.forEach((_track, uid) => {
      participants.push({ participant_id: String(uid), language: 'en' });
    });

    try {
      const body = await requestJson('/api/translation/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel: this.channelName, participants })
      });

      this.translationLegs = body.legs || {};
      if (!this.translatedAudioChosen && this.speakerId in (body.translated_audio || {})) {
        this.translatedAudioEnabled = Boolean(body.translated_audio[this.speakerId]);
        this.updateTranslatedAudioUi();
      }
      if (!body.available) {
        console.warn('🌐 Live Translate is unavailable; the session continues without '
          + 'translated audio. Transcripts are unaffected.');
      }
    } catch (err) {
      console.warn('🌐 Could not start the translation legs:', err.message);
    }
  }

  /**
   * Closes this channel's translation legs when the conversation ends (REQ-17).
   */
  async stopTranslationLegs() {
    try {
      await requestJson('/api/translation/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel: this.channelName })
      });
    } catch (err) {
      console.warn('🌐 Could not stop the translation legs:', err.message);
    }
    this.translationLegs = {};
  }

  /**
   * Repaints the translated-audio control from the current gate state.
   */
  updateTranslatedAudioUi() {
    const btn = document.getElementById('btn-translated-audio');
    const text = document.getElementById('btn-translated-audio-text');
    const on = Boolean(this.translatedAudioEnabled);

    if (text) text.textContent = `Translated Audio: ${on ? 'On' : 'Off'}`;
    if (btn) {
      btn.classList.toggle('active', on);
      btn.setAttribute('aria-pressed', String(on));
    }
  }

  /**
   * Locks or releases the mode switcher.
   *
   * The backend is the authority - it rejects a mid-session mode change with a 400 -
   * so this only stops the user from making a request that is guaranteed to fail.
   */
  setModeSwitcherLocked(locked) {
    document.querySelectorAll('#mode-switcher button').forEach((btn) => {
      btn.disabled = locked;
      btn.classList.toggle('locked', locked);
    });
  }

  /**
   * Adapts a stored QuizItem (REQ-13) to the widget's payload shape.
   *
   * The widget predates the artifact contract and speaks in question/options/correct
   * index; the stored quiz speaks in prompt/expected answer. Translating here keeps the
   * widget unaware of the artifact schema.
   */
  quizFromArtifact(quiz) {
    const options = quiz.options || [];
    return {
      active: true,
      question: quiz.prompt,
      options,
      correct_index: Math.max(0, options.indexOf(quiz.expected_answer)),
      explanation: quiz.explanation || ''
    };
  }

  /**
   * Binds DOM event listeners for toolbar actions, mode switching, and simulations.
   */
  bindDomEvents() {
    // Session mode switcher (REQ-12). The mode decides which assistance the backend
    // runs and which note vocabulary it may emit, and it cannot change mid-session -
    // artifacts already generated under the first mode would contradict the second.
    const modeButtons = document.querySelectorAll('#mode-switcher button');
    modeButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        if (this.isAiActive || this.isAiPending) {
          console.warn('Session mode is fixed while a conversation is live. End it first.');
          return;
        }
        modeButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.setSessionMode(btn.getAttribute('data-mode'));
      });
    });
    this.setSessionMode(this.currentMode);

    // Export the stored session (REQ-15 / REQ-19). One control, one destination
    // selector: Markdown opens as a document, Notion and Anki report back as cards.
    const btnExport = document.getElementById('btn-export');
    if (btnExport) {
      btnExport.addEventListener('click', () => this.exportSession());
    }

    // Agent tools (REQ-18-20). Each control is disabled until /api/tools/status says the
    // server holds credentials for it, so a click cannot produce a bare 503.
    const btnResearch = document.getElementById('btn-research');
    if (btnResearch) {
      btnResearch.addEventListener('click', () => this.researchTopic());
    }

    const btnSchedule = document.getElementById('btn-schedule');
    if (btnSchedule) {
      btnSchedule.addEventListener('click', () => this.scheduleFollowUp());
    }

    // Camera assist (REQ-22), gated on the same availability check as the tools above.
    const btnCamera = document.getElementById('btn-camera');
    if (btnCamera) {
      btnCamera.addEventListener('click', () => this.toggleCamera());
    }

    // D-UIUX-4: correcting the browser's camera choice, when it picked a device that
    // opens cleanly and shows nothing.
    const cameraDevice = document.getElementById('camera-device');
    if (cameraDevice) {
      cameraDevice.addEventListener('change', () => {
        this.selectCameraDevice(cameraDevice.value);
      });
    }

    const btnCapture = document.getElementById('btn-capture');
    if (btnCapture) {
      btnCapture.addEventListener('click', () => this.captureCameraFrame());
    }

    // Direct query (REQ-21). A form rather than a click handler, so Enter in the box
    // asks the question - which is how anyone actually uses a text field.
    const queryForm = document.getElementById('query-form');
    if (queryForm) {
      queryForm.addEventListener('submit', (event) => {
        event.preventDefault();
        this.askDirectQuery();
      });
    }

    // Join Channel Button
    const btnJoin = document.getElementById('btn-join');
    if (btnJoin) {
      btnJoin.addEventListener('click', () => this.toggleJoinChannel());
    }

    // Mic Toggle Button
    const btnMic = document.getElementById('btn-mic');
    if (btnMic) {
      btnMic.addEventListener('click', () => this.toggleMicrophone());
    }

    // Gemini Live translated audio (REQ-17), alongside the other direct-AI controls
    const btnTranslatedAudio = document.getElementById('btn-translated-audio');
    if (btnTranslatedAudio) {
      btnTranslatedAudio.addEventListener('click', () => this.toggleTranslatedAudio());
    }
    this.updateTranslatedAudioUi();

    // Convo AI: direct spoken conversation with the AI co-teacher (REQ-09)
    const btnConvoAI = document.getElementById('btn-convoai');
    if (btnConvoAI) {
      btnConvoAI.addEventListener('click', () => this.toggleConvoAI());
    }

    // Release the agent if the learner closes the tab mid-conversation
    window.addEventListener('beforeunload', () => {
      const payload = new Blob(
        [JSON.stringify({ channel: this.channelName })],
        { type: 'application/json' }
      );

      if (this.isAiActive || this.isAiPending) {
        navigator.sendBeacon('/api/convoai/stop', payload);
      } else if (this.isJoined) {
        // Closing the tab is leaving the channel, and a session nobody is in must not
        // be inherited by whoever joins next.
        navigator.sendBeacon('/api/session/stop', payload);
      }
    });

    // Simulation Demo Trigger
    const btnSim = document.getElementById('btn-simulation');
    if (btnSim) {
      btnSim.addEventListener('click', () => this.runSimulationDemo());
    }

    // Clear Feed
    const btnClear = document.getElementById('btn-clear');
    if (btnClear) {
      btnClear.addEventListener('click', () => {
        this.subtitles.clear();
        this.idiomCard.clear();
        this.quizWidget.clear();
      });
    }
  }

  /**
   * Sets up actions triggered from the Teacher Oversight bar.
   */
  setupTeacherBarActions() {
    this.teacherBar.onAction('break_silence', () => {
      this.streamManager.handleStreamMessage(0, {
        event_type: 'topic_prompt',
        payload: {
          topic_title: 'Break Silence: Japanese Omiage & Indian Sweets',
          prompt: 'What special regional souvenir (omiyage or mithai) would you bring to each other?'
        }
      });
      this.streamManager.handleStreamMessage(0, {
        event_type: 'teacher_alert',
        payload: {
          message: 'Teacher injected silence-breaker prompt to re-engage dialogue.',
          severity: 'info'
        }
      });
    });

    this.teacherBar.onAction('nudge_turn', () => {
      this.streamManager.handleStreamMessage(0, {
        event_type: 'subtitles',
        payload: {
          speaker: 'AI Co-Teacher',
          text: 'Aarav-san, what do you think about Kenji-san’s favorite festival?',
          transliteration: 'Aarav-san, Kenji-san no matsuri ni tsuite dou omoimasu ka?',
          translation_en: 'Aarav-san, what do you think about Kenji-san’s favorite festival?',
          translation_ja: 'アーラヴさん、健二さんのお気に入りのお祭りについてどう思いますか？',
          translation_hi: 'आरव, आप केंजी के पसंदीदा त्योहार के बारे में क्या सोचते हैं?'
        }
      });
    });

    this.teacherBar.onAction('quiz', () => {
      this.streamManager.handleStreamMessage(0, {
        event_type: 'quiz',
        payload: {
          question: "Which term is used in Hindi for formal 'You'?",
          options: ['Aap (आप)', 'Tum (तुम)', 'Tu (तू)'],
          correct_index: 0,
          explanation: "'Aap' (आप) is the respectful honorific register in Hindi, equivalent to 'Keigo' in Japanese."
        }
      });
    });

    this.teacherBar.onAction('praise', () => {
      this.streamManager.handleStreamMessage(0, {
        event_type: 'subtitles',
        payload: {
          speaker: 'AI Co-Teacher',
          text: 'Great cross-cultural exchange! Your honorific registers and pronunciation were spot on.',
          transliteration: 'Subarashii kouryuu desu! Pronunciation mo batsugun desu.',
          translation_en: 'Great cross-cultural exchange! Your pronunciation was spot on.',
          translation_ja: '素晴らしい文化交流です！発音も完璧です。',
          translation_hi: 'बहुत बढ़िया बातचीत! आपका उच्चारण बहुत सटीक था।'
        }
      });
    });
  }

  /**
   * Wires Convo AI transcript and agent-state events into the UI.
   *
   * Algorithm:
   * 1. On transcript update, re-render the whole subtitle stream. The toolkit
   *    delivers the complete history each time, so appending would duplicate rows.
   * 2. On agent state change, reflect listening/thinking/speaking in the status pill.
   * 3. On agent error, surface the failing module rather than failing silently.
   */
  setupTranscriptListeners() {
    this.transcriptService.on('transcript', (transcript) => {
      this.renderTranscript(transcript);
    });

    this.transcriptService.on('agentState', (state) => {
      if (this.isAiActive && state) {
        this.updateConvoAIUi('live', state);
      }
    });

    this.transcriptService.on('agentError', (error) => {
      const detail = `${error?.type || 'agent'}: ${error?.message || 'unknown error'}`;
      this.updateConvoAIUi('error', detail);
    });
  }

  /**
   * Re-renders the subtitle stream from a complete Convo AI transcript history.
   *
   * The toolkit's TRANSCRIPT_UPDATED event carries the full conversation every
   * time, so the stream is rebuilt rather than appended to.
   *
   * @param {Array} transcript - Full transcript history from the toolkit
   */
  renderTranscript(transcript) {
    if (!Array.isArray(transcript)) return;

    this.subtitles.clear();
    transcript.forEach((item) => {
      const text = item.text || item.content || '';
      if (!text) return;

      // The agent's own turns are flagged by the toolkit; everything else is the learner.
      const isAgent = item.isAgent ?? item.is_agent ?? (item.role === 'assistant');

      this.subtitles.addSubtitle({
        speaker: isAgent ? 'EchoSphere AI Co-Teacher' : 'You',
        original_text: text
      });
    });
  }

  /**
   * Reports why the microphone is unusable, or null when it is available.
   *
   * Browsers only expose getUserMedia on a secure context. localhost and 127.0.0.1
   * are treated as trustworthy, but a LAN IP over plain HTTP is not - on such an
   * origin `navigator.mediaDevices` is undefined entirely, so mic capture cannot be
   * attempted at all and the AI co-teacher could never hear the learner.
   *
   * @returns {string|null} Human-readable reason, or null if the mic is usable
   */
  getMicUnavailableReason() {
    if (!window.isSecureContext) {
      return (
        `Microphone blocked: ${window.location.origin} is not a secure origin. ` +
        `Use http://localhost:${window.location.port || 80} or an HTTPS URL.`
      );
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return 'Microphone blocked: this browser exposes no getUserMedia API.';
    }
    return null;
  }

  /**
   * Starts the backend session this channel's tools are governed by (REQ-12).
   *
   * Called on every join path - live, mic-blocked, and simulated - because a session is
   * what makes search, direct query, camera assist, notes, quizzes, and export resolve
   * at all; whether RTC audio itself came up is a separate question.
   *
   * Failure is logged and swallowed on purpose: a session-start failure must not strand
   * an otherwise working voice connection. The session-dependent controls already report
   * their own unavailability when used (`tool.status` / a 404 notice).
   *
   * `participants` lists only this client's own speaker id. The peer cards in the left
   * column are static demo text, not captured identities, so naming them here would
   * record participants who were never in the session.
   */
  async startSession() {
    try {
      const body = await requestJson('/api/session/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          channel: this.channelName,
          mode: this.currentMode,
          participants: [this.speakerId]
        })
      });
      const previousId = this.sessionId;
      this.sessionId = body?.session_id || null;
      // A different session is a different set of artifacts: what the poll already
      // rendered belongs to the old one and must not suppress the new one's quizzes.
      if (previousId && previousId !== this.sessionId) this.renderedQuizIds.clear();
      console.log(`🗂️ Session ${this.sessionId} started on '${this.channelName}'.`);
      return true;
    } catch (err) {
      console.warn('Could not start the backend session:', err.message);
      this.sessionId = null;
      return false;
    }
  }

  /**
   * Ends the backend session so a rejoin starts clean rather than inheriting this one.
   */
  async stopSession() {
    try {
      await requestJson('/api/session/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel: this.channelName })
      });
    } catch (err) {
      console.warn('Could not stop the backend session:', err.message);
    }
    this.sessionId = null;
  }

  /**
   * Begins polling for stored notes and quizzes while a session is live (D-UIUX-2).
   *
   * Interim delivery, not a replacement for push: generation is already automatic
   * server-side, and this is what makes it visible until a real event transport exists.
   */
  startArtifactPolling() {
    if (this.artifactPollHandle) return;
    this.refreshSessionArtifacts();
    this.artifactPollHandle = window.setInterval(
      () => this.refreshSessionArtifacts(),
      ARTIFACT_POLL_INTERVAL_MS
    );
  }

  /**
   * Stops the artifact poll and forgets which quizzes were rendered, so the next
   * session's identically-worded quiz is not mistaken for one already on screen.
   */
  stopArtifactPolling() {
    if (this.artifactPollHandle) {
      window.clearInterval(this.artifactPollHandle);
      this.artifactPollHandle = null;
    }
    this.renderedQuizIds.clear();
  }

  /**
   * Renders any note or quiz the session has produced that is not on screen yet.
   *
   * Algorithm:
   * 1. Skip when no session is live - the endpoints are session-governed and would 404.
   * 2. Upsert every note; NotesPanel is keyed by note id, so a re-sent note replaces
   *    itself rather than duplicating, exactly as the data-stream path behaves.
   * 3. Render only quizzes whose id has not been rendered before.
   *
   * A failed poll is logged once and left alone: the session may simply have ended
   * between two ticks, and a retry storm of visible errors helps nobody.
   */
  async refreshSessionArtifacts() {
    if (!this.isJoined && !this.isAiActive) return;

    const params = new URLSearchParams({
      channel: this.channelName,
      actor: this.speakerId
    });

    try {
      const [notesBody, quizBody] = await Promise.all([
        requestJson(`/api/session/notes?${params.toString()}`),
        requestJson(`/api/session/quizzes?${params.toString()}`)
      ]);

      (notesBody?.notes || []).forEach((note) => this.notesPanel.upsert({ note }));

      (quizBody?.quizzes || []).forEach((quiz) => {
        if (!quiz?.id || this.renderedQuizIds.has(quiz.id)) return;
        this.renderedQuizIds.add(quiz.id);
        this.quizWidget.renderQuiz(this.quizFromArtifact(quiz));
      });
    } catch (err) {
      console.log('ℹ️ Session artifacts unavailable:', err.message);
    }
  }

  /**
   * Joins or leaves the Agora RTC Channel.
   *
   * Algorithm:
   * 1. If not joined: Initialize Agora RTC client or simulation bridge, update UI indicators.
   * 2. If joined: Release microphone track, leave channel, update UI.
   */
  async toggleJoinChannel() {
    const btnJoinText = document.getElementById('btn-join-text');
    const btnJoin = document.getElementById('btn-join');
    const btnMic = document.getElementById('btn-mic');
    const connStatus = document.getElementById('connection-status');
    const connDot = document.getElementById('connection-dot');

    if (!this.isJoined) {
      // Declared outside the try because the catch below is a real join path too - a
      // participant whose microphone fails still joins and still needs live events -
      // and it cannot subscribe without these credentials.
      let creds = null;
      try {
        creds = await this.convoai.fetchRtcCredentials(this.localUid);
        const micBlockedReason = this.getMicUnavailableReason();

        if (creds && !creds.simulated && creds.app_id && !micBlockedReason) {
          // Live path: real SD-RTN join with microphone capture.
          this.agoraClient = AgoraRTC.createClient({ mode: 'rtc', codec: 'vp8' });

          // All handlers must be registered BEFORE join() so no early event is missed.
          this.registerRtcHandlers();
          this.streamManager.attachClient(this.agoraClient);

          await this.agoraClient.join(creds.app_id, creds.channel, creds.token, creds.uid);

          this.localAudioTrack = await AgoraRTC.createMicrophoneAudioTrack({
            AEC: true,  // Acoustic echo cancellation (REQ-02)
            ANS: true,  // Automatic noise suppression
            AGC: true   // Automatic gain control
          });
          await this.agoraClient.publish([this.localAudioTrack]);

          this.hasLiveAudio = true;
          this.rtcCredentials = creds;
          if (connStatus) connStatus.textContent = 'Live Audio Connected';
          console.log('✅ Connected to EchoSphere SD-RTN Voice Channel.');

          // Subscribe to Convo AI transcripts over RTM. Non-fatal on failure:
          // the voice conversation still works, only live subtitles are lost.
          await this.transcriptService.connect({
            rtcClient: this.agoraClient,
            appId: creds.app_id,
            channel: creds.channel,
            rtmToken: creds.rtm_token,
            rtmUserId: creds.rtm_user_id
          });
        } else if (micBlockedReason) {
          // Credentials may be fine, but this origin can never capture audio.
          console.error(`🎙️ ${micBlockedReason}`);
          if (connStatus) connStatus.textContent = 'Insecure Origin — No Microphone';
        } else {
          // Simulated path: no Agora credentials configured on the backend.
          if (connStatus) connStatus.textContent = 'Simulated Audio (No Credentials)';
          console.log('ℹ️ Joined in simulated mode — set AGORA_APP_ID to enable live audio.');
        }

        // D-UIUX-1: the backend session is created here, on every join path, because
        // it - not the RTC connection - is what search, direct query, camera assist,
        // notes, quizzes, and export are all governed by.
        await this.startSession();

        await this.subscribeToSessionEvents(creds);

        this.isJoined = true;
        // REQ-23: the session clock starts when the participant joins, and speech
        // measurement starts with it when there is a real microphone track to measure.
        this.startSessionTimer();
        this.startSpeechMeasurement();
        this.startArtifactPolling();
        if (btnJoinText) btnJoinText.textContent = 'Leave Channel';
        if (btnJoin) btnJoin.classList.add('danger');
        if (btnMic) btnMic.disabled = false;
        if (connDot) connDot.style.backgroundColor = 'var(--macos-accent-green)';
      } catch (err) {
        // Microphone denial or RTC failure must not strand the UI in a half-joined state.
        console.warn('RTC Connect Notice (Running in Local Mode):', err);
        await this.releaseRtcResources();
        await this.startSession();
        // Subscribed here too: this path is a real join (the session exists and the
        // button now says "Leave"), and a failed microphone is the case that needs
        // delivered subtitles most, not least.
        await this.subscribeToSessionEvents(creds);
        this.isJoined = true;
        this.startSessionTimer();
        this.startArtifactPolling();
        if (btnJoinText) btnJoinText.textContent = 'Leave Channel';
        if (btnJoin) btnJoin.classList.add('danger');
        if (btnMic) btnMic.disabled = false;
        if (connStatus) connStatus.textContent = 'Simulated Audio (Mic Unavailable)';
      }
    } else {
      // Marked as left first: stopConvoAI() re-creates the session for a participant
      // who is still in the channel, and on this path nobody is.
      this.isJoined = false;

      // Stop any running AI agent before tearing down the channel.
      if (this.isAiActive) {
        await this.stopConvoAI();
      }

      await this.releaseRtcResources();
      this.stopArtifactPolling();
      // A rejoin must start a new session rather than reuse this one's notes, quizzes,
      // and speaking balance - the channel outlives the session, the session does not.
      await this.stopSession();
      // The camera belongs to the session too: leaving the channel must not leave a
      // capture stream running behind a hidden panel (REQ-22).
      this.stopCamera();

      if (btnJoinText) btnJoinText.textContent = 'Join Channel';
      if (btnJoin) btnJoin.classList.remove('danger');
      if (btnMic) btnMic.disabled = true;
      if (connStatus) connStatus.textContent = 'Disconnected';
      if (connDot) connDot.style.backgroundColor = 'var(--macos-text-tertiary)';
    }
  }

  /**
   * Registers Agora RTC event handlers.
   *
   * Must be called BEFORE client.join() so that the Convo AI agent's arrival is never
   * missed: a fast-starting agent can publish audio before a late listener attaches.
   *
   * Algorithm:
   * 1. On 'user-published', subscribe and play remote audio so the learner hears the AI.
   * 2. On 'user-joined', mark the AI as live once its agent participant appears.
   * 3. On 'user-unpublished' / 'user-left', release the cached remote track.
   */
  registerRtcHandlers() {
    if (!this.agoraClient) return;

    this.agoraClient.on('user-published', async (user, mediaType) => {
      await this.agoraClient.subscribe(user, mediaType);
      if (mediaType === 'audio' && user.audioTrack) {
        user.audioTrack.play();
        this.remoteAudioTracks.set(String(user.uid), user.audioTrack);
        console.log(`🔊 Playing remote audio from UID ${user.uid}`);
      }
    });

    this.agoraClient.on('user-joined', (user) => {
      console.log('👤 Remote user joined:', user.uid);
      // The agent is the participant that arrives while a start request is pending.
      if (this.isAiPending) {
        this.markAiLive();
      }
    });

    this.agoraClient.on('user-unpublished', (user) => {
      this.remoteAudioTracks.delete(String(user.uid));
    });

    this.agoraClient.on('user-left', (user) => {
      this.remoteAudioTracks.delete(String(user.uid));
    });
  }

  /**
   * Closes the local microphone track and leaves the RTC channel.
   * Safe to call when nothing is currently allocated.
   */
  /**
   * Subscribes to this channel's live session events (REQ-03, D-UIUX-2).
   *
   * Called from every join path, including the one a failed microphone lands on: the
   * events carry subtitles, idiom cards, quizzes, notes, reference cards, and tool
   * status, none of which depend on this participant having a working mic. For the
   * ambient (non-Convo-AI) path this is the only route those ever take to the browser.
   *
   * Never throws: without it the UI falls back to the Task 1.3 REST poll for notes and
   * quizzes, which is degraded but not broken, and joining must not fail over it.
   *
   * @param {Object|null} creds - The `/api/rtc/token` response for this join
   * @returns {Promise<boolean>} True when live delivery is active
   */
  async subscribeToSessionEvents(creds) {
    return this.sessionEvents.connect({
      appId: creds?.app_id,
      channel: creds?.channel || this.channelName,
      token: creds?.events_rtm_token,
      userId: creds?.events_rtm_user_id
    });
  }

  async releaseRtcResources() {
    await this.transcriptService.disconnect();
    // Released with the rest of the join, so a rejoin subscribes cleanly rather than
    // stacking a second RTM login on the same identity.
    await this.sessionEvents.disconnect();

    // REQ-23: stop sampling before the track it samples is closed, and flush whatever
    // was measured but not yet reported - the last few seconds of speech count too.
    this.stopSpeechMeasurement();
    this.stopSessionTimer();

    if (this.localAudioTrack) {
      this.localAudioTrack.close();
      this.localAudioTrack = null;
    }
    this.remoteAudioTracks.clear();
    this.hasLiveAudio = false;
    this.rtcCredentials = null;

    if (this.agoraClient) {
      try {
        await this.agoraClient.leave();
      } catch (err) {
        console.warn('RTC leave notice:', err);
      }
      this.agoraClient = null;
    }
  }

  /**
   * Toggles local microphone mute state.
   *
   * Also mutes the published RTC track when a live channel is connected, so muting
   * genuinely stops audio reaching the AI co-teacher rather than only changing labels.
   */
  async toggleMicrophone() {
    this.isMuted = !this.isMuted;
    const btnMicText = document.getElementById('btn-mic-text');
    const btnMicIcon = document.getElementById('btn-mic-icon');
    const btnMic = document.getElementById('btn-mic');

    if (this.localAudioTrack) {
      await this.localAudioTrack.setEnabled(!this.isMuted);
    }

    if (this.isMuted) {
      if (btnMicText) btnMicText.textContent = 'Unmute Mic';
      if (btnMicIcon) btnMicIcon.textContent = '🔇';
      if (btnMic) btnMic.classList.add('danger');
    } else {
      if (btnMicText) btnMicText.textContent = 'Mute Mic';
      if (btnMicIcon) btnMicIcon.textContent = '🎙️';
      if (btnMic) btnMic.classList.remove('danger');
    }
  }

  /**
   * Starts or stops a live spoken conversation with the AI co-teacher (REQ-09).
   */
  async toggleConvoAI() {
    if (this.isAiActive || this.isAiPending) {
      await this.stopConvoAI();
    } else {
      await this.startConvoAI();
    }
  }

  /**
   * Starts the Convo AI agent and joins it to the learner's channel.
   *
   * Algorithm:
   * 1. Ensure the learner is in the channel first, so the agent has someone to talk to.
   * 2. Refuse to start when the microphone is not actually published - a live agent
   *    would sit in the channel hearing silence, then expire on idle_timeout.
   * 3. Request the agent from the backend in the selected target language.
   * 4. Enter a pending state: the agent is starting but is not audible yet.
   * 5. Go live on the RTC 'user-joined' event; a simulated agent goes live immediately
   *    because no real participant will ever join the channel.
   */
  async startConvoAI() {
    // Step 1: Joining first guarantees the learner's microphone is already published
    if (!this.isJoined) {
      await this.toggleJoinChannel();
    }

    // Step 2: Fail loudly rather than starting an agent that cannot hear anything.
    // Only enforced when the backend holds real credentials: fully simulated mode is
    // a legitimate offline demo path that never needs a microphone.
    const creds = await this.convoai.fetchRtcCredentials(this.localUid);
    const isLiveBackend = Boolean(creds && !creds.simulated && creds.app_id);

    if (isLiveBackend && !this.hasLiveAudio) {
      const reason = this.getMicUnavailableReason()
        || 'Microphone is not published to the channel.';
      console.error(`🚫 Not starting the AI co-teacher. ${reason}`);
      this.updateConvoAIUi('error', reason);
      return;
    }

    const language = document.getElementById('convoai-language')?.value || 'en';

    // Step 4: Pending state
    this.isAiPending = true;
    this.updateConvoAIUi('starting');

    try {
      this.setModeSwitcherLocked(true);
      const agent = await this.convoai.startAgent(language, this.currentMode, this.speakerId);
      console.log('🤖 Convo AI agent accepted:', agent);

      // REQ-17: the legs are started after the session exists, since the session's mode
      // is what decides the topology. Not awaited into the agent's critical path - a
      // slow Gemini handshake must not delay the voice the learner is waiting for.
      this.startTranslationLegs();

      // Step 5: A simulated agent never produces a 'user-joined' event
      if (agent.simulated) {
        this.markAiLive();
      } else {
        // Guard against an agent that fails to appear on the RTC channel
        this.aiJoinTimeout = setTimeout(() => {
          if (this.isAiPending) {
            const detail = 'AI never joined the channel — check the browser console.';
            console.warn(
              `⏱️ ${detail} The agent was accepted by the backend but no RTC ` +
              `'user-joined' event arrived within 15s.`
            );
            this.stopConvoAI({ reason: detail });
          }
        }, 15000);
      }
    } catch (err) {
      console.error('Failed to start the AI co-teacher:', err);
      this.isAiPending = false;
      this.updateConvoAIUi('error', err.message);
    }
  }

  /**
   * Stops the running Convo AI agent and resets the control.
   *
   * @param {Object} [options]
   * @param {string} [options.reason] - When the stop was automatic rather than a
   *   deliberate click, the surfaced reason so the pill explains itself instead of
   *   silently snapping back to idle.
   */
  async stopConvoAI({ reason = '' } = {}) {
    if (this.aiJoinTimeout) {
      clearTimeout(this.aiJoinTimeout);
      this.aiJoinTimeout = null;
    }

    await this.convoai.stopAgent();
    await this.stopTranslationLegs();
    this.isAiActive = false;
    this.isAiPending = false;
    // The session is over, so its mode is no longer binding: the next one may differ.
    this.setModeSwitcherLocked(false);
    this.teacherBar.setVisible(false);
    this.updateConvoAIUi(reason ? 'error' : 'idle', reason);

    // Stopping the agent ends the session the backend registered for it. Someone still
    // in the channel has not left, so the session the channel's tools resolve against
    // is re-created rather than left absent until the next join (D-UIUX-1).
    if (this.isJoined) await this.startSession();

    console.log('🛑 Convo AI agent stopped.');
  }

  /**
   * Promotes a pending agent to the live listening state.
   * Called from the RTC 'user-joined' handler, or directly in simulated mode.
   */
  markAiLive() {
    if (this.aiJoinTimeout) {
      clearTimeout(this.aiJoinTimeout);
      this.aiJoinTimeout = null;
    }
    this.isAiPending = false;
    this.isAiActive = true;
    this.teacherBar.setVisible(this.currentMode === 'language_learning');
    this.updateConvoAIUi('live');
  }

  /**
   * Reflects Convo AI agent state in the toolbar button and header status pill.
   *
   * @param {string} state - One of 'idle', 'starting', 'live', 'error'
   * @param {string} [detail] - Optional detail message for the error state
   */
  updateConvoAIUi(state, detail = '') {
    const btn = document.getElementById('btn-convoai');
    const btnText = document.getElementById('btn-convoai-text');
    const btnIcon = document.getElementById('btn-convoai-icon');
    const pill = document.getElementById('convoai-status-pill');
    const pillText = document.getElementById('convoai-status-text');
    const langSelect = document.getElementById('convoai-language');

    // Keep the pill short enough to read in the header; the full text goes in the
    // tooltip so a long diagnostic reason is never truncated away entirely.
    const shortDetail = detail.length > 60 ? `${detail.slice(0, 57)}…` : detail;

    // Agent state arrives from RTM as 'listening' | 'thinking' | 'speaking' | 'silent'
    const agentStateLabels = {
      listening: 'AI is Listening',
      thinking: 'AI is Thinking…',
      speaking: 'AI is Speaking',
      silent: 'AI is Idle',
      idle: 'AI is Idle'
    };
    const liveLabel = agentStateLabels[detail] || 'AI is Listening';

    const states = {
      idle:     { text: 'Talk to AI',    icon: '🤖', pill: '',                       show: false },
      starting: { text: 'Connecting…',   icon: '⏳', pill: 'AI Co-Teacher Joining…', show: true },
      live:     { text: 'End AI Chat',   icon: '🎧', pill: liveLabel,                show: true },
      error:    { text: 'Talk to AI',    icon: '⚠️', pill: shortDetail || 'AI Error', show: true }
    };
    const config = states[state] || states.idle;

    if (btnText) btnText.textContent = config.text;
    if (btnIcon) btnIcon.textContent = config.icon;
    if (btn) {
      btn.classList.toggle('listening', state === 'live');
      btn.disabled = state === 'starting';
    }
    // The agent's language is fixed for the lifetime of a session
    if (langSelect) langSelect.disabled = (state === 'live' || state === 'starting');

    if (pill) {
      pill.classList.toggle('hidden', !config.show);
      pill.classList.toggle('error', state === 'error');
      pill.title = state === 'error' ? detail : '';
    }
    if (pillText) pillText.textContent = config.pill;
  }

  /**
   * Executes a realistic multi-turn simulated tandem dialogue demo.
   *
   * Algorithm:
   * 0. Refuse to run while a session is live (REQ-23).
   * 1. Dispatches timed stream packets across Japanese, Hindi, and English.
   * 2. Injects live subtitles with Romaji transliterations.
   * 3. Dispatches cultural idiom cards (e.g. Ichigo Ichie, Jugaad).
   * 4. Replays a scripted speaking balance for the offline preview only.
   * 5. Triggers an AI interactive comprehension quiz.
   *
   * Step 0 is the point of REQ-23's other half. This timeline's `speaking_balance`
   * packets - 70/30, then 52/48 - were for a long time the *only* balance this UI ever
   * showed, and they describe nobody: they are a fixture on a timer. Left runnable during
   * a live conversation, they would overwrite the measured balance with invented numbers,
   * which is worse than showing none. It stays available for previewing the UI with no
   * backend, and says plainly that is what it is.
   */
  runSimulationDemo() {
    if (this.isJoined || this.isAiActive || this.isAiPending) {
      console.warn(
        '▶️ The simulation is an offline UI preview and would overwrite the measured '
        + 'speaking balance with scripted values. Leave the channel to run it.'
      );
      return;
    }

    console.log('▶️ Running Tandem Multi-Lingual Simulation Demo (scripted, offline preview)...');

    const demoTimeline = [
      {
        delay: 500,
        event: 'subtitles',
        payload: {
          speaker: 'Kenji',
          text: 'こんにちは！今日は日本の夏祭りについて話しましょう。一期一会ですね！',
          transliteration: 'Konnichiwa! Kyou wa Nihon no natsumatsuri ni tsuite hanashimashou. Ichigo ichie desu ne!',
          translation_en: 'Hello! Today let’s talk about Japanese summer festivals. Every encounter is once-in-a-lifetime!',
          translation_hi: 'नमस्ते! आज हम जापानी ग्रीष्मकालीन त्योहारों के बारे में बात करेंगे। हर मुलाकात अनमोल है।'
        }
      },
      {
        delay: 1500,
        event: 'idiom_card',
        payload: {
          phrase: '一期一会 (Ichigo Ichie)',
          romaji: 'Ichigo ichie',
          meaning: 'Treasure every encounter; a once-in-a-lifetime meeting.',
          cultural_note: 'Emanates from Sen no Rikyu’s tea ceremony philosophy, reminding speakers to treat every dialogue with utmost sincerity.'
        }
      },
      {
        delay: 2800,
        event: 'speaking_balance',
        payload: {
          speaker_percentages: { 'Kenji': 70, 'Aarav': 30 }
        }
      },
      {
        delay: 3800,
        event: 'subtitles',
        payload: {
          speaker: 'Aarav',
          text: 'नमस्ते केंजी भाई! बिल्कुल, भारत में दिवाली का त्योहार बहुत धूमधाम से मनाया जाता है।',
          transliteration: 'Namaste Kenji bhai! Bilkul, Bharat mein Diwali ka tyohar bahut dhoom-dham se manaya jata hai.',
          translation_en: 'Namaste Kenji! Absolutely, in India the festival of Diwali is celebrated with immense joy and grandeur.',
          translation_ja: '健二さん、こんにちは！まさにその通りですね。インドではディワリ祭がとても盛大に祝われます。'
        }
      },
      {
        delay: 5000,
        event: 'speaking_balance',
        payload: {
          speaker_percentages: { 'Kenji': 52, 'Aarav': 48 }
        }
      },
      {
        delay: 6000,
        event: 'idiom_card',
        payload: {
          phrase: 'धूमधाम (Dhoom-Dham)',
          romaji: 'Dhoom-dhaam',
          meaning: 'Pomp and show; celebration with great zeal and fanfare.',
          cultural_note: 'A rhythmic Hindi compound word expressing high energy community celebrations like Diwali or Holi.'
        }
      },
      {
        delay: 7200,
        event: 'quiz',
        payload: {
          question: "What concept does Kenji's idiom '一期一会' (Ichigo Ichie) emphasize?",
          options: [
            'Cherishing each moment as unique',
            'Saving money for the future',
            'Studying every day without rest'
          ],
          correct_index: 0,
          explanation: "'Ichigo Ichie' literally means 'one time, one meeting'—celebrating the uniqueness of every shared human moment."
        }
      },
      {
        delay: 8500,
        event: 'teacher_alert',
        payload: {
          message: 'Both learners achieved balanced speaking turns (52% / 48%). Idiom understanding confirmed.',
          severity: 'info'
        }
      }
    ];

    demoTimeline.forEach(({ delay, event, payload }) => {
      setTimeout(() => {
        this.streamManager.handleStreamMessage(0, {
          event_type: event,
          payload: payload,
          timestamp_ms: Date.now()
        });
      }, delay);
    });
  }
}

// Instantiate and attach to window
window.addEventListener('DOMContentLoaded', () => {
  window.tandemApp = new TandemApp();
});
