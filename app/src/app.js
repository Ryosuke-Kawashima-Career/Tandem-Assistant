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
import { ConvoAITranscriptService } from './services/convoaiTranscript.js';
import { Subtitles } from './components/Subtitles.js';
import { IdiomCard } from './components/IdiomCard.js';
import { TopicWidget } from './components/TopicWidget.js';
import { TeacherBar } from './components/TeacherBar.js';
import { QuizWidget } from './components/QuizWidget.js';
import { NotesPanel } from './components/NotesPanel.js';

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

    // Convo AI state (REQ-09)
    this.convoai = new ConvoAIService(this.channelName);
    this.isAiActive = false;
    this.isAiPending = false;
    this.remoteAudioTracks = new Map();

    // True only when a real RTC join published a live microphone track. A spoken AI
    // conversation is impossible without this, so it gates startConvoAI().
    this.hasLiveAudio = false;

    // Live transcripts and agent state, delivered over RTM by the Convo AI Engine
    this.transcriptService = new ConvoAITranscriptService();
    this.rtcCredentials = null;
    this.setupTranscriptListeners();

    // Step 2: Initialize UI Components
    this.subtitles = new Subtitles('#subtitles-container');
    this.idiomCard = new IdiomCard('#scaffolding-container');
    this.quizWidget = new QuizWidget('#scaffolding-container');
    this.notesPanel = new NotesPanel('#notes-container', {
      emptyHint: '#notes-empty-hint',
      countBadge: '#notes-count'
    });
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

    // Step 4: UI Event Handlers
    this.bindDomEvents();
    this.setupTeacherBarActions();
    this.checkBackendConnection();

    console.log('🚀 EchoSphere Tandem Client initialized with macOS Light Theme.');
  }

  /**
   * Pings the EchoSphere backend server to verify live status.
   */
  async checkBackendConnection() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        const connStatus = document.getElementById('connection-status');
        const connDot = document.getElementById('connection-dot');
        if (connStatus) connStatus.textContent = `Backend Live (v${data.version})`;
        if (connDot) connDot.style.backgroundColor = 'var(--macos-accent-green)';
        console.log('✅ Connected to EchoSphere Backend API:', data);
      }
    } catch (err) {
      console.log('ℹ️ Running in standalone offline mode:', err);
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

    // Convo AI: direct spoken conversation with the AI co-teacher (REQ-09)
    const btnConvoAI = document.getElementById('btn-convoai');
    if (btnConvoAI) {
      btnConvoAI.addEventListener('click', () => this.toggleConvoAI());
    }

    // Release the agent if the learner closes the tab mid-conversation
    window.addEventListener('beforeunload', () => {
      if (this.isAiActive || this.isAiPending) {
        navigator.sendBeacon(
          '/api/convoai/stop',
          new Blob([JSON.stringify({ channel: this.channelName })], { type: 'application/json' })
        );
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
      try {
        const creds = await this.convoai.fetchRtcCredentials(this.localUid);
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

        this.isJoined = true;
        if (btnJoinText) btnJoinText.textContent = 'Leave Channel';
        if (btnJoin) btnJoin.classList.add('danger');
        if (btnMic) btnMic.disabled = false;
        if (connDot) connDot.style.backgroundColor = 'var(--macos-accent-green)';
      } catch (err) {
        // Microphone denial or RTC failure must not strand the UI in a half-joined state.
        console.warn('RTC Connect Notice (Running in Local Mode):', err);
        await this.releaseRtcResources();
        this.isJoined = true;
        if (btnJoinText) btnJoinText.textContent = 'Leave Channel';
        if (btnJoin) btnJoin.classList.add('danger');
        if (btnMic) btnMic.disabled = false;
        if (connStatus) connStatus.textContent = 'Simulated Audio (Mic Unavailable)';
      }
    } else {
      // Stop any running AI agent before tearing down the channel.
      if (this.isAiActive) {
        await this.stopConvoAI();
      }

      await this.releaseRtcResources();
      this.isJoined = false;

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
  async releaseRtcResources() {
    await this.transcriptService.disconnect();

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
      const agent = await this.convoai.startAgent(language, this.currentMode);
      console.log('🤖 Convo AI agent accepted:', agent);

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
    this.isAiActive = false;
    this.isAiPending = false;
    // The session is over, so its mode is no longer binding: the next one may differ.
    this.setModeSwitcherLocked(false);
    this.teacherBar.setVisible(false);
    this.updateConvoAIUi(reason ? 'error' : 'idle', reason);
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
   * 1. Dispatches timed stream packets across Japanese, Hindi, and English.
   * 2. Injects live subtitles with Romaji transliterations.
   * 3. Dispatches cultural idiom cards (e.g. Ichigo Ichie, Jugaad).
   * 4. Updates real-time speaking balance metrics.
   * 5. Triggers an AI interactive comprehension quiz.
   */
  runSimulationDemo() {
    console.log('▶️ Running Tandem Multi-Lingual Simulation Demo...');

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
