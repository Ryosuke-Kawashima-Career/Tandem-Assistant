/**
 * Summary:
 *   app.js is the main client entry point for the EchoSphere Tandem Co-Teacher Web Client.
 *   It coordinates:
 *     1. Agora RTC Web SDK connection (microphone capture, SD-RTN voice channel join/leave).
 *     2. Agora RTC Data Stream listener via AgoraStreamManager.
 *     3. UI Components: Subtitles, IdiomCard, TopicWidget, TeacherBar, QuizWidget.
 *     4. Role switcher (Student View vs Teacher Oversight Dashboard).
 *     5. Interactive multi-turn demo simulation covering Japanese, Hindi, and English exchange.
 *
 * Key Controllers:
 *   - TandemApp: Manages overall state, event dispatching, and device lifecycles.
 */

import { AgoraStreamManager } from './services/agoraStream.js';
import { Subtitles } from './components/Subtitles.js';
import { IdiomCard } from './components/IdiomCard.js';
import { TopicWidget } from './components/TopicWidget.js';
import { TeacherBar } from './components/TeacherBar.js';
import { QuizWidget } from './components/QuizWidget.js';

class TandemApp {
  /**
   * Initialize Tandem Application.
   * 
   * Algorithm:
   * 1. Initialize client state (channel, role, joined status, mic status).
   * 2. Instantiate UI component controllers.
   * 3. Attach AgoraStreamManager and wire stream event listeners.
   * 4. Bind DOM UI event listeners.
   */
  constructor() {
    // Step 1: Client State
    this.appId = 'mock_agora_app_id';
    this.channelName = 'tokyo-mumbai-101';
    this.currentRole = 'student';
    this.isJoined = false;
    this.isMuted = false;
    this.localAudioTrack = null;
    this.agoraClient = null;

    // Step 2: Initialize UI Components
    this.subtitles = new Subtitles('#subtitles-container');
    this.idiomCard = new IdiomCard('#scaffolding-container');
    this.quizWidget = new QuizWidget('#scaffolding-container');
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

    console.log('🚀 EchoSphere Tandem Client initialized with macOS Light Theme.');
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
  }

  /**
   * Binds DOM event listeners for toolbar actions, role switching, and simulations.
   */
  bindDomEvents() {
    // Role switcher
    const roleButtons = document.querySelectorAll('#role-switcher button');
    roleButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        roleButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.currentRole = btn.getAttribute('data-role');
        this.teacherBar.setVisible(this.currentRole === 'teacher');
      });
    });

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
        if (window.AgoraRTC) {
          this.agoraClient = window.AgoraRTC.createClient({ mode: 'rtc', codec: 'vp8' });
          this.streamManager.attachClient(this.agoraClient);
          // Optional real join if credentials provided
        }
        this.isJoined = true;
        if (btnJoinText) btnJoinText.textContent = 'Leave Channel';
        if (btnJoin) btnJoin.classList.add('danger');
        if (btnMic) btnMic.disabled = false;
        if (connStatus) connStatus.textContent = 'Live Audio Connected';
        if (connDot) connDot.style.backgroundColor = 'var(--macos-accent-green)';
        console.log('✅ Connected to EchoSphere SD-RTN Voice Channel.');
      } catch (err) {
        console.warn('RTC Connect Notice (Running in Local Mode):', err);
        this.isJoined = true;
        if (btnJoinText) btnJoinText.textContent = 'Leave Channel';
        if (btnJoin) btnJoin.classList.add('danger');
        if (btnMic) btnMic.disabled = false;
      }
    } else {
      this.isJoined = false;
      if (this.localAudioTrack) {
        this.localAudioTrack.close();
        this.localAudioTrack = null;
      }
      if (btnJoinText) btnJoinText.textContent = 'Join Channel';
      if (btnJoin) btnJoin.classList.remove('danger');
      if (btnMic) btnMic.disabled = true;
      if (connStatus) connStatus.textContent = 'Disconnected';
      if (connDot) connDot.style.backgroundColor = 'var(--macos-text-tertiary)';
    }
  }

  /**
   * Toggles local microphone mute state.
   */
  toggleMicrophone() {
    this.isMuted = !this.isMuted;
    const btnMicText = document.getElementById('btn-mic-text');
    const btnMicIcon = document.getElementById('btn-mic-icon');
    const btnMic = document.getElementById('btn-mic');

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
