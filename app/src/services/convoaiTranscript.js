/**
 * Summary:
 *   convoaiTranscript.js subscribes to the Agora Conversational AI transcript and
 *   agent-state stream for the EchoSphere Tandem Co-Teacher client.
 *
 *   The Convo AI Engine publishes what it heard the learner say and what the agent
 *   replied over RTM, on a channel named after the RTC channel. This is the only
 *   real-time path for those transcripts: the EchoSphere backend has no server-side
 *   Agora RTC SDK, so it cannot push subtitles to the browser itself.
 *
 *   Requires `advanced_features.enable_rtm: true` and `parameters.data_channel: "rtm"`
 *   in the agent join payload (set in src/rtc/convoai_client.py).
 *
 * Key Class:
 *   - ConvoAITranscriptService: RTM login, toolkit lifecycle, and event fan-out.
 */

import AgoraRTM from 'agora-rtm';
import { AgoraVoiceAI, AgoraVoiceAIEvents } from 'agora-agent-client-toolkit';

export class ConvoAITranscriptService {
  /**
   * Initialize the transcript service.
   *
   * Algorithm:
   * 1. Initialize client handles as null (nothing is connected yet).
   * 2. Initialize the listener registry for transcript and agent-state events.
   */
  constructor() {
    this.rtmClient = null;
    this.voiceAI = null;
    this.isSubscribed = false;
    this.listeners = { transcript: [], agentState: [], agentError: [] };
  }

  /**
   * Registers a listener. Supported: 'transcript', 'agentState', 'agentError'.
   *
   * @param {string} event - Event name
   * @param {Function} callback - Handler invoked with the event payload
   */
  on(event, callback) {
    if (!this.listeners[event]) this.listeners[event] = [];
    this.listeners[event].push(callback);
  }

  /**
   * Invokes every listener registered for an event, isolating handler failures.
   */
  _emit(event, ...args) {
    (this.listeners[event] || []).forEach((cb) => {
      try {
        cb(...args);
      } catch (err) {
        console.error(`Error in ConvoAI ${event} listener:`, err);
      }
    });
  }

  /**
   * Connects RTM and starts receiving agent messages for a channel.
   *
   * Algorithm:
   * 1. Tear down any previous instance - AgoraVoiceAI is a singleton, so a second
   *    init() would silently replace the first.
   * 2. Log in to RTM using the identity the RTM token was minted for.
   * 3. Initialize the toolkit against the already-joined RTC client.
   * 4. Register every handler BEFORE subscribing, or messages already in flight
   *    are missed.
   * 5. Subscribe to the channel.
   *
   * @param {Object} options
   * @param {Object} options.rtcClient - The already-joined Agora RTC client
   * @param {string} options.appId - Agora App ID
   * @param {string} options.channel - RTC channel name (RTM uses the same name)
   * @param {string} options.rtmToken - RTM token minted for rtmUserId
   * @param {string} options.rtmUserId - RTM identity; must match the token subject
   * @returns {Promise<boolean>} True when subscribed successfully
   */
  async connect({ rtcClient, appId, channel, rtmToken, rtmUserId }) {
    // Step 1: Never stack instances of the singleton
    await this.disconnect();

    try {
      // Step 2: RTM login. The identity must match the token subject exactly.
      this.rtmClient = new AgoraRTM.RTM(appId, rtmUserId);
      await this.rtmClient.login({ token: rtmToken });

      // Step 3: Toolkit init is async - awaiting is required
      this.voiceAI = await AgoraVoiceAI.init({
        rtcEngine: rtcClient,
        rtmConfig: { rtmEngine: this.rtmClient }
      });

      // Step 4: Handlers first, subscription second
      this.voiceAI.on(AgoraVoiceAIEvents.TRANSCRIPT_UPDATED, (transcript) => {
        // Delivers the FULL history every time - replace, never append
        this._emit('transcript', transcript);
      });

      this.voiceAI.on(AgoraVoiceAIEvents.AGENT_STATE_CHANGED, (agentUserId, event) => {
        this._emit('agentState', event?.state, agentUserId);
      });

      this.voiceAI.on(AgoraVoiceAIEvents.AGENT_ERROR, (agentUserId, error) => {
        console.error('[Convo AI agent error]', error);
        this._emit('agentError', error);
      });

      // Step 5: RTM channel name mirrors the RTC channel name
      this.voiceAI.subscribeMessage(channel);
      this.isSubscribed = true;
      console.log(`✅ Subscribed to Convo AI transcripts on '${channel}' as '${rtmUserId}'.`);
      return true;
    } catch (err) {
      console.error('Failed to subscribe to Convo AI transcripts:', err);
      await this.disconnect();
      return false;
    }
  }

  /**
   * Stops receiving messages and releases the RTM session.
   * Safe to call when nothing is connected.
   */
  async disconnect() {
    if (this.voiceAI) {
      try {
        this.voiceAI.unsubscribe();
        this.voiceAI.destroy();
      } catch (err) {
        console.warn('Convo AI toolkit teardown notice:', err);
      }
      this.voiceAI = null;
    }

    if (this.rtmClient) {
      try {
        await this.rtmClient.logout();
      } catch (err) {
        console.warn('RTM logout notice:', err);
      }
      this.rtmClient = null;
    }

    this.isSubscribed = false;
  }
}
