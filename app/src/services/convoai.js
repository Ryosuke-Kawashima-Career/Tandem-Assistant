/**
 * Summary:
 *   convoai.js provides the frontend client for EchoSphere's Conversational AI
 *   (Convo AI) backend surface.
 *   It wraps the REST endpoints that start, stop, and query the AI co-teacher agent
 *   that joins the Agora RTC channel and converses with a learner by voice, and
 *   fetches the RTC credentials the browser needs to join that same channel.
 *
 *   This is the client half of REQ-09 (Direct AI Audio Conversation).
 *
 * Key Class:
 *   - ConvoAIService: REST bridge to /api/convoai/* and /api/rtc/token.
 */

import { requestJson } from './http.js';

export class ConvoAIService {
  /**
   * Initialize the Convo AI service client.
   *
   * Algorithm:
   * 1. Store the channel this service manages agents for.
   * 2. Initialize the active agent descriptor as null.
   *
   * @param {string} channelName - RTC channel the AI agent should join
   */
  constructor(channelName) {
    this.channelName = channelName;
    this.activeAgent = null;
  }

  /**
   * Fetches Agora RTC credentials (App ID + channel token) from the backend.
   *
   * @param {number} uid - Local user UID to mint the token for
   * @returns {Promise<Object|null>} Credentials object, or null when unavailable
   */
  async fetchRtcCredentials(uid = 0) {
    try {
      return await requestJson(
        `/api/rtc/token?channel=${encodeURIComponent(this.channelName)}&uid=${uid}`
      );
    } catch (err) {
      console.warn('Could not fetch RTC credentials:', err.message);
      return null;
    }
  }

  /**
   * Starts a Convo AI agent that joins the channel and speaks with the learner.
   *
   * A resolved promise means the backend accepted the request, not that the agent
   * is already audible: the caller must still wait for the RTC 'user-joined' event
   * before treating the AI as live.
   *
   * @param {string} language - Target language: 'hi', 'ja', or 'en'
   * @param {string} speakerId - The learner's identity. Recorded as the session's
   *   participant, which is what artifact access is later authorized against (REQ-16).
   * @param {string} mode - Session mode: 'language_learning' or 'international_work'.
   *   Required by the backend (REQ-12); a missing or unknown value is a 400, never a
   *   silent default, because every artifact the session produces is mode-shaped.
   * @returns {Promise<Object>} Agent descriptor including agent_id and status
   */
  async startAgent(language = 'en', mode = 'language_learning', speakerId = 'Learner') {
    // requestJson, not fetch + res.json(): a non-JSON answer here is the difference
    // between "the backend refused and said why" and an unreadable parse error, and
    // this is the call a learner triggers by pressing a button.
    const data = await requestJson('/api/convoai/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channel: this.channelName, language, mode, speaker_id: speakerId })
    });

    if (!data.success) {
      throw new Error(data.error || 'Failed to start the AI co-teacher.');
    }

    this.activeAgent = data.agent;
    return data.agent;
  }

  /**
   * Stops the running Convo AI agent and clears local state.
   *
   * @returns {Promise<boolean>} True when the agent was stopped
   */
  async stopAgent() {
    const agentId = this.activeAgent ? this.activeAgent.agent_id : null;

    try {
      const data = await requestJson('/api/convoai/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel: this.channelName, agent_id: agentId })
      });
      return Boolean(data.success);
    } catch (err) {
      // A 404 means the agent was already gone, which is a successful outcome for the
      // caller: either way no agent is attached to this channel any more.
      console.warn('Failed to stop the AI co-teacher:', err.message);
      return false;
    } finally {
      this.activeAgent = null;
    }
  }

  /**
   * Queries the lifecycle status of the agent attached to this channel.
   *
   * @returns {Promise<Object|null>} Agent status descriptor, or null when idle
   */
  async getStatus() {
    try {
      const data = await requestJson(
        `/api/convoai/status?channel=${encodeURIComponent(this.channelName)}`
      );
      return data.agent || null;
    } catch (err) {
      return null;
    }
  }

  /**
   * Reports whether an agent is currently tracked as active by this client.
   *
   * @returns {boolean}
   */
  isActive() {
    return this.activeAgent !== null;
  }
}
