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
      const res = await fetch(
        `/api/rtc/token?channel=${encodeURIComponent(this.channelName)}&uid=${uid}`
      );
      if (!res.ok) return null;
      return await res.json();
    } catch (err) {
      console.warn('Could not fetch RTC credentials:', err);
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
   * @returns {Promise<Object>} Agent descriptor including agent_id and status
   */
  async startAgent(language = 'en') {
    const res = await fetch('/api/convoai/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channel: this.channelName, language })
    });

    const data = await res.json();
    if (!res.ok || !data.success) {
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
      const res = await fetch('/api/convoai/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel: this.channelName, agent_id: agentId })
      });
      const data = await res.json();
      return Boolean(data.success);
    } catch (err) {
      console.warn('Failed to stop the AI co-teacher:', err);
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
      const res = await fetch(
        `/api/convoai/status?channel=${encodeURIComponent(this.channelName)}`
      );
      const data = await res.json();
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
