/**
 * Summary:
 *   sessionEvents.js receives the events the EchoSphere backend generates during a
 *   session - subtitles, idiom cards, quizzes, notes, reference cards, tool status -
 *   over an Agora Signaling (RTM) message channel (REQ-03, D-UIUX-2).
 *
 *   Those events were always meant to arrive over Agora's RTC Data Stream, but the
 *   backend has no in-channel native SDK and so could never publish one; everything it
 *   produced reached local Python callbacks and stopped there. The server now publishes
 *   to the RTM channel of the same name via Agora's Signaling REST API, and this
 *   service subscribes to it.
 *
 *   Each message is handed to the existing `AgoraStreamManager.handleStreamMessage`
 *   rather than parsed here, because the server sends the identical
 *   `{event_type, payload, timestamp_ms}` envelope the RTC path used. Every widget
 *   already wired to that manager therefore works with no change: the transport moved,
 *   the contract did not.
 *
 * Key Class:
 *   - SessionEventService: RTM login, channel subscription, and event hand-off.
 */

import AgoraRTM from 'agora-rtm';

export class SessionEventService {
  /**
   * Initialize the service.
   *
   * @param {Object} streamManager - AgoraStreamManager that already fans events out
   *                                 to the UI widgets
   */
  constructor(streamManager) {
    this.streamManager = streamManager;
    this.rtmClient = null;
    this.channel = null;
    this.isSubscribed = false;
  }

  /**
   * Logs in to RTM and starts receiving this channel's session events.
   *
   * Algorithm:
   * 1. Tear down any previous subscription, so a rejoin cannot stack two clients.
   * 2. Log in with the events-specific identity the token was minted for.
   * 3. Register the message handler BEFORE subscribing, or events already in flight
   *    on a busy channel are dropped.
   * 4. Subscribe to the channel, whose name mirrors the RTC channel.
   *
   * Returns false rather than throwing: losing live delivery must degrade the UI to
   * the REST poll that already backs notes and quizzes, never block joining a call.
   *
   * @param {Object} options
   * @param {string} options.appId - Agora App ID
   * @param {string} options.channel - Channel name (shared with RTC by convention)
   * @param {string} options.token - RTM token minted for userId
   * @param {string} options.userId - RTM identity; must match the token subject
   * @returns {Promise<boolean>} True when subscribed
   */
  async connect({ appId, channel, token, userId }) {
    await this.disconnect();

    if (!appId || !channel || !token || !userId) {
      console.warn('📡 Session events: missing RTM credentials; live delivery is off.');
      return false;
    }

    try {
      this.rtmClient = new AgoraRTM.RTM(appId, userId);

      // Step 3: handler first, subscription second
      this.rtmClient.addEventListener('message', (event) => {
        this.handleMessage(event);
      });

      await this.rtmClient.login({ token });
      await this.rtmClient.subscribe(channel);

      this.channel = channel;
      this.isSubscribed = true;
      console.log(`📡 Live session events subscribed on '${channel}' as '${userId}'.`);
      return true;
    } catch (err) {
      console.warn('📡 Session events unavailable, falling back to polling:', err?.message || err);
      await this.disconnect();
      return false;
    }
  }

  /**
   * Forwards one RTM message into the existing stream-manager fan-out.
   *
   * Only messages this server published are forwarded. RTM channels are shared
   * namespaces, and a participant's own chat or a future feature's traffic on the same
   * channel must not be parsed as a session event.
   *
   * @param {Object} event - RTM message event
   */
  handleMessage(event) {
    if (event?.publisher && event.publisher !== 'echosphere-server') return;
    if (!this.streamManager) return;

    this.streamManager.handleStreamMessage(event?.publisher || 'server', event?.message);
  }

  /**
   * Stops receiving events and releases the RTM session. Safe when nothing is connected.
   */
  async disconnect() {
    if (this.rtmClient) {
      // Logged out in its own try: a failed unsubscribe must not skip the logout, or
      // the RTM identity stays live and the next login is refused as already in use.
      try {
        if (this.channel) await this.rtmClient.unsubscribe(this.channel);
      } catch (err) {
        console.warn('📡 Session events unsubscribe notice:', err?.message || err);
      }

      try {
        await this.rtmClient.logout();
      } catch (err) {
        console.warn('📡 Session events logout notice:', err?.message || err);
      }
      this.rtmClient = null;
    }

    this.channel = null;
    this.isSubscribed = false;
  }
}
