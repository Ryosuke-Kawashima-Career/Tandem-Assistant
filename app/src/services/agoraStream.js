/**
 * Summary:
 *   agoraStream.js provides the frontend listener and event dispatcher for
 *   Agora RTC Data Streams in the EchoSphere Web Client.
 *   It deserializes real-time JSON payloads transmitted over the Agora SD-RTN data channel
 *   (such as live tri-lingual subtitles, Romaji transliterations, idiom cards,
 *   quiz popups, and teacher alerts) and dispatches them to reactive UI widgets.
 *
 * Key Class:
 *   - AgoraStreamManager: Manages stream listeners, binary UTF-8 decoding, and event callbacks.
 */

export class AgoraStreamManager {
  /**
   * Initialize the Agora Data Stream Manager.
   * 
   * Algorithm:
   * 1. Initialize subscriber callback registry for each supported event type.
   * 2. Initialize active client reference and message decoder.
   */
  constructor() {
    // Step 1: Callback registry
    this.listeners = {
      subtitles: [],
      idiom_card: [],
      topic_prompt: [],
      quiz: [],
      teacher_alert: [],
      speaking_balance: [],
      all: []
    };

    // Step 2: Text decoder for UTF-8 bytes
    this.textDecoder = new TextDecoder('utf-8');
    this.agoraClient = null;
  }

  /**
   * Attaches the stream manager to an active Agora RTC client instance.
   * 
   * Algorithm:
   * 1. Store client reference.
   * 2. Register listener for the native Agora Web SDK 'stream-message' event.
   * 
   * @param {Object} client - Agora Web SDK RTC Client instance
   */
  attachClient(client) {
    this.agoraClient = client;

    if (this.agoraClient && typeof this.agoraClient.on === 'function') {
      this.agoraClient.on('stream-message', (uid, streamData) => {
        this.handleStreamMessage(uid, streamData);
      });
      console.log('✅ AgoraStreamManager attached to Agora RTC Client.');
    }
  }

  /**
   * Ingests and decodes raw binary stream messages from Agora SD-RTN.
   * 
   * Algorithm:
   * 1. Decode Uint8Array / ArrayBuffer bytes into a UTF-8 string.
   * 2. Parse JSON packet extracting event_type and payload.
   * 3. Dispatch the parsed payload to all matching registered callback handlers.
   * 
   * @param {number|string} uid - Remote speaker UID
   * @param {Uint8Array|string} streamData - Raw incoming packet bytes or JSON string
   */
  handleStreamMessage(uid, streamData) {
    try {
      let jsonString = '';
      if (typeof streamData === 'string') {
        jsonString = streamData;
      } else if (streamData instanceof Uint8Array || streamData instanceof ArrayBuffer) {
        jsonString = this.textDecoder.decode(streamData);
      } else {
        jsonString = JSON.stringify(streamData);
      }

      const packet = JSON.parse(jsonString);
      const eventType = packet.event_type || 'unknown';
      const payload = packet.payload || {};
      const timestamp = packet.timestamp_ms || Date.now();

      // Dispatch to specific event handlers
      if (this.listeners[eventType]) {
        this.listeners[eventType].forEach(callback => {
          try {
            callback(payload, { uid, timestamp, eventType });
          } catch (err) {
            console.error(`Error in listener for ${eventType}:`, err);
          }
        });
      }

      // Dispatch to wildcard 'all' handlers
      this.listeners.all.forEach(callback => {
        try {
          callback(eventType, payload, { uid, timestamp });
        } catch (err) {
          console.error('Error in wildcard listener:', err);
        }
      });
    } catch (error) {
      console.error('Failed to parse incoming RTC Data Stream message:', error);
    }
  }

  /**
   * Registers a listener callback for a specific event type.
   * Supported types: 'subtitles', 'idiom_card', 'topic_prompt', 'quiz', 'teacher_alert', 'speaking_balance', 'all'.
   * 
   * @param {string} eventType - The event name to listen for
   * @param {Function} callback - Callback function receiving (payload, metadata)
   * @returns {Function} Unsubscribe function to remove the listener
   */
  on(eventType, callback) {
    if (!this.listeners[eventType]) {
      this.listeners[eventType] = [];
    }
    this.listeners[eventType].push(callback);

    // Return unregister cleanup function
    return () => {
      this.listeners[eventType] = this.listeners[eventType].filter(cb => cb !== callback);
    };
  }

  /**
   * Removes all registered listeners and detaches client.
   */
  destroy() {
    Object.keys(this.listeners).forEach(key => {
      this.listeners[key] = [];
    });
    this.agoraClient = null;
  }
}
