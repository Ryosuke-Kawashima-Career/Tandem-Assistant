import { defineConfig } from 'vite';

/**
 * Summary:
 *   Vite configuration for the EchoSphere web client.
 *
 *   The dev server proxies every backend route to the Flask app on port 8000. Without
 *   this, `npm run dev` serves the UI but answers every `/api/...` fetch with its own
 *   HTML fallback, so the client fails with "Failed to execute 'json' on 'Response'"
 *   before any request ever reaches the backend. Serving the built bundle from Flask
 *   (http://localhost:8000) never had this problem, which is why the gap survived.
 *
 *   `/chat/completions` is proxied too: the Convo AI Engine calls it on the public
 *   tunnel rather than through this dev server, but a developer poking the bridge by
 *   hand from the running page should reach the real backend, not an HTML page.
 */
const BACKEND_ORIGIN = process.env.ECHOSPHERE_BACKEND_ORIGIN || 'http://localhost:8000';

export default defineConfig({
  server: {
    proxy: {
      '/api': { target: BACKEND_ORIGIN, changeOrigin: true },
      '/health': { target: BACKEND_ORIGIN, changeOrigin: true },
      '/chat': { target: BACKEND_ORIGIN, changeOrigin: true }
    }
  }
});
