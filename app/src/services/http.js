/**
 * Summary:
 *   http.js wraps `fetch` for every EchoSphere backend call so a non-JSON response
 *   reports what actually happened instead of throwing
 *   "Failed to execute 'json' on 'Response'".
 *
 *   That message is what the browser produces when `res.json()` meets an HTML error
 *   page, a proxy miss, or an empty body - and it names none of them. The backend
 *   answers HTML in several ordinary situations (a Flask 500, a dev server with no API
 *   proxy, a tunnel's own error page), and in each one the useful information is the
 *   status code and the first line of the body, both of which the raw error discards.
 *
 * Key Functions:
 *   - requestJson: fetch + parse, raising an error that names status, URL, and body.
 *   - HttpError: carries the status and body snippet for callers that want to branch.
 */

// Enough of the body to identify what answered - a doctype, a proxy banner, a
// traceback's first line - without pasting a whole error page into the UI.
const BODY_SNIPPET_LIMIT = 200;

export class HttpError extends Error {
  /**
   * @param {string} message - Human-readable description, already actionable
   * @param {Object} details - `status`, `url`, and `body` snippet when available
   */
  constructor(message, { status = 0, url = '', body = '' } = {}) {
    super(message);
    this.name = 'HttpError';
    this.status = status;
    this.url = url;
    this.body = body;
  }
}

/**
 * Describes a non-JSON body well enough to act on.
 *
 * The HTML case is called out by name because it has one overwhelmingly common cause in
 * this project: the request reached something that is not the EchoSphere backend - the
 * Vite dev server without its `/api` proxy, or a tunnel serving its own error page.
 *
 * @param {string} text - Raw response body
 * @returns {string} A diagnosis fragment to append to the error message
 */
function describeNonJsonBody(text) {
  const trimmed = (text || '').trim();

  if (!trimmed) {
    return 'the response body was empty';
  }

  if (trimmed.startsWith('<')) {
    return 'the response was HTML, not JSON - the request reached something other than '
      + 'the EchoSphere backend (a dev server without the /api proxy, or an error page)';
  }

  return `the response was not JSON: ${JSON.stringify(trimmed.slice(0, BODY_SNIPPET_LIMIT))}`;
}

/**
 * Performs a request and parses a JSON response, failing with a legible error.
 *
 * Algorithm:
 * 1. Issue the request; a transport failure is reported as such rather than as a parse
 *    error, because "the server is not running" and "the server replied oddly" call for
 *    different fixes.
 * 2. Read the body as text once - a Response body can only be consumed once, so reading
 *    text first is what makes the snippet available in the failure path at all.
 * 3. Parse it as JSON, reporting the status and body when that fails.
 * 4. Reject an unsuccessful status, preferring the backend's own `error` message, which
 *    is written to be shown to a person.
 *
 * @param {string} url - Request URL
 * @param {Object} [options] - Standard fetch options
 * @returns {Promise<Object>} Parsed JSON body
 * @throws {HttpError} On transport failure, a non-JSON body, or an error status
 */
export async function requestJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (err) {
    throw new HttpError(
      `Could not reach ${url}: ${err.message}. Is the EchoSphere backend running?`,
      { url }
    );
  }

  const text = await response.text();

  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new HttpError(
      `${url} answered ${response.status} ${response.statusText} but `
      + `${describeNonJsonBody(text)}.`,
      { status: response.status, url, body: text.slice(0, BODY_SNIPPET_LIMIT) }
    );
  }

  if (!response.ok) {
    throw new HttpError(
      data.error || `${url} failed with ${response.status} ${response.statusText}.`,
      { status: response.status, url, body: text.slice(0, BODY_SNIPPET_LIMIT) }
    );
  }

  return data;
}
