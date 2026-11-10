/**
 * Helpers for applying server-side widget defaults.
 *
 * Precedence: explicitly provided embed config > server defaults > built-in defaults.
 */

/** Keys where a user-provided object is merged over the server object (sub-key level). */
const OBJECT_MERGE_KEYS = new Set(['colors', 'logoBorder', 'logoGlow', 'logoBackground']);

/** Keys that are never applied from server defaults (secrets / non-serializable callbacks). */
const IGNORED_DEFAULT_KEYS = new Set(['apiKey', 'apiSecret', 'onOpen', 'onInit', 'onError']);

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Compute the subset of server defaults that should be applied on top of a
 * widget instance whose constructor received `userConfig`.
 *
 * - Keys explicitly present in `userConfig` win (user intent beats defaults).
 * - For OBJECT_MERGE_KEYS both objects are merged server-under-user so a
 *   partially specified `colors` map still inherits the remaining defaults.
 * - Secrets and function-valued config are never taken from the server payload.
 */
export function mergeServerDefaults(
  userConfig: Record<string, unknown>,
  serverDefaults: Record<string, unknown>,
): Record<string, unknown> {
  const merged: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(serverDefaults)) {
    if (value === undefined || IGNORED_DEFAULT_KEYS.has(key)) {
      continue;
    }
    if (key === 'language' && value === 'auto') {
      continue;
    }

    const userValue = userConfig[key];
    if (userValue === undefined) {
      merged[key] = value;
      continue;
    }

    if (OBJECT_MERGE_KEYS.has(key) && isPlainObject(value) && isPlainObject(userValue)) {
      merged[key] = { ...value, ...userValue };
    }
    // Primitive key explicitly set by the user -> user wins, key stays untouched.
  }

  return merged;
}

/**
 * Strip secrets/callbacks/placeholders from a server defaults payload.
 * Used when defaults are merged wholesale under user config (initAgentixChat).
 */
export function sanitizeServerDefaults(
  serverDefaults: Record<string, unknown>,
): Record<string, unknown> {
  const sanitized: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(serverDefaults)) {
    if (value === undefined || IGNORED_DEFAULT_KEYS.has(key)) {
      continue;
    }
    if (key === 'language' && value === 'auto') {
      continue;
    }
    sanitized[key] = value;
  }
  return sanitized;
}
