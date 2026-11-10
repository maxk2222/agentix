/**
 * Agentix CyberChatbot Widget - Main Entry Point
 */

import { ChatWidget } from './components/widget';
import { sanitizeServerDefaults } from './utils/defaults';
import type { ChatWidgetConfig } from './types/index';

// Export for use
export { ChatWidget };
export type { ChatWidgetConfig, ChatMessage } from './types/index';

/**
 * Global window interface
 */
declare global {
  interface Window {
    AgentixChatWidget: typeof ChatWidget;
    initAgentixChat: (config?: ChatWidgetConfig) => Promise<ChatWidget>;
  }
}

function deepMerge(target: Record<string, any>, source: Record<string, any>): Record<string, any> {
  const result = { ...target };
  for (const key in source) {
    if (source[key] !== undefined) {
      if (
        typeof source[key] === 'object' &&
        source[key] !== null &&
        !Array.isArray(source[key]) &&
        typeof result[key] === 'object' &&
        result[key] !== null
      ) {
        result[key] = deepMerge(result[key], source[key]);
      } else {
        result[key] = source[key];
      }
    }
  }
  return result;
}

function getWidgetBaseUrl(): string {
  const scripts = document.querySelectorAll('script');
  for (const script of scripts) {
    if (script.src.includes('widget.umd.js') || script.src.includes('widget.esm.js')) {
      const url = new URL(script.src);
      // Keep any path prefix so subpath/reverse-proxy deployments resolve correctly.
      const prefix = url.pathname.replace(/\/widget\/widget\.(umd|esm)\.js$/, '');
      return url.origin + prefix;
    }
  }
  return window.location.origin;
}

async function fetchWidgetDefaults(): Promise<ChatWidgetConfig> {
  try {
    const baseUrl = getWidgetBaseUrl();
    const res = await fetch(`${baseUrl}/widget-defaults`, {
      method: 'GET',
      credentials: 'omit',
    });
    if (!res.ok) return {};
    const data = await res.json();
    if (typeof data !== 'object' || data === null || Array.isArray(data)) return {};
    return sanitizeServerDefaults(data as Record<string, unknown>) as ChatWidgetConfig;
  } catch {
    return {};
  }
}

// Make available globally
window.AgentixChatWidget = ChatWidget;

/**
 * Initialize chat widget globally.
 * Fetches server-side defaults first, then merges with client-provided config.
 * Client config always overrides defaults.
 */
window.initAgentixChat = async (config?: ChatWidgetConfig): Promise<ChatWidget> => {
  const defaults = await fetchWidgetDefaults();
  const merged = deepMerge(defaults, config || {});
  return new ChatWidget(merged);
};

// Auto-initialize if config is in HTML data attribute
document.addEventListener('DOMContentLoaded', async () => {
  const script = document.currentScript as HTMLScriptElement | null;
  if (script?.dataset.autoInit === 'true') {
    const configStr = script?.dataset.config;
    const config = configStr ? JSON.parse(configStr) : undefined;
    const defaults = await fetchWidgetDefaults();
    const merged = deepMerge(defaults, config || {});
    new ChatWidget(merged);
  }
});
