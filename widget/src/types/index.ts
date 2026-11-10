import type { EmotionContextPayload } from '../emotion/types';
export type { EmotionContextPayload };

/**
 * Custom color overrides for the widget
 */
export interface ChatWidgetColors {
  primary?: string;
  headerBackground?: string;
  headerText?: string;
  buttonBackground?: string;
  buttonText?: string;
  background?: string;
  text?: string;
  assistantMessageBackground?: string;
  userMessageBackground?: string;
  inputBackground?: string;
  inputText?: string;
}

/**
 * Configuration interface for the chat widget
 */
export interface ChatWidgetConfig {
  /** OpenAI API Key (can be set via backend proxy) */
  apiKey?: string;
  /** OpenAI API URL - default: https://api.openai.com/v1 */
  apiUrl?: string;
  /** OpenAI Model - default: gpt-4-turbo */
  model?: string;
  /** Widget position on screen */
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
  /** Widget theme */
  theme?: 'light' | 'dark';
  /** Custom system prompt */
  systemPrompt?: string;
  /** Widget width in pixels */
  width?: number;
  /** Widget height in pixels */
  height?: number;
  /** Whether to show on load */
  showOnLoad?: boolean;
  /** Widget title shown in header */
  title?: string;
  /** Widget subtitle shown below title in header */
  subtitle?: string;
  /** Widget logo URL fallback (used for both if specific URLs not set); empty = no logo */
  logoUrl?: string;
  /** Logo URL specifically for the floating toggle button; falls back to logoUrl */
  logoUrlButton?: string;
  /** Logo URL specifically for the chat header; falls back to logoUrl */
  logoUrlHeader?: string;
  /** Whether to show the logo in the floating toggle button (default: true) */
  showLogoInButton?: boolean;
  /** Whether to show the logo in the chat header (default: true) */
  showLogoInHeader?: boolean;
  /** Logo border: enabled, color, opacity (0-1) */
  logoBorder?: { enabled?: boolean; color?: string; opacity?: number };
  /** Logo glow effect: enabled, color, opacity (0-1) */
  logoGlow?: { enabled?: boolean; color?: string; opacity?: number };
  /** Logo background color behind the logo */
  logoBackground?: { enabled?: boolean; color?: string; opacity?: number };
  /** Privacy overlay text. Empty string skips the overlay entirely. */
  privacyText?: string;
  /** Privacy overlay text in German (overrides privacyText when language is 'de') */
  privacyTextDe?: string;
  /** Privacy overlay text in English (overrides privacyText when language is 'en') */
  privacyTextEn?: string;
  /** Whether the privacy overlay is enabled (default: true) */
  privacyEnabled?: boolean;
  /** Widget language for UI text. If omitted, auto-detected from page lang or browser. */
  language?: 'de' | 'en';
  /** Whether to send an initial greeting message when the widget opens (default: true) */
  initialMessage?: boolean;
  /** Content of the initial greeting message (default: 'Hello! Language: English' for en, 'Hallo!' for de) */
  initialMessageContent?: string;
  /** Initial greeting message in German (overrides initialMessageContent when language is 'de') */
  initialMessageContentDe?: string;
  /** Initial greeting message in English (overrides initialMessageContent when language is 'en') */
  initialMessageContentEn?: string;
  /** Whether the initial message is sent silently (default: true) */
  initialMessageSilent?: boolean;
  /** Custom color overrides */
  colors?: ChatWidgetColors;
  /** Callback when widget is initialized */
  onInit?: () => void;
  /** Callback when widget is opened */
  onOpen?: () => void;
  /** Interval in ms for toggle button bounce animation when closed (default: 15000) */
  bounceInterval?: number;
  /** Text shown in speech bubble during bounce animation (default: 'Hier chatten!') */
  bounceText?: string;
  /** Bounce text in German (overrides bounceText when language is 'de') */
  bounceTextDe?: string;
  /** Bounce text in English (overrides bounceText when language is 'en') */
  bounceTextEn?: string;
  /** Input placeholder text (auto-set from language-specific override or WIDGET_I18N default) */
  inputPlaceholder?: string;
  /** Input placeholder in German (overrides default when language is 'de') */
  inputPlaceholderDe?: string;
  /** Input placeholder in English (overrides default when language is 'en') */
  inputPlaceholderEn?: string;
  /** Callback on error */
  onError?: (error: Error) => void;
}

/**
 * Chat message interface
 */
export interface ToolCallLive {
  tool_name: string;
  args: Record<string, any>;
  ok: boolean;
  error?: string;
  output?: string;
  duration_ms?: number;
}

export interface ToolExecutionEvent {
  tool: string;
  stage: 'started' | 'result';
  code?: string;
  packages?: string[];
  output?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  isStreaming?: boolean;
  reasoning?: string;
  toolCalls?: ToolCallLive[];
  terminalEvents?: ToolExecutionEvent[];
  emotion?: any;
}

/**
 * OpenAI API request interface
 */
export interface OpenAIRequest {
  model: string;
  messages: Array<{
    role: 'user' | 'assistant' | 'system';
    content: string;
  }>;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  // Optional conversation id to associate requests with a session or client-provided id
  conversation_id?: string;
  // Optional emotion context payload
  emotion_context?: EmotionContextPayload;
}

/**
 * OpenAI API response interface
 */
export interface OpenAIResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

/**
 * OpenAI Streaming response chunk
 */
export interface OpenAIStreamChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string;
      reasoning_content?: string;
      tool_execution?: ToolExecutionEvent;
      tool_calls?: ToolCallLive[];
      emotion?: any;
    };
    finish_reason: string | null;
  }>;
}
