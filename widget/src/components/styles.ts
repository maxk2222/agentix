/**
 * Agentix CyberChatbot Widget - Styles
 * Modern, responsive chat widget with dark/light mode support
 */

export const WIDGET_STYLES = `
  * {
    box-sizing: border-box;
  }

  .agentix-chat-widget-container {
    position: fixed;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    z-index: 999999;
    font-size: 14px;
    /* Default CSS variables */
    --chat-primary: #0d3d66;
    --chat-header-bg: #0d3d66;
    --chat-header-text: #ffffff;
    --chat-bg: #ffffff;
    --chat-text: #333333;
    --chat-user-msg-bg: #0d3d66;
    --chat-user-msg-text: #ffffff;
    --chat-assistant-msg-bg: #f5f5f5;
    --chat-assistant-msg-text: #333333;
    --chat-input-bg: #ffffff;
    --chat-input-text: #333333;
    --chat-input-border: #d0d0d0;
    --chat-send-btn-bg: #0d3d66;
    --chat-send-btn-text: #ffffff;
    --chat-btn-shadow: rgba(0, 0, 0, 0.15);
  }

  .agentix-chat-widget-container.bottom-right {
    bottom: 20px;
    right: 20px;
  }

  .agentix-chat-widget-container.bottom-left {
    bottom: 20px;
    left: 20px;
  }

  .agentix-chat-widget-container.top-right {
    top: 20px;
    right: 20px;
  }

  .agentix-chat-widget-container.top-left {
    top: 20px;
    left: 20px;
  }

  /* Floating Button */
  .agentix-chat-toggle-btn {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    position: relative;
    z-index: 1000;
    background: var(--chat-primary);
    color: white;
  }

  .agentix-chat-toggle-btn img {
    width: 32px;
    height: 32px;
    object-fit: contain;
    border-radius: 4px;
  }

  .agentix-chat-toggle-btn.bouncing {
    animation: agentix-chat-bounce 1.5s ease infinite;
  }

  @keyframes agentix-chat-bounce {
    0%, 100% { transform: translateY(0); }
    25% { transform: translateY(-8px); }
    50% { transform: translateY(0); }
    75% { transform: translateY(-4px); }
  }

  .agentix-chat-bubble {
    position: absolute;
    right: 70px;
    bottom: 10px;
    background: var(--chat-primary, #0d3d66);
    color: #fff;
    padding: 8px 14px;
    border-radius: 10px;
    font-size: 14px;
    white-space: nowrap;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    opacity: 0;
    transform: translateX(10px);
    transition: opacity 0.3s ease, transform 0.3s ease;
    pointer-events: none;
    z-index: 1001;
  }

  .agentix-chat-bubble::after {
    content: '';
    position: absolute;
    right: -6px;
    top: 50%;
    transform: translateY(-50%);
    border-width: 6px 0 6px 6px;
    border-style: solid;
    border-color: transparent transparent transparent var(--chat-primary, #0d3d66);
  }

  .agentix-chat-bubble.visible {
    opacity: 1;
    transform: translateX(0);
  }

  .agentix-chat-toggle-btn:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
  }

  .agentix-chat-toggle-btn:active {
    transform: scale(0.95);
  }

  /* Badge */
  .agentix-chat-badge {
    position: absolute;
    top: -5px;
    right: -5px;
    background: #ef4444;
    color: white;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: bold;
    border: 2px solid white;
  }

  /* Chat Window */
  .agentix-chat-window {
    position: fixed;
    display: flex;
    flex-direction: column;
    border-radius: 12px;
    box-shadow: 0 5px 40px rgba(0, 0, 0, 0.16);
    opacity: 0;
    transform: scale(0.95) translateY(10px);
    pointer-events: none;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 999;
    background: var(--chat-bg);
    color: var(--chat-text);
  }

  .agentix-chat-window.open {
    opacity: 1;
    transform: scale(1) translateY(0);
    pointer-events: auto;
  }

  .agentix-chat-window.resizing {
    transition: none !important;
    will-change: width, height;
  }

  /* Ensure input stays at bottom */
  .agentix-chat-window .agentix-chat-input {
    flex-shrink: 0;
  }

  .agentix-chat-window.bottom-right {
    bottom: 90px;
    right: 20px;
  }

  .agentix-chat-window.bottom-left {
    bottom: 90px;
    left: 20px;
  }

  .agentix-chat-window.top-right {
    top: 90px;
    right: 20px;
  }

  .agentix-chat-window.top-left {
    top: 90px;
    left: 20px;
  }

  /* Light theme */
  .agentix-chat-widget-container.light .agentix-chat-toggle-btn {
    background: var(--chat-primary);
    color: white;
  }

  .agentix-chat-widget-container.light .agentix-chat-header {
    background: var(--chat-header-bg);
    color: var(--chat-header-text);
    border-bottom: none;
  }

  .agentix-chat-widget-container.light .agentix-chat-message.user {
    background: var(--chat-user-msg-bg);
    color: var(--chat-user-msg-text);
  }

  .agentix-chat-widget-container.light .agentix-chat-message.assistant {
    background: var(--chat-assistant-msg-bg);
    color: var(--chat-assistant-msg-text);
  }

  .agentix-chat-widget-container.light .agentix-chat-input-bar {
    background: transparent;
    border-top: none;
  }

  .agentix-chat-widget-container.light .agentix-chat-input-inner {
    background: var(--chat-input-bg);
    border: 1px solid #e0e0e0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
  }

  .agentix-chat-widget-container.light .agentix-chat-input-field {
    background: transparent;
    color: var(--chat-input-text);
    border: none;
  }

  .agentix-chat-widget-container.light .agentix-chat-input-field:focus {
    outline: none;
  }

  /* Dark theme */
  .agentix-chat-widget-container.dark .agentix-chat-toggle-btn {
    background: var(--chat-primary);
    color: white;
  }

  .agentix-chat-widget-container.dark .agentix-chat-header {
    background: var(--chat-header-bg);
    color: var(--chat-header-text);
    border-bottom: 1px solid var(--chat-primary);
  }

  .agentix-chat-widget-container.dark .agentix-chat-message.user {
    background: var(--chat-user-msg-bg);
    color: var(--chat-user-msg-text);
  }

  .agentix-chat-widget-container.dark .agentix-chat-message.assistant {
    background: var(--chat-assistant-msg-bg);
    color: var(--chat-assistant-msg-text);
  }

  .agentix-chat-widget-container.dark .agentix-chat-input-bar {
    background: transparent;
    border-top: none;
  }

  .agentix-chat-widget-container.dark .agentix-chat-input-inner {
    background: #1e293b;
    border: 1px solid #334155;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
  }

  .agentix-chat-widget-container.dark .agentix-chat-input-field {
    background: transparent;
    color: var(--chat-input-text);
    border: none;
  }

  .agentix-chat-widget-container.dark .agentix-chat-input-field:focus {
    outline: none;
  }

  /* Header */
  .agentix-chat-header {
    padding: 8px 16px;
    border-radius: 12px 12px 0 0;
    display: flex;
    align-items: center;
    flex-shrink: 0;
    gap: 12px;
  }

  .agentix-chat-header-logo {
    height: 36px;
    width: auto;
    object-fit: contain;
    flex-shrink: 0;
    display: block;
  }

  .agentix-chat-logo-wrapper {
    display: inline-flex;
    border-radius: 6px;
    overflow: hidden;
  }

  .agentix-chat-header-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }

  .agentix-chat-header-title {
    font-size: 14px;
    font-weight: 600;
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .agentix-chat-header-subtitle {
    font-size: 11px;
    opacity: 0.8;
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .agentix-chat-close-btn {
    background: none;
    border: none;
    color: currentColor;
    font-size: 20px;
    cursor: pointer;
    padding: 4px 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0.7;
    transition: opacity 0.2s;
    margin-left: auto;
  }

  .agentix-chat-close-btn:hover {
    opacity: 1;
  }

  /* Messages Container */
  .agentix-chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .agentix-chat-messages::-webkit-scrollbar {
    width: 6px;
  }

  .agentix-chat-widget-container.dark .agentix-chat-messages::-webkit-scrollbar-track {
    background: #0f3460;
  }

  .agentix-chat-widget-container.light .agentix-chat-messages::-webkit-scrollbar-track {
    background: #f5f5f5;
  }

  .agentix-chat-widget-container.dark .agentix-chat-messages::-webkit-scrollbar-thumb {
    background: #16213e;
    border-radius: 3px;
  }

  .agentix-chat-widget-container.light .agentix-chat-messages::-webkit-scrollbar-thumb {
    background: #d0d0d0;
    border-radius: 3px;
  }

  /* Message (modern speech bubbles) */
  .agentix-chat-message {
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    gap: 8px;
    max-width: 85%;
    padding: 10px 14px;
    border-radius: 18px;
    word-wrap: break-word;
    white-space: normal;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    animation: widget-message-appear 0.25s ease-out;
    position: relative;
    transition: box-shadow 0.2s ease;
  }

  @keyframes widget-message-appear {
    from {
      opacity: 0;
      transform: translateY(8px) scale(0.97);
    }
    to {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
  }

  .agentix-chat-message.user {
    align-self: flex-end;
    border-radius: 18px 18px 4px 18px;
  }

  .agentix-chat-message.assistant {
    align-self: flex-start;
    border-radius: 18px 18px 18px 4px;
    padding-bottom: 6px;
  }

  .agentix-chat-message-content {
    font-size: 14px;
    line-height: 1.6;
    display: block;
    flex: 1;
    min-width: 0;
  }

  .agentix-chat-message-content p {
    margin: 0 0 3px 0;
  }

  .agentix-chat-message-content p:last-child {
    margin-bottom: 0;
  }

  .agentix-chat-message-content code {
    background: rgba(0, 0, 0, 0.1);
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Monaco', 'Courier New', monospace;
    font-size: 12px;
  }

  .agentix-chat-message-content pre {
    background: rgba(0, 0, 0, 0.2);
    padding: 10px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 6px 0;
  }

  .agentix-chat-message-content pre code {
    background: none;
    padding: 0;
    color: inherit;
  }

  .agentix-chat-message-content ul,
  .agentix-chat-message-content ol {
    margin: 6px 0;
    padding-left: 20px;
  }

  .agentix-chat-message-content li {
    margin: 2px 0;
  }

  /* Streaming indicator */
  .agentix-chat-message.streaming {
    opacity: 0.8;
  }

  .agentix-chat-message-content.streaming::after {
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
    animation: blink 1.4s infinite;
    margin-left: 6px;
  }

  @keyframes blink {
    0%, 60%, 100% {
      opacity: 0.3;
    }
    30% {
      opacity: 1;
    }
  }

  /* Input Area (floating bar like dashboard) */
  .agentix-chat-input-bar {
    padding: 0 12px 12px;
    border-top: none;
    background: transparent;
    flex-shrink: 0;
  }

  .agentix-chat-input-inner {
    display: flex;
    gap: 8px;
    align-items: flex-end;
    background: var(--chat-input-bg);
    border: 1px solid var(--chat-input-border);
    border-radius: 20px;
    padding: 6px 10px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    max-width: 100%;
    margin: 0 auto;
    transition: max-width 0.4s ease, padding 0.4s ease;
  }

  .agentix-chat-input-inner.phone-mode {
    width: fit-content;
    max-width: 100%;
    justify-content: center;
    gap: 0;
    align-items: center;
    flex-wrap: nowrap;
    overflow: hidden;
  }

  .agentix-chat-input-field,
  .agentix-chat-send-btn,
  .agentix-chat-stop-btn,
  .agentix-chat-mic-btn,
  .agentix-chat-emotion-btn {
    transition: all 0.4s ease;
  }

  .agentix-chat-input-field.phone-mode-hidden {
    display: none !important;
    opacity: 0;
    flex: 0;
    width: 0;
    height: 0;
    min-height: 0;
    padding: 0;
    margin: 0;
    border: none;
    overflow: hidden;
    pointer-events: none;
  }

  .agentix-chat-send-btn.phone-mode-hidden,
  .agentix-chat-stop-btn.phone-mode-hidden,
  .agentix-chat-mic-btn.phone-mode-hidden,
  .agentix-chat-emotion-btn.phone-mode-hidden {
    opacity: 0;
    width: 0;
    height: 0;
    padding: 0;
    margin: 0;
    transform: scale(0);
    overflow: hidden;
    pointer-events: none;
  }

  .agentix-chat-input-field {
    flex: 1;
    padding: 6px 10px;
    border: none;
    border-radius: 16px;
    font-size: 14px;
    font-family: inherit;
    background: transparent;
    color: var(--chat-input-text);
    resize: none;
    max-height: 100px;
    min-height: 32px;
    overflow-y: hidden;
    line-height: 1.5;
  }

  .agentix-chat-input-field:focus {
    outline: none;
  }

  .agentix-chat-input-field::placeholder {
    opacity: 0.5;
  }

  .agentix-chat-input-field:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .agentix-chat-send-btn {
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--chat-send-btn-bg);
    color: var(--chat-send-btn-text);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    flex-shrink: 0;
    transition: filter 0.15s ease, transform 0.1s ease;
  }

  .agentix-chat-send-btn:hover:not(:disabled) {
    filter: brightness(1.1);
  }

  .agentix-chat-send-btn:active:not(:disabled) {
    transform: scale(0.96);
  }

  .agentix-chat-send-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* Stop button */
  .agentix-chat-stop-btn {
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    background: transparent;
    border: none;
    cursor: pointer;
    flex-shrink: 0;
  }

  .agentix-chat-stop-pulse {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #dc2626;
    animation: widget-stop-pulse 1.6s ease-in-out infinite;
  }

  .agentix-chat-stop-btn:hover .agentix-chat-stop-pulse {
    animation-play-state: paused;
    opacity: 0.9;
  }

  .agentix-chat-stop-icon {
    position: absolute;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
  }

  @keyframes widget-stop-pulse {
    0%, 100% { transform: scale(1); opacity: 0.55; }
    50% { transform: scale(1.2); opacity: 0.85; }
  }

  /* Empty state */
  .agentix-chat-empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    opacity: 0.5;
    text-align: center;
    padding: 20px;
  }

  .agentix-chat-empty-state-icon {
    font-size: 40px;
    margin-bottom: 12px;
  }

  .agentix-chat-empty-state-text {
    font-size: 14px;
    line-height: 1.45;
  }

  /* Error state */
  .agentix-chat-error {
    background: #fee2e2;
    border: 1px solid #fca5a5;
    color: #dc2626;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 0;
  }

  .agentix-chat-widget-container.dark .agentix-chat-error {
    background: #7f1d1d;
    border-color: #dc2626;
    color: #fca5a5;
  }

  .agentix-chat-error-message {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #fffbeb;
    border: 1px solid #fbbf24;
    color: #92400e;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 16px;
    font-size: 13px;
    font-weight: 500;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
  }

  .agentix-chat-widget-container.dark .agentix-chat-error-message {
    background: #451a03;
    border-color: #f59e0b;
    color: #fcd34d;
  }

  .agentix-chat-error-icon {
    flex-shrink: 0;
    font-size: 16px;
    line-height: 1;
  }

  /* Responsive */
  @media (max-width: 600px) {
    .agentix-chat-window {
      position: fixed !important;
      top: 0 !important;
      left: 0 !important;
      bottom: 0 !important;
      right: 0 !important;
      width: 100vw !important;
      height: 100vh !important;
      border-radius: 0 !important;
      max-width: none !important;
      max-height: none !important;
    }

    .agentix-chat-window.open {
      transform: none !important;
    }

    .agentix-chat-toggle-btn {
      display: none;
    }

    .agentix-chat-message {
      max-width: 90%;
    }
  }

  /* Loading spinner */
  .agentix-chat-spinner {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
    animation: spin 1.4s infinite;
  }

  @keyframes spin {
    0% { opacity: 0.3; }
    50% { opacity: 1; }
    100% { opacity: 0.3; }
  }

  /* Thinking animation - wave dots */
  .agentix-chat-thinking {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
    color: var(--chat-text-secondary, #555);
    padding: 4px 0;
    transition: opacity 0.3s ease-out;
  }

  .agentix-chat-thinking.fade-out {
    opacity: 0;
    pointer-events: none;
  }

  .agentix-chat-thinking-label {
    font-size: 13px;
    opacity: 1;
  }

  .agentix-chat-thinking-dots {
    display: inline-flex;
    align-items: center;
    gap: 3px;
  }

  .agentix-chat-thinking-dots span {
    display: inline-block;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: currentColor;
    opacity: 0.8;
    animation: thinking-wave 1.6s infinite ease-in-out both;
  }

  .agentix-chat-thinking-dots span:nth-child(1) {
    animation-delay: 0s;
  }

  .agentix-chat-thinking-dots span:nth-child(2) {
    animation-delay: 0.2s;
  }

  .agentix-chat-thinking-dots span:nth-child(3) {
    animation-delay: 0.4s;
  }

  @keyframes thinking-wave {
    0%, 80%, 100% {
      transform: scale(0.6);
      opacity: 0.5;
    }
    40% {
      transform: scale(1.2);
      opacity: 1;
    }
  }

  /* Privacy Modal - inside chat window */
  .agentix-chat-privacy-modal {
    display: none;
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(13, 61, 102, 0.95);
    /* dynamic overlay color via inline style if needed */
    z-index: 10;
    align-items: center;
    justify-content: center;
    padding: 20px;
    border-radius: 12px;
  }

  .agentix-chat-privacy-content {
    background: white;
    border-radius: 12px;
    padding: 24px;
    max-width: 360px;
    width: 100%;
    box-shadow: 0 10px 40px rgba(13, 61, 102, 0.2);
  }

  .agentix-chat-privacy-title {
    margin: 0 0 16px 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--chat-primary);
  }

  .agentix-chat-privacy-text {
    margin: 0 0 24px 0;
    font-size: 14px;
    line-height: 1.5;
    color: #475569;
  }

  .agentix-chat-privacy-buttons {
    display: flex;
    gap: 12px;
    justify-content: flex-end;
  }

  .agentix-chat-privacy-accept,
  .agentix-chat-privacy-decline {
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    border: none;
    transition: all 0.2s;
  }

  .agentix-chat-privacy-accept {
    background: var(--chat-primary);
    color: white;
  }

  .agentix-chat-privacy-accept:hover {
    opacity: 0.9;
  }

  .agentix-chat-privacy-decline {
    background: #e2e8f0;
    color: #475569;
  }

  .agentix-chat-privacy-decline:hover {
    background: #cbd5e1;
  }

  /* HTML render toggle inside code blocks */
  .agentix-chat-code-toggle {
    display: inline-flex;
    gap: 0;
    background: rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 6px;
  }

  .agentix-chat-widget-container.dark .agentix-chat-code-toggle {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(255, 255, 255, 0.1);
  }

  .agentix-chat-code-toggle button {
    padding: 4px 10px;
    border: none;
    background: transparent;
    color: var(--chat-text-secondary, #888);
    font-size: 12px;
    cursor: pointer;
    transition: background 0.15s ease, color 0.15s ease;
  }

  .agentix-chat-code-toggle button.active {
    background: var(--chat-primary);
    color: white;
  }

  .agentix-chat-code-toggle button:not(.active):hover {
    background: rgba(0, 0, 0, 0.08);
  }

  .agentix-chat-widget-container.dark .agentix-chat-code-toggle button:not(.active):hover {
    background: rgba(255, 255, 255, 0.08);
  }

  .agentix-chat-html-iframe {
    width: 100%;
    min-height: 200px;
    border: 1px solid rgba(0, 0, 0, 0.15);
    border-radius: 8px;
    background: white;
  }

  .agentix-chat-widget-container.dark .agentix-chat-html-iframe {
    border-color: rgba(255, 255, 255, 0.15);
  }

  /* Resize Handle */
  .agentix-chat-resize-handle {
    position: absolute;
    top: 0;
    left: 0;
    width: 30px;
    height: 30px;
    cursor: nwse-resize;
    background: rgba(13, 61, 102, 0.1);
    border-radius: 12px 0 0 0;
    z-index: 100;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .agentix-chat-resize-handle::before {
    content: '⇲';
    font-size: 16px;
    color: var(--chat-primary);
    opacity: 0.7;
    transform: rotate(45deg);
  }

  /* Audio buttons: mic, phone */
  .agentix-chat-mic-btn,
  .agentix-chat-phone-btn {
    background: transparent;
    border: none;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    color: var(--chat-text-secondary, #888);
    transition: background 0.2s ease, color 0.2s ease, transform 0.15s ease;
    position: relative;
    flex-shrink: 0;
  }

  .agentix-chat-mic-btn:hover,
  .agentix-chat-phone-btn:hover {
    background: var(--chat-primary);
    color: white;
    transform: scale(1.08);
  }

  .agentix-chat-mic-btn:active,
  .agentix-chat-phone-btn:active {
    transform: scale(0.95);
  }

  .agentix-chat-mic-btn.recording {
    background: #dc2626;
    color: white;
    animation: widget-mic-recording-pulse 1.5s ease-in-out infinite;
  }

  .agentix-chat-mic-btn.awaiting {
    animation: widget-mic-awaiting 1.5s ease-in-out infinite;
  }

  @keyframes widget-mic-awaiting {
    0%, 100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
    50% { box-shadow: 0 0 0 6px rgba(220, 38, 38, 0); }
  }

  @keyframes widget-mic-recording-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
    50% { box-shadow: 0 0 0 6px rgba(220, 38, 38, 0); }
  }

  .agentix-chat-phone-btn.active {
    background: #dc2626;
    color: white;
    animation: widget-phone-pulse 2s ease-in-out infinite;
  }

  @keyframes widget-phone-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
    50% { box-shadow: 0 0 0 8px rgba(220, 38, 38, 0); }
  }

  /* Message actions (speaker button etc.) – outside bubble */
  .agentix-chat-message-actions {
    position: absolute;
    right: -32px;
    top: 4px;
    display: flex;
    gap: 6px;
    opacity: 0;
    transition: opacity 0.2s ease;
    flex-shrink: 0;
  }

  .agentix-chat-message:hover .agentix-chat-message-actions {
    opacity: 1;
  }

  /* Speaker button */
  .agentix-chat-speak-btn {
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    color: var(--chat-text-secondary, #888);
    transition: background 0.2s ease, color 0.2s ease, transform 0.15s ease;
    position: relative;
  }

  .agentix-chat-speak-btn:hover {
    background: var(--chat-primary);
    color: white;
    transform: scale(1.1);
  }

  .agentix-chat-speak-btn:active {
    transform: scale(0.95);
  }

  .agentix-chat-speak-btn.speaking {
    background: #dc2626;
    color: white;
    opacity: 1;
    animation: widget-speak-pulse-red 1.2s ease-in-out infinite;
  }

  .agentix-chat-speak-btn .speak-ring {
    position: absolute;
    border-radius: 50%;
    border: 2px solid var(--chat-primary);
    opacity: 0;
    animation: widget-speak-pulse 1.4s ease-out infinite;
  }

  .agentix-chat-speak-btn.speaking .speak-ring {
    border-color: #dc2626;
  }

  .agentix-chat-speak-btn.speaking .speak-ring:nth-child(1) { animation-delay: 0s; }
  .agentix-chat-speak-btn.speaking .speak-ring:nth-child(2) { animation-delay: 0.35s; }
  .agentix-chat-speak-btn.speaking .speak-ring:nth-child(3) { animation-delay: 0.7s; }

  @keyframes widget-speak-pulse-red {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }

  @keyframes widget-speak-pulse {
    0% { width: 28px; height: 28px; opacity: 0.6; }
    100% { width: 48px; height: 48px; opacity: 0; }
  }

  .agentix-chat-speaking {
    animation: widget-speaking-glow 1s ease-in-out infinite alternate;
  }

  @keyframes widget-speaking-glow {
    from { box-shadow: 0 0 0 0 rgba(13, 61, 102, 0.2); }
    to { box-shadow: 0 0 0 6px rgba(13, 61, 102, 0.05); }
  }

  /* Reasoning Box */
  .agentix-chat-reasoning-box {
    margin-bottom: 8px;
    border-radius: 8px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    overflow: hidden;
  }
  .agentix-chat-reasoning-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 8px 12px;
    background: none;
    border: none;
    cursor: pointer;
    font-size: 12px;
    color: #475569;
  }
  .agentix-chat-reasoning-label {
    font-weight: 600;
  }
  .agentix-chat-reasoning-thinking-label {
    font-weight: 600;
  }
  .agentix-chat-reasoning-thinking-label::before {
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #3b82f6;
    margin-right: 6px;
    animation: widget-reasoning-dot 1.4s infinite;
  }
  @keyframes widget-reasoning-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
  .agentix-chat-reasoning-body {
    padding: 8px 12px;
    font-size: 13px;
    color: #334155;
    border-top: 1px solid #e2e8f0;
  }

  /* Tool Calls */
  .agentix-chat-toolcalls-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
  }
  .agentix-chat-toolcall-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    user-select: none;
  }
  .agentix-chat-toolcall-badge.ok {
    background: #dcfce7;
    color: #166534;
  }
  .agentix-chat-toolcall-badge.err {
    background: #fee2e2;
    color: #991b1b;
  }

  /* Terminal Box */
  .agentix-chat-terminal-box {
    margin-top: 8px;
    border-radius: 8px;
    background: #0f172a;
    border: 1px solid #1e293b;
    overflow: hidden;
  }
  .agentix-chat-terminal-toggle {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    padding: 8px 12px;
    background: none;
    border: none;
    cursor: pointer;
    font-size: 12px;
    color: #94a3b8;
  }
  .agentix-chat-terminal-label {
    font-weight: 600;
  }
  .agentix-chat-terminal-body {
    padding: 8px 12px;
    font-size: 12px;
    color: #e2e8f0;
    border-top: 1px solid #1e293b;
    max-height: 200px;
    overflow-y: auto;
  }
  .agentix-chat-terminal-prompt {
    color: #22c55e;
    margin-bottom: 4px;
  }
  .agentix-chat-terminal-prompt-char {
    color: #22c55e;
    margin-right: 4px;
  }
  .agentix-chat-terminal-code {
    background: #1e293b;
    padding: 8px;
    border-radius: 4px;
    margin: 4px 0;
    overflow-x: auto;
    color: #e2e8f0;
  }
  .agentix-chat-terminal-output {
    color: #cbd5e1;
    margin: 4px 0;
    white-space: pre-wrap;
  }

  /* Modal */
  .agentix-chat-modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000000;
    padding: 16px;
  }
  .agentix-chat-modal-box {
    background: var(--chat-bg, #fff);
    border-radius: 12px;
    max-width: 640px;
    width: 90vw;
    max-height: 80vh;
    overflow-y: auto;
    padding: 20px;
    box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1);
  }
  .agentix-chat-modal-box.preview-modal {
    max-width: 90vw;
    width: 90vw;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    padding: 0;
  }
  .agentix-chat-modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    padding: 16px 20px 0;
  }
  .agentix-chat-modal-header h4 {
    margin: 0;
    font-size: 16px;
  }
  .agentix-chat-modal-close {
    background: none;
    border: none;
    cursor: pointer;
    font-size: 18px;
    color: #64748b;
    padding: 4px;
    line-height: 1;
  }
  .agentix-chat-modal-label {
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 4px;
    margin-top: 12px;
  }
  .agentix-chat-modal-args {
    background: #f1f5f9;
    padding: 10px;
    border-radius: 8px;
    font-size: 12px;
    overflow-x: auto;
    margin: 0;
  }
  .agentix-chat-modal-meta {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 8px;
  }
  .agentix-chat-modal-err {
    font-size: 12px;
    color: #991b1b;
    background: #fee2e2;
    padding: 8px;
    border-radius: 6px;
    margin-bottom: 8px;
  }
  .agentix-chat-preview-content {
    flex: 1;
    padding: 0 20px 20px;
    min-height: 0;
  }
  .agentix-chat-preview-frame {
    width: 100%;
    height: 100%;
    border: none;
    border-radius: 8px;
    background: #fff;
  }

  /* HTML maximize button */
  .agentix-chat-html-iframe-wrapper {
    position: relative;
    max-width: 100%;
  }
  .agentix-chat-html-maximize {
    position: absolute;
    top: 4px;
    right: 4px;
    background: rgba(255,255,255,0.9);
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    cursor: pointer;
    padding: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
  }

  /* Emotion button */
  .agentix-chat-emotion-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    border: none;
    background: transparent;
    color: var(--chat-text-secondary, #666);
    cursor: pointer;
    transition: all 0.2s ease;
    flex-shrink: 0;
  }
  .agentix-chat-emotion-btn:hover {
    background: rgba(0,0,0,0.05);
  }
  .agentix-chat-emotion-btn.active {
    color: #0d3d66;
  }
  .agentix-chat-emotion-btn.active::after {
    content: '';
    position: absolute;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #22c55e;
    bottom: 4px;
    right: 4px;
  }
  .agentix-chat-emotion-btn.error {
    color: #dc2626;
  }
`;

