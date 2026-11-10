import type { ChatWidgetConfig, ChatMessage } from '../types/index';
import { OpenAIClient } from '../utils/api';
import { renderMarkdown, sanitizeInput, extractCodeBlocks, buildHtmlDoc } from '../utils/markdown';
import { mergeServerDefaults } from '../utils/defaults';
import { createElement, addEvent, scrollToBottom } from '../utils/dom';
import { extractCompleteSentences, TTSSegmentPlayer } from '../utils/tts';
import { blobToWav } from '../utils/audio';
import { WIDGET_STYLES } from './styles';
import { WIDGET_FAVICON_BASE64, WIDGET_LOGO_BASE64 } from '../assets/logos';
import { createVideoSnippetBuffer } from '../emotion/videoSnippetBuffer';
import { createAudioSnippetBuffer } from '../emotion/audioSnippetBuffer';
import { fuseEmotions } from '../emotion/emotionAggregator';
import type { EmotionContextPayload, EmotionRuntimeStatus, SnippetConfig, EmotionResult } from '../emotion/types';

const WIDGET_I18N: Record<'de' | 'en', Record<string, string>> = {
  de: {
    inputPlaceholder: 'Nachricht eingeben...',
    privacyTitle: 'Datenschutzinformation',
    privacyAccept: 'Ich akzeptiere',
    privacyDecline: 'Ablehnen',
    privacyDisabledPlaceholder: 'Datenschutz muss akzeptiert werden...',
    emptyStateText: 'Hallo! Ich bin dein Agentix CyberChatbot.<br>Wie kann ich dir heute helfen?',
    bounceText: 'Hier chatten!',
    initialMessageContent: 'Hallo!',
    privacyText: 'Willkommen beim Chatbot! Ihre Konversationen werden ausschließlich auf den Servern des Betreibers verarbeitet. Es werden keine personenbezogenen Daten an Dritte weitergegeben.',
    error429: 'Zu viele Anfragen. Bitte warten Sie einen Moment und versuchen es erneut.',
    error401: 'Nicht autorisiert. Bitte überprüfen Sie Ihre API-Konfiguration.',
    error403: 'Zugriff verweigert. Bitte überprüfen Sie Ihre Berechtigungen.',
    error404: 'Der angeforderte Dienst wurde nicht gefunden.',
    error500: 'Der Server ist vorübergehend nicht erreichbar. Bitte versuchen Sie es später erneut.',
    errorDefault: 'Ein Fehler ist aufgetreten. Bitte versuchen Sie es erneut.',
    codeView: 'Code',
    webView: 'Web',
    sttEmpty: 'Ich konnte dich leider nicht verstehen.',
  },
  en: {
    inputPlaceholder: 'Type your message here...',
    privacyTitle: 'Privacy Notice',
    privacyAccept: 'Accept',
    privacyDecline: 'Decline',
    privacyDisabledPlaceholder: 'Please accept the privacy notice to start chatting...',
    emptyStateText: 'Hello! I am your Agentix CyberChatbot.<br>How can I help you today?',
    bounceText: 'Chat here!',
    initialMessageContent: 'Hello! Language: English',
    privacyText: 'Welcome to the chatbot! Your conversations are processed exclusively on the operator\'s servers. No personal data is shared with third parties.',
    error429: 'Too many requests. Please wait a moment and try again.',
    error401: 'Unauthorized. Please check your API configuration.',
    error403: 'Access denied. Please check your permissions.',
    error404: 'The requested service was not found.',
    error500: 'The server is temporarily unavailable. Please try again later.',
    errorDefault: 'An error occurred. Please try again.',
    codeView: 'Code',
    webView: 'Web',
    sttEmpty: 'I could not understand you, sorry.',
  },
};

export function detectLanguage(): 'de' | 'en' {
  const browserLang = navigator.language;
  if (browserLang) {
    const normalized = browserLang.toLowerCase().split('-')[0];
    if (normalized === 'de') return 'de';
    if (normalized === 'en') return 'en';
  }
  const browserLangs = navigator.languages;
  if (browserLangs && browserLangs.length > 0) {
    const normalized = browserLangs.map((l) => l.toLowerCase().split('-')[0]);
    if (normalized.includes('de')) return 'de';
    if (normalized.includes('en')) return 'en';
  }
  const pageLang = document.documentElement.lang;
  if (pageLang) {
    const normalized = pageLang.toLowerCase().split('-')[0];
    if (normalized === 'de') return 'de';
    if (normalized === 'en') return 'en';
  }
  return 'de';
}

/**
 * Agentix CyberChatbot Widget - Main Component
 */
export class ChatWidget {
  private config: Required<ChatWidgetConfig>;
  private readonly userConfig: ChatWidgetConfig;
  private container: HTMLElement | null = null;
  private window: HTMLElement | null = null;
  private messagesContainer: HTMLElement | null = null;
  private inputField: HTMLTextAreaElement | null = null;
  private sendButton: HTMLElement | null = null;
  private stopButton: HTMLElement | null = null;
  private micButton: HTMLElement | null = null;
  private phoneButton: HTMLElement | null = null;
  private emotionButton: HTMLElement | null = null;
  private toggleButton: HTMLElement | null = null;
  private isOpen: boolean = false;
  private isLoading: boolean = false;
  private messages: ChatMessage[] = [];
  private apiClient: OpenAIClient | null = null;
  private abortController: AbortController | null = null;
  private unreadCount: number = 0;
  private styleElement: HTMLStyleElement | null = null;
  private privacyAccepted: boolean = false;
  private privacyModal: HTMLElement | null = null;
  private bounceTimer: number | null = null;
  private bubbleElement: HTMLElement | null = null;
  private _warnedNoKey: boolean = false;
  private audioCapabilities: {
    stt: { button_visible: boolean; language?: string; provider?: string; model?: string }
    tts: { button_visible: boolean; voice_id: string | null; auto_read?: boolean; streaming_enabled?: boolean }
    phone: { button_visible: boolean }
  } | null = null;
  private ttsSentenceBuffer: string = '';
  private ttsSegmentPlayer: import('../utils/tts').TTSSegmentPlayer | null = null;
  private ttsAudio: HTMLAudioElement | null = null;
  private isRecording: boolean = false;
  private phoneMode: boolean = false;
  private awaitingUserSpeech: boolean = false;
  private vadSpeechDetected: boolean = false;
  private recordingStartTime: number = 0;
  private phoneEmptyRetries: number = 0;
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private vadSilenceStart: number = 0;
  private vadRafId: number | null = null;
  private previewHtml: string | null = null;
  private modalOverlay: HTMLElement | null = null;
  private emotionConfig: {
    enabled: boolean;
    text: { enabled: boolean; rules_enabled: boolean };
    webcam: {
      enabled: boolean;
      button_visible: boolean;
      runtime: string;
      execution_providers: string[];
      face_detector: string;
      face_model_url: string;
      expression_model_url: string;
      snippet_window_ms: number;
      sample_fps: number;
      max_frames_per_snippet: number;
      min_valid_face_frames: number;
      frame_upload_enabled: boolean;
    };
    audio: { enabled: boolean; model_url: string; min_snr_db?: number; noise_gate_min?: number; noise_gate_max?: number };
    prompt_steering: { enabled: boolean; min_confidence: number };
    analytics: { enabled: boolean };
  } | null = null;
  private webcamConsent = false;
  private webcamActive = false;
  private emotionRuntimeStatus: EmotionRuntimeStatus = 'uninitialized';
  private snippetBufferRef: ReturnType<typeof createVideoSnippetBuffer> | null = null;
  private audioSnippetBufferRef: ReturnType<typeof createAudioSnippetBuffer> | null = null;
  private emotionTimelineRef: {
    phases: Record<import('../emotion/types').EmotionPhase, { snapshots: EmotionResult[]; startTime: number; endTime?: number }>;
    activePhase: import('../emotion/types').EmotionPhase;
    startTime: number;
  } | null = null;
  private emotionPollIntervalRef: ReturnType<typeof setInterval> | null = null;
  private lastEmotionModelRef: string | null = null;
  private continuousAudioStreamRef: MediaStream | null = null;
  private hasTransitionedToTypingRef = false;
  private audioSnapshotsRef: EmotionResult[] = [];
  private userScrolledUpRef = false;
  private isStartingAudioRef = false;
  private isStartingWebcamRef = false;
  private windowResizeHandler: (() => void) | null = null;

  constructor(config: ChatWidgetConfig = {}) {
    this.userConfig = config;
    const lang = (config.language && config.language as string !== 'auto') ? config.language : detectLanguage();
    const i18n = WIDGET_I18N[lang];
    const privacyOverride = lang === 'de' ? config.privacyTextDe : config.privacyTextEn;
    const initialOverride = lang === 'de' ? config.initialMessageContentDe : config.initialMessageContentEn;
    const bounceOverride = lang === 'de' ? config.bounceTextDe : config.bounceTextEn;
    const placeholderOverride = lang === 'de' ? config.inputPlaceholderDe : config.inputPlaceholderEn;
    this.config = {
      apiKey: config.apiKey || import.meta.env.VITE_OPENAI_API_KEY || '',
      apiUrl: config.apiUrl || import.meta.env.VITE_OPENAI_API_URL || (window.location.origin + '/v1'),
      model: config.model || import.meta.env.VITE_OPENAI_MODEL || 'CyberSecurityChatbot',
      position: config.position || 'bottom-right',
      theme: config.theme || 'light',
      language: lang,
      systemPrompt: config.systemPrompt || '',
      width: config.width || 650,
      height: config.height || 550,
      showOnLoad: config.showOnLoad ?? false,
      title: config.title || 'CyberSecurityChatbot',
      subtitle: config.subtitle || 'Powered by Agentix',
      logoUrl: config.logoUrl || '',
      logoUrlButton: config.logoUrlButton || config.logoUrl || WIDGET_FAVICON_BASE64,
      logoUrlHeader: config.logoUrlHeader || config.logoUrl || WIDGET_LOGO_BASE64,
      showLogoInHeader: config.showLogoInHeader ?? true,
      showLogoInButton: config.showLogoInButton ?? true,
      logoBorder: config.logoBorder || { enabled: false, color: '#ffffff', opacity: 0.35 },
      logoGlow: config.logoGlow || { enabled: true, color: '#ffffff', opacity: 0.4 },
      logoBackground: config.logoBackground || { enabled: true, color: '#ffffff', opacity: 0.35 },
      privacyText: privacyOverride || config.privacyText || i18n.privacyText,
      privacyTextDe: config.privacyTextDe || '',
      privacyTextEn: config.privacyTextEn || '',
      privacyEnabled: config.privacyEnabled ?? true,
      initialMessage: config.initialMessage ?? true,
      initialMessageContent: initialOverride || config.initialMessageContent || i18n.initialMessageContent,
      initialMessageContentDe: config.initialMessageContentDe || '',
      initialMessageContentEn: config.initialMessageContentEn || '',
      initialMessageSilent: config.initialMessageSilent ?? true,
      bounceInterval: config.bounceInterval ?? 15000,
      bounceText: bounceOverride || config.bounceText || i18n.bounceText,
      bounceTextDe: config.bounceTextDe || '',
      bounceTextEn: config.bounceTextEn || '',
      inputPlaceholder: placeholderOverride || i18n.inputPlaceholder,
      inputPlaceholderDe: config.inputPlaceholderDe || '',
      inputPlaceholderEn: config.inputPlaceholderEn || '',
      colors: config.colors || {},
      onInit: config.onInit || (() => {}),
      onOpen: config.onOpen || (() => {}),
      onError: config.onError || ((error) => console.error(error))
    };

    this.validateConfig();
    this.init();
    this.loadServerDefaults();
  }

  /**
   * Fetch server-side widget defaults and merge with local config.
   * Explicitly provided embed config always wins over server defaults
   * (except object keys like `colors`, which merge sub-key level).
   */
  private async loadServerDefaults(): Promise<void> {
    try {
      const baseUrl = this.config.apiUrl.replace(/\/v1$/, '');
      // Fetch widget defaults and global branding in parallel; either may fail.
      const [defRes, brandRes] = await Promise.allSettled([
        fetch(`${baseUrl}/widget-defaults`, { credentials: 'omit' }),
        fetch(`${baseUrl}/branding`, { credentials: 'omit' }),
      ]);
      let defaults: unknown = null;
      if (defRes.status === 'fulfilled' && defRes.value.ok) {
        defaults = await defRes.value.json();
      }
      // Branding fields are the lowest-priority defaults: builtin < branding <
      // widget defaults < explicit embed config.
      const brandDefaults: Record<string, unknown> = {};
      if (brandRes.status === 'fulfilled' && brandRes.value.ok) {
        const branding = await brandRes.value.json();
        if (branding && typeof branding === 'object' && !Array.isArray(branding)) {
          if (branding.logo_url) {
            brandDefaults.logoUrlHeader = branding.logo_url;
            brandDefaults.logoUrlButton = branding.logo_url;
          }
          if (!brandDefaults.logoUrlButton && branding.favicon_url) {
            brandDefaults.logoUrlButton = branding.favicon_url;
          }
          if (branding.app_name) brandDefaults.title = branding.app_name;
          if (branding.powered_by) brandDefaults.subtitle = branding.powered_by;
        }
      }
      const validDefaults =
        defaults && typeof defaults === 'object' && !Array.isArray(defaults)
          ? (defaults as Record<string, unknown>)
          : {};
      const merged = { ...brandDefaults, ...validDefaults };
      if (Object.keys(merged).length === 0) return;
      const filtered = mergeServerDefaults(
        this.userConfig as Record<string, unknown>,
        merged,
      );
      if (Object.keys(filtered).length === 0) return;
      this.updateConfig(filtered as Partial<ChatWidgetConfig>);
      console.log('[Widget] Merged server defaults:', filtered);
    } catch {
      // Silently ignore – widget works fine without server defaults
    }
  }

  /**
   * Validate configuration
   */
  private validateConfig(): void {
    if (!this.config.apiKey && !this._warnedNoKey) {
      this._warnedNoKey = true;
      const warning = new Error('No API key provided. Widget will attempt unauthenticated requests; ensure your API allows this.');
      this.config.onError(warning);
      // Allow initialization to proceed for open APIs without throwing.
    }
  }

  /**
   * Initialize widget
   */
  private init(): void {
    try {
      this.setupAPIClient();
      this.injectStyles();
      this.createDOM();
      this.applyColorVariables();
      this.attachEventListeners();
      this.loadMessages();
      this.checkPrivacyStatus();
      this.fetchAudioCapabilities();
      this.fetchEmotionConfig();

      if (!this.isOpen) {
        this.startBounce();
      }

      this.config.onInit();
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Widget initialization failed');
      this.config.onError(err);
    }
  }

  /**
   * Check privacy consent status
   */
  private checkPrivacyStatus(): void {
    // If privacy overlay is disabled, auto-accept
    if (!this.config.privacyEnabled) {
      this.privacyAccepted = true;
      this.enableChat();
      if (this.privacyModal) {
        this.privacyModal.style.display = 'none';
      }
      if (this.window && this.shouldOpenOnLoad()) {
        this.isOpen = true;
        this.window.classList.add('open');
        this.toggleButton?.setAttribute('aria-expanded', 'true');
        this.unreadCount = 0;
        this.updateBadge();
        this.inputField?.focus();
        this.config.onOpen();
        this.sendInitialMessageIfNeeded();
      }
      return;
    }

    const accepted = localStorage.getItem('agentix-chat-privacy-accepted');
    if (accepted === 'true') {
      this.privacyAccepted = true;
      this.enableChat();
      // Open window if showOnLoad is true and user hasn't closed it yet
      if (this.window && this.shouldOpenOnLoad()) {
        this.isOpen = true;
        this.window.classList.add('open');
        this.toggleButton?.setAttribute('aria-expanded', 'true');
        this.unreadCount = 0;
        this.updateBadge();
        this.inputField?.focus();
        this.config.onOpen();
        this.sendInitialMessageIfNeeded();
      }
    } else {
      this.privacyAccepted = false;
      this.disableChat();
      // Open window and show privacy modal if not yet accepted and user hasn't closed it yet
      if (this.window && this.shouldOpenOnLoad()) {
        this.isOpen = true;
        this.window.classList.add('open');
        this.toggleButton?.setAttribute('aria-expanded', 'true');
        this.unreadCount = 0;
        this.updateBadge();
        // Show privacy modal
        if (this.privacyModal) {
          this.privacyModal.style.display = 'flex';
        }
      }
    }
  }

  /**
   * Determine if widget should open on load based on stored state
   */
  private shouldOpenOnLoad(): boolean {
    const stored = localStorage.getItem('agentix-chat-open');
    if (stored === null) return this.config.showOnLoad; // First visit
    return stored === 'true';
  }

  /**
   * Setup OpenAI API client
   */
  private setupAPIClient(): void {
    const sessionConversationId = this.getOrCreateSessionConversationId();
    this.apiClient = new OpenAIClient(
      this.config.apiKey,
      this.config.apiUrl,
      this.config.model,
      sessionConversationId
    );
  }

  /**
   * Fetch audio capabilities from backend
   */
  private async fetchAudioCapabilities(): Promise<void> {
    try {
      const baseUrl = this.config.apiUrl.replace(/\/v1$/, '');
      const url = `${baseUrl}/audio/config?model=${encodeURIComponent(this.config.model)}`;
      console.log('[Widget] Fetching audio config:', url);
      const res = await fetch(url, {
        credentials: 'include',
      });
      if (!res.ok) {
        console.warn('[Widget] Audio config fetch failed:', res.status, await res.text());
        this.audioCapabilities = null;
        if (this.micButton) this.micButton.style.display = 'none';
        if (this.phoneButton) this.phoneButton.style.display = 'none';
        return;
      }
      this.audioCapabilities = await res.json();
      console.log('[Widget] Audio capabilities:', this.audioCapabilities);
      if (this.micButton) {
        this.micButton.style.display = this.audioCapabilities?.stt?.button_visible ? '' : 'none';
      }
      if (this.phoneButton) {
        this.phoneButton.style.display = this.audioCapabilities?.phone?.button_visible ? '' : 'none';
      }
      // Update visibility of existing speak buttons
      const speakWrappers = this.container?.querySelectorAll('.agentix-chat-message-actions');
      if (speakWrappers) {
        speakWrappers.forEach((wrapper) => {
          (wrapper as HTMLElement).style.display = this.audioCapabilities?.tts?.button_visible ? '' : 'none';
        });
      }
    } catch (err) {
      console.warn('[Widget] Audio config error:', err);
      this.audioCapabilities = null;
      if (this.micButton) this.micButton.style.display = 'none';
      if (this.phoneButton) this.phoneButton.style.display = 'none';
      const speakWrappers = this.container?.querySelectorAll('.agentix-chat-message-actions');
      if (speakWrappers) {
        speakWrappers.forEach((wrapper) => {
          (wrapper as HTMLElement).style.display = 'none';
        });
      }
    }
  }

  /**
   * Fetch emotion config from backend
   */
  private async fetchEmotionConfig(): Promise<void> {
    if (this.lastEmotionModelRef === this.config.model) return;
    try {
      const baseUrl = this.config.apiUrl.replace(/\/v1$/, '');
      const res = await fetch(`${baseUrl}/emotion/config?model=${encodeURIComponent(this.config.model)}`, {
        credentials: 'include',
      });
      if (!res.ok) {
        this.emotionConfig = null;
        return;
      }
      this.emotionConfig = await res.json();
      this.lastEmotionModelRef = this.config.model;
      this.updateEmotionButtonVisibility();
    } catch {
      this.emotionConfig = null;
      this.updateEmotionButtonVisibility();
    }
  }

  /**
   * Update emotion button visibility based on mode and config
   */
  private updateEmotionButtonVisibility(): void {
    if (!this.emotionButton) return;
    if (this.phoneMode) {
      this.emotionButton.style.display = this.emotionConfig?.enabled ? '' : 'none';
    } else {
      this.emotionButton.style.display = this.emotionConfig?.webcam?.enabled ? '' : 'none';
    }
  }

  /**
   * Smart scroll to bottom (respects user scroll up)
   */
  private smartScrollToBottom(): void {
    if (!this.userScrolledUpRef && this.messagesContainer) {
      scrollToBottom(this.messagesContainer);
    }
  }

  /**
   * Start webcam emotion detection
   */
  private async startWebcamEmotion(): Promise<void> {
    if (!this.emotionConfig?.webcam?.enabled || this.webcamActive) return;
    if (this.isStartingWebcamRef) return;
    if (!this.webcamConsent) {
      const agreed = window.confirm(
        'Webcam-Emotion Detection\n\n' +
        'Die Kamera wird lokal im Browser ausgewertet. ' +
        'Es werden keine Bilder oder Videos aufgezeichnet oder übertragen. ' +
        'Nur aggregierte Emotions-Metadaten werden gesendet.\n\n' +
        'Möchten Sie fortfahren?'
      );
      if (!agreed) return;
      this.webcamConsent = true;
    }
    this.isStartingWebcamRef = true;
    this.emotionRuntimeStatus = 'loading';
    try {
      const cfg = this.emotionConfig.webcam;
      const snippetConfig: SnippetConfig = {
        windowMs: cfg.snippet_window_ms,
        sampleFps: cfg.sample_fps,
        maxFrames: cfg.max_frames_per_snippet,
        minValidFaceFrames: cfg.min_valid_face_frames,
        faceModelUrl: cfg.face_model_url,
        expressionModelUrl: cfg.expression_model_url,
        executionProviders: cfg.execution_providers,
      };
      this.snippetBufferRef = createVideoSnippetBuffer(snippetConfig);
      await this.snippetBufferRef.start();
      if (this.isStartingWebcamRef) {
        // Start audio independently of webcam
        if (this.emotionConfig?.audio?.enabled) {
          await this.startContinuousAudioEmotion().catch(() => {});
        }
        this.webcamActive = true;
        this.emotionRuntimeStatus = 'ready';
      }
    } catch {
      this.emotionRuntimeStatus = 'error';
      this.webcamActive = false;
    } finally {
      this.isStartingWebcamRef = false;
    }
  }

  /**
   * Stop webcam emotion detection
   */
  private stopWebcamEmotion(): void {
    this.isStartingWebcamRef = false;
    if (this.snippetBufferRef) {
      this.snippetBufferRef.stop();
      this.snippetBufferRef = null;
    }
    this.stopContinuousAudioEmotion();
    this.webcamActive = false;
    this.emotionRuntimeStatus = 'uninitialized';
  }

  /**
   * Start continuous audio emotion
   */
  private async startContinuousAudioEmotion(): Promise<void> {
    if (this.audioSnippetBufferRef?.isActive()) return;
    if (this.isStartingAudioRef) return;
    if (!this.emotionConfig?.audio?.enabled) {
      console.info('[Widget] Audio emotion disabled in config');
      return;
    }
    this.isStartingAudioRef = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!this.isStartingAudioRef) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      this.continuousAudioStreamRef = stream;
      const cfg = this.emotionConfig.webcam;
      const audioCfg: SnippetConfig = {
        windowMs: cfg?.snippet_window_ms || 3000,
        sampleFps: cfg?.sample_fps || 3,
        maxFrames: cfg?.max_frames_per_snippet || 12,
        minValidFaceFrames: cfg?.min_valid_face_frames || 3,
        faceModelUrl: cfg?.face_model_url || '/models/emotion/face-detector.onnx',
        expressionModelUrl: this.emotionConfig.audio.model_url || '/models/emotion/audio-emotion.onnx',
        executionProviders: cfg?.execution_providers || ['wasm'],
        minSnrDb: this.emotionConfig.audio.min_snr_db ?? 10,
        noiseGateMin: this.emotionConfig.audio.noise_gate_min ?? 0.001,
        noiseGateMax: this.emotionConfig.audio.noise_gate_max ?? 0.5,
      };
      console.info('[Widget] Starting continuous audio emotion with modelUrl:', audioCfg.expressionModelUrl);
      this.audioSnippetBufferRef = createAudioSnippetBuffer(audioCfg);
      this.audioSnippetBufferRef.start(stream);
      console.info('[Widget] Continuous audio emotion started');
    } catch (err) {
      console.warn('[Widget] Audio emotion start failed:', err);
    } finally {
      this.isStartingAudioRef = false;
    }
  }

  /**
   * Stop continuous audio emotion
   */
  private stopContinuousAudioEmotion(): void {
    this.isStartingAudioRef = false;
    if (this.audioSnippetBufferRef) {
      this.audioSnippetBufferRef.stop();
      this.audioSnippetBufferRef = null;
    }
    if (this.continuousAudioStreamRef) {
      this.continuousAudioStreamRef.getTracks().forEach((t) => t.stop());
      this.continuousAudioStreamRef = null;
    }
  }

  /**
   * Capture pre-send emotion snapshot
   */
  private async capturePreSendEmotion(): Promise<EmotionContextPayload['visual_pre_send'] | null> {
    if (!this.snippetBufferRef || !this.snippetBufferRef.isActive()) return null;
    return this.snippetBufferRef.captureSnapshot();
  }

  /**
   * Build emotion payload for API request
   */
  private buildEmotionPayload(): EmotionContextPayload | undefined {
    if (!this.emotionConfig?.enabled) return undefined;
    const payload: EmotionContextPayload = {};
    payload.consent = {
      webcam_opt_in: this.webcamConsent,
      frames_sent_to_backend: false,
      audio_opt_in: this.audioSnippetBufferRef?.isActive() ?? false,
    };
    return payload;
  }

  /**
   * Start continuous emotion polling
   */
  private startEmotionPolling(initialPhase: import('../emotion/types').EmotionPhase = 'waiting_for_response'): void {
    if (this.emotionPollIntervalRef) return;
    this.hasTransitionedToTypingRef = false;
    this.audioSnapshotsRef = [];
    const phases = this.createEmptyPhases();
    phases[initialPhase].startTime = Date.now();
    this.emotionTimelineRef = {
      phases,
      activePhase: initialPhase,
      startTime: Date.now(),
    };
    this.emotionPollIntervalRef = setInterval(() => {
      this.processEmotionBatch();
    }, 2000);
  }

  /**
   * Stop continuous emotion polling
   */
  private stopEmotionPolling(): void {
    if (this.emotionPollIntervalRef) {
      clearInterval(this.emotionPollIntervalRef);
      this.emotionPollIntervalRef = null;
    }
    this.processEmotionBatch();
  }

  /**
   * Switch emotion phase
   */
  private switchEmotionPhase(nextPhase: import('../emotion/types').EmotionPhase): void {
    const tl = this.emotionTimelineRef;
    if (!tl || tl.activePhase === nextPhase) return;
    const current = tl.phases[tl.activePhase];
    if (current.snapshots.length > 0) {
      const fused = fuseEmotions(current.snapshots);
      if (fused) {
        tl.phases[tl.activePhase] = {
          snapshots: [fused],
          startTime: current.startTime,
          endTime: Date.now(),
        };
      } else {
        current.endTime = Date.now();
      }
    } else {
      current.endTime = Date.now();
    }
    tl.phases[nextPhase].startTime = Date.now();
    tl.activePhase = nextPhase;
  }

  private createEmptyPhases(): Record<import('../emotion/types').EmotionPhase, { snapshots: EmotionResult[]; startTime: number; endTime?: number }> {
    return {
      waiting_for_response: { snapshots: [], startTime: Date.now() },
      reading: { snapshots: [], startTime: 0 },
      typing: { snapshots: [], startTime: 0 },
      speaking: { snapshots: [], startTime: 0 },
    };
  }

  /**
   * Process emotion batch
   */
  private processEmotionBatch(): void {
    const tl = this.emotionTimelineRef;
    if (!tl) return;
    if (this.snippetBufferRef?.isActive()) {
      this.snippetBufferRef.captureSnapshot().then((res) => {
        if (res) tl.phases[tl.activePhase].snapshots.push(res);
      }).catch(() => {});
    }
    if (this.audioSnippetBufferRef?.isActive()) {
      this.audioSnippetBufferRef.captureSnapshot().then((res) => {
        if (res) {
          tl.phases[tl.activePhase].snapshots.push(res);
          this.audioSnapshotsRef.push(res);
        }
      }).catch(() => {});
    }
  }

  /**
   * Build emotion phases
   */
  private buildEmotionPhases(): import('../emotion/types').PhaseEmotion[] {
    const tl = this.emotionTimelineRef;
    if (!tl) return [];
    const result: import('../emotion/types').PhaseEmotion[] = [];
    const phaseOrder: import('../emotion/types').EmotionPhase[] = ['waiting_for_response', 'reading', 'typing', 'speaking'];
    for (const phase of phaseOrder) {
      const data = tl.phases[phase];
      if (data.snapshots.length === 0 && !data.endTime && phase !== tl.activePhase) {
        continue;
      }
      if (data.snapshots.length > 0) {
        const fused = fuseEmotions(data.snapshots);
        if (fused) {
          result.push({
            phase,
            primary_emotion: fused.primary_emotion,
            confidence: fused.confidence,
            intensity: fused.intensity,
            source: `phase__${phase}`,
            timestamp_ms: data.startTime || Date.now(),
            sampled_frames: fused.sampled_frames,
            valid_face_frames: fused.valid_face_frames,
            face_detection_rate: fused.face_detection_rate,
            window_ms: fused.window_ms,
          });
          continue;
        }
      }
      if (data.startTime > 0 || phase === tl.activePhase) {
        result.push({
          phase,
          primary_emotion: 'unknown',
          confidence: 0,
          intensity: 'low',
          source: `phase__${phase}`,
          timestamp_ms: data.startTime || Date.now(),
        });
      }
    }
    return result;
  }

  /**
   * Compute overall fused emotion
   */
  private computeOverallFused(phases: import('../emotion/types').PhaseEmotion[]): EmotionResult | null {
    if (phases.length === 0) return null;
    const asResults: EmotionResult[] = phases.map((p) => ({
      primary_emotion: p.primary_emotion,
      confidence: p.confidence,
      intensity: p.intensity,
      source: p.source,
      timestamp_ms: p.timestamp_ms,
      sampled_frames: p.sampled_frames,
      valid_face_frames: p.valid_face_frames,
      face_detection_rate: p.face_detection_rate,
      window_ms: p.window_ms,
    }));
    return fuseEmotions(asResults);
  }

  /**
   * Clean text for TTS: remove URLs, markdown, HTML tags, and emojis
   */
  private cleanTextForTTS(text: string): string {
    let clean = text;
    // Replace markdown links [text](url) with just the text
    clean = clean.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
    // Replace bare URLs with a placeholder
    clean = clean.replace(/https?:\/\/[^\s]+/g, 'Link');
    // Remove markdown heading markers and add a period for prosody
    clean = clean.replace(/^#{1,6}\s+/gm, '');
    // Remove markdown formatting characters
    clean = clean.replace(/[*_`#]/g, '');
    // Remove markdown list markers at line start
    clean = clean.replace(/^[\-*]\s+/gm, '');
    clean = clean.replace(/^\d+\.\s+/gm, '');
    // Remove emojis
    clean = clean.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{FE00}-\u{FE0F}\u{1F900}-\u{1F9FF}\u{1F018}-\u{1F270}\u{238C}\u{2B06}\u{2B07}\u{2B05}\u{27A1}\u{2194}-\u{21AA}]/gu, '');
    // Remove HTML tags
    clean = clean.replace(/<[^>]+>/g, '');
    // Collapse multiple spaces inside a line, but keep paragraph breaks
    clean = clean.replace(/[ \t]+/g, ' ');
    // Trim each line
    clean = clean.split('\n').map((l) => l.trim()).join('\n');
    // Normalize paragraph breaks to exactly two newlines
    clean = clean.replace(/\n{3,}/g, '\n\n');
    return clean.trim();
  }

  /**
   * Stop all TTS playback (single audio and streaming player)
   */
  private stopAllTTS(): void {
    if (this.ttsAudio) {
      this.ttsAudio.pause();
      this.ttsAudio = null;
    }
    this.ttsSegmentPlayer?.abort();
    this.ttsSegmentPlayer = null;
    this.ttsSentenceBuffer = '';
    this.switchEmotionPhase('typing');
    // Remove speaking state from all buttons and messages
    if (this.messagesContainer) {
      this.messagesContainer.querySelectorAll('.agentix-chat-speaking').forEach((el) => el.classList.remove('agentix-chat-speaking'));
      this.messagesContainer.querySelectorAll('.agentix-chat-speak-btn.speaking').forEach((el) => {
        el.classList.remove('speaking');
        const playIcon = (el as HTMLElement).querySelector('.speak-icon') as HTMLElement | null;
        const stopIcon = (el as HTMLElement).querySelector('.speak-stop-icon') as HTMLElement | null;
        if (playIcon) playIcon.style.display = '';
        if (stopIcon) stopIcon.style.display = 'none';
      });
    }
  }

  /**
   * Speak text via TTS (toggle: click again to stop)
   */
  private async speakText(text: string, messageElement: HTMLElement, speakBtn?: HTMLElement): Promise<void> {
    if (!this.audioCapabilities?.tts?.button_visible || !this.audioCapabilities?.tts?.voice_id) return;
    // If already speaking, stop everything
    if (this.ttsAudio || (this.ttsSegmentPlayer && this.ttsSegmentPlayer.queueLength > 0)) {
      this.stopAllTTS();
      return;
    }
    const cleanText = this.cleanTextForTTS(text);
    if (!cleanText) return;
    const voiceId = this.audioCapabilities.tts.voice_id;

    const stopSpeaking = () => {
      messageElement.classList.remove('agentix-chat-speaking');
      speakBtn?.classList.remove('speaking');
      const playIcon = speakBtn?.querySelector('.speak-icon') as HTMLElement | null;
      const stopIcon = speakBtn?.querySelector('.speak-stop-icon') as HTMLElement | null;
      if (playIcon) playIcon.style.display = '';
      if (stopIcon) stopIcon.style.display = 'none';
      this.switchEmotionPhase('typing');
      if (this.phoneMode) {
        this.awaitingUserSpeech = true;
        this.updateMicButton();
        setTimeout(() => this.startRecording(), 300);
      }
    };

    try {
      messageElement.classList.add('agentix-chat-speaking');
      speakBtn?.classList.add('speaking');
      this.switchEmotionPhase('speaking');
      const playIcon = speakBtn?.querySelector('.speak-icon') as HTMLElement | null;
      const stopIcon = speakBtn?.querySelector('.speak-stop-icon') as HTMLElement | null;
      if (playIcon) playIcon.style.display = 'none';
      if (stopIcon) stopIcon.style.display = '';

      // Use sentence-level streaming for faster playback start
      this.ttsSegmentPlayer = new TTSSegmentPlayer();
      const { chunks } = extractCompleteSentences(cleanText);
      for (const chunk of chunks) {
        this.ttsSegmentPlayer.add(chunk.text, voiceId, this.fetchTTSSegment.bind(this), chunk.pauseAfterMs);
      }

      // Wait for playback to finish, then clean up
      const waitForPlayback = () => {
        if (this.ttsSegmentPlayer && this.ttsSegmentPlayer.queueLength > 0) {
          setTimeout(waitForPlayback, 200);
        } else {
          stopSpeaking();
          this.ttsSegmentPlayer = null;
        }
      };
      setTimeout(waitForPlayback, 200);
    } catch {
      messageElement.classList.remove('agentix-chat-speaking');
      speakBtn?.classList.remove('speaking');
      const playIcon = speakBtn?.querySelector('.speak-icon') as HTMLElement | null;
      const stopIcon = speakBtn?.querySelector('.speak-stop-icon') as HTMLElement | null;
      if (playIcon) playIcon.style.display = '';
      if (stopIcon) stopIcon.style.display = 'none';
      this.ttsSegmentPlayer = null;
    }
  }

  /**
   * STT Recording
   */
  private async startRecording(): Promise<void> {
    console.info('[Widget] startRecording called, isRecording=', this.isRecording);
    if (!this.audioCapabilities?.stt?.button_visible || this.isRecording) {
      console.info('[Widget] startRecording early return');
      return;
    }
    // If emotion button is not active but audio emotion is enabled,
    // start a temporary continuous audio buffer for this STT session only
    if (!this.webcamActive && this.emotionConfig?.audio?.enabled && !this.audioSnippetBufferRef?.isActive()) {
      console.info('[Widget] startRecording: auto-starting temporary audio emotion buffer');
      await this.startContinuousAudioEmotion().catch(() => {});
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : MediaRecorder.isTypeSupported('audio/ogg')
          ? 'audio/ogg'
          : '';
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      this.mediaRecorder = recorder;
      this.audioChunks = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) this.audioChunks.push(e.data);
      };
      recorder.onstop = async () => {
        console.info('[Widget] recorder.onstop fired');
        stream.getTracks().forEach((t) => t.stop());
        this.isRecording = false;
        this.updateMicButton();

        // Capture audio prosody emotion from the continuous buffer (if running)
        let audioEmotion: EmotionContextPayload['audio_prosody'] | null = null;
        console.info('[Widget] STT stop: audioSnippetBuffer active=', this.audioSnippetBufferRef?.isActive());
        if (this.audioSnippetBufferRef?.isActive()) {
          audioEmotion = await this.audioSnippetBufferRef.captureSnapshot();
          console.info('[Widget] STT captureSnapshot result:', audioEmotion);
        }

        // Stop the temporary audio emotion stream if emotion button is not active
        if (!this.webcamActive) {
          this.stopContinuousAudioEmotion();
        }

        if (this.audioChunks.length === 0) return;
        const audioBlob = new Blob(this.audioChunks, { type: mimeType || 'audio/webm' });
        try {
          const wavBlob = await blobToWav(audioBlob);
          const formData = new FormData();
          formData.append('audio', wavBlob, 'recording.wav');
          if (this.audioCapabilities?.stt?.language) {
            formData.append('language', this.audioCapabilities.stt.language);
          }
          const baseUrl = this.config.apiUrl.replace(/\/v1$/, '');
          const res = await fetch(`${baseUrl}/audio/stt`, {
            method: 'POST',
            credentials: 'include',
            body: formData,
          });
          if (!res.ok) {
            if (res.status === 400) {
              const i18n = WIDGET_I18N[this.config.language];
              this._showToast(i18n.sttEmpty || 'Ich konnte dich leider nicht verstehen.');
            }
            throw new Error(await res.text());
          }
          const data = await res.json();
          const transcript = (data.text || '').trim();

          const hasAudioEmotions = audioEmotion;
          if (transcript.length >= 2 && this.inputField) {
            this.inputField.value = transcript;
            this.autoResizeTextarea();
          }
          if (transcript.length >= 2 || hasAudioEmotions) {
            this.phoneEmptyRetries = 0;
            setTimeout(() => {
              if (this.inputField) this.inputField.value = '';
              this.sendMessage(false, transcript, { audio_prosody: audioEmotion });
            }, 200);
          } else if (this.phoneMode) {
            this.phoneEmptyRetries++;
            if (this.phoneEmptyRetries < 3) {
              this.awaitingUserSpeech = true;
              this.updateMicButton();
              setTimeout(() => this.startRecording(), 400);
            } else {
              this.phoneEmptyRetries = 0;
              this.awaitingUserSpeech = false;
              this.updateMicButton();
            }
          }
        } catch (err: any) {
          console.error('STT error:', err);
        }
      };

      recorder.start();
      this.isRecording = true;
      this.awaitingUserSpeech = false;
      this.updateMicButton();

      /* ---- Voice Activity Detection (VAD) for phone mode ---- */
      if (this.phoneMode) {
        const audioCtx = new AudioContext();
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 512;
        const source = audioCtx.createMediaStreamSource(stream);
        source.connect(analyser);
        this.audioContext = audioCtx;
        this.analyser = analyser;
        this.vadSilenceStart = performance.now();
        this.vadSpeechDetected = false;
        this.recordingStartTime = performance.now();

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        const checkSilence = () => {
          if (!this.analyser || !this.mediaRecorder) return;
          this.analyser.getByteTimeDomainData(dataArray);
          let sum = 0;
          for (let i = 0; i < dataArray.length; i++) {
            const x = (dataArray[i] - 128) / 128;
            sum += x * x;
          }
          const rms = Math.sqrt(sum / dataArray.length);
          const now = performance.now();
          if (rms >= 0.02) {
            this.vadSpeechDetected = true;
            this.vadSilenceStart = now;
          }
          if (this.vadSpeechDetected && now - this.recordingStartTime > 2000 && rms < 0.02) {
            if (now - this.vadSilenceStart > 1500) {
              this.stopRecording();
              return;
            }
          }
          this.vadRafId = requestAnimationFrame(checkSilence);
        };
        this.vadRafId = requestAnimationFrame(checkSilence);
      }

      const timeoutRecorder = recorder;
      setTimeout(() => {
        if (timeoutRecorder.state === 'recording') {
          timeoutRecorder.stop();
        }
      }, 30000);
    } catch (err: any) {
      const helpMsg =
        'Mikrofon-Zugriff verweigert.\n\n' +
        'Bitte pruefe:\n' +
        '1. Browser-Adressleiste -> Schloss -> Berechtigungen -> Mikrofon -> Erlauben\n' +
        '2. macOS -> Systemeinstellungen -> Datenschutz & Sicherheit -> Mikrofon -> Browser aktivieren\n' +
        '3. Seite danach neu laden (F5).';
      alert(helpMsg);
    }
  }

  private stopRecording(): void {
    console.info('[Widget] stopRecording called, isRecording=', this.isRecording);
    if (this.vadRafId) {
      cancelAnimationFrame(this.vadRafId);
      this.vadRafId = null;
    }
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    this.analyser = null;
    this.vadSpeechDetected = false;
    this.recordingStartTime = 0;
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.stop();
      this.mediaRecorder.stream?.getTracks().forEach((t) => t.stop());
    }
    this.isRecording = false;
  }

  private toggleRecording(): void {
    if (this.isRecording) {
      this.stopRecording();
    } else {
      this.startRecording();
    }
  }

  private togglePhoneMode(): void {
    this.phoneMode = !this.phoneMode;
    const inputInner = this.inputField?.parentElement as HTMLElement | null;
    if (this.phoneMode) {
      this.phoneEmptyRetries = 0;
      if (!this.isRecording) {
        setTimeout(() => this.startRecording(), 100);
      }
      this.inputField?.classList.add('phone-mode-hidden');
      this.sendButton?.classList.add('phone-mode-hidden');
      this.stopButton?.classList.add('phone-mode-hidden');
      this.micButton?.classList.add('phone-mode-hidden');
      inputInner?.classList.add('phone-mode');
    } else {
      this.stopRecording();
      this.stopAllTTS();
      if (!this.webcamActive) {
        this.stopEmotionPolling();
        this.stopContinuousAudioEmotion();
      }
      this.awaitingUserSpeech = false;
      this.phoneEmptyRetries = 0;
      this.inputField?.classList.remove('phone-mode-hidden');
      this.sendButton?.classList.remove('phone-mode-hidden');
      if (this.sendButton) this.sendButton.style.display = '';
      this.stopButton?.classList.remove('phone-mode-hidden');
      this.micButton?.classList.remove('phone-mode-hidden');
      inputInner?.classList.remove('phone-mode');
    }
    this.updateEmotionButtonVisibility();
    this.updateMicButton();
    this.updatePhoneButton();
  }

  private updatePhoneButton(): void {
    if (!this.phoneButton) return;
    const svg = this.phoneButton.querySelector('svg');
    if (this.phoneMode) {
      this.phoneButton.classList.add('active');
      if (svg) {
        svg.innerHTML = '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/><line x1="2" y1="2" x2="22" y2="22"/>';
      }
    } else {
      this.phoneButton.classList.remove('active');
      if (svg) {
        svg.innerHTML = '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/>';
      }
    }
  }

  private updateEmotionButton(btn?: HTMLElement): void {
    const target = btn || this.emotionButton;
    if (!target) return;
    target.classList.toggle('active', this.webcamActive);
    target.classList.toggle('error', this.emotionRuntimeStatus === 'error');
    target.title = this.webcamActive ? 'Emotion Detection aktiv' : 'Emotion Detection starten';
  }

  private updateMicButton(): void {
    if (!this.micButton) return;
    this.micButton.style.display = this.audioCapabilities?.stt?.button_visible ? '' : 'none';
    if (this.isRecording) {
      this.micButton.classList.add('recording');
      this.micButton.setAttribute('title', 'Aufnahme stoppen');
    } else {
      this.micButton.classList.remove('recording');
      this.micButton.setAttribute('title', 'Spracheingabe');
    }
    if (this.awaitingUserSpeech) {
      this.micButton.classList.add('awaiting');
    } else {
      this.micButton.classList.remove('awaiting');
    }
  }

  /**
   * Return existing session conversation id or create a new one.
   * Widget IDs are prefixed with `widget-` per spec.
   */
  private getOrCreateSessionConversationId(): string {
    try {
      const key = 'agentix-conversation-id';
      const existing = sessionStorage.getItem(key);
      if (existing && existing.trim()) return existing;
      const id = `widget-${(crypto as any).randomUUID ? (crypto as any).randomUUID() : String(Date.now())}`;
      sessionStorage.setItem(key, id);
      return id;
    } catch (err) {
      // Fallback to timestamp-prefixed id if sessionStorage/crypto are unavailable
      return `widget-${String(Date.now())}`;
    }
  }

  /**
   * Inject CSS styles into the document
   */
  private injectStyles(): void {
    if (this.styleElement) return;

    this.styleElement = document.createElement('style');
    this.styleElement.textContent = WIDGET_STYLES;
    document.head.appendChild(this.styleElement);
  }

  /**
   * Apply custom color CSS variables to the widget container
   */
  private applyColorVariables(): void {
    if (!this.container) return;
    const c = this.config.colors;
    const map: Record<string, string | undefined> = {
      '--chat-primary': c.primary,
      '--chat-header-bg': c.headerBackground,
      '--chat-header-text': c.headerText,
      '--chat-bg': c.background,
      '--chat-text': c.text,
      '--chat-user-msg-bg': c.userMessageBackground,
      '--chat-user-msg-text': c.primary ? '#ffffff' : undefined,
      '--chat-assistant-msg-bg': c.assistantMessageBackground,
      '--chat-assistant-msg-text': c.text,
      '--chat-input-bg': c.inputBackground,
      '--chat-input-text': c.inputText,
      '--chat-input-border': c.primary,
      '--chat-send-btn-bg': c.buttonBackground,
      '--chat-send-btn-text': c.buttonText,
    };
    for (const [key, value] of Object.entries(map)) {
      if (value) {
        this.container.style.setProperty(key, value);
      }
    }
  }

  /**
   * Convert hex color to rgba string with given opacity
   */
  private hexToRgba(hex: string, opacity: number): string {
    const clean = hex.replace('#', '');
    let r: number, g: number, b: number;
    if (clean.length === 3) {
      r = parseInt(clean[0] + clean[0], 16);
      g = parseInt(clean[1] + clean[1], 16);
      b = parseInt(clean[2] + clean[2], 16);
    } else {
      r = parseInt(clean.substring(0, 2), 16);
      g = parseInt(clean.substring(2, 4), 16);
      b = parseInt(clean.substring(4, 6), 16);
    }
    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
  }

  /**
   * Create a logo image element optionally wrapped with border/glow/background
   */
  private createLogoElement(className: string, url: string): HTMLElement {
    const logoImg = createElement('img', {
      class: className,
      src: url,
      alt: 'Logo',
      onerror: 'this.style.display="none"'
    }) as HTMLImageElement;

    const border = this.config.logoBorder;
    const glow = this.config.logoGlow;
    const bg = this.config.logoBackground;
    const needsWrapper = border?.enabled || glow?.enabled || bg?.enabled;
    if (needsWrapper) {
      const wrapper = createElement('span', {
        class: 'agentix-chat-logo-wrapper',
      });
      wrapper.style.display = 'inline-flex';
      wrapper.style.padding = '2px';
      wrapper.style.borderRadius = '6px';
      if (bg?.enabled) {
        const bgColor = this.hexToRgba(bg.color || '#ffffff', bg.opacity ?? 0.35);
        wrapper.style.background = bgColor;
      }
      if (border?.enabled) {
        const borderColor = this.hexToRgba(border.color || '#ffffff', border.opacity ?? 0.35);
        wrapper.style.border = `1px solid ${borderColor}`;
      }
      if (glow?.enabled) {
        const glowColor = this.hexToRgba(glow.color || '#ffffff', glow.opacity ?? 0.4);
        wrapper.style.boxShadow = `0 0 6px ${glowColor}`;
      }
      wrapper.appendChild(logoImg);
      return wrapper;
    }

    return logoImg;
  }

  /**
   * Rebuild the header logo element. Removes ALL '.agentix-chat-header-logo' and
   * '.agentix-chat-logo-wrapper' nodes inside the header — previous updates used
   * to remove only the <img>, leaving the (white, glowing) wrapper behind,
   * which rendered as a stray dot between the new logo and the title.
   */
  private refreshHeaderLogo(): void {
    const header = this.container?.querySelector('.agentix-chat-header');
    if (!header) return;
    header
      .querySelectorAll('.agentix-chat-header-logo, .agentix-chat-logo-wrapper')
      .forEach((el) => el.remove());
    if (this.config.logoUrlHeader && this.config.showLogoInHeader) {
      const logoEl = this.createLogoElement('agentix-chat-header-logo', this.config.logoUrlHeader);
      header.insertBefore(logoEl, header.firstChild);
    }
  }

  /**
   * Rebuild the toggle-button content (logo element or 💬 fallback).
   */
  private refreshToggleButtonLogo(): void {
    if (!this.toggleButton) return;
    this.toggleButton.innerHTML = '';
    if (this.config.logoUrlButton && this.config.showLogoInButton) {
      const logoEl = this.createLogoElement('', this.config.logoUrlButton);
      const img = logoEl.querySelector('img');
      if (img) {
        img.onerror = function(this: HTMLImageElement) {
          this.style.display = 'none';
          const parent = this.closest('.agentix-chat-logo-wrapper');
          if (parent) {
            parent.textContent = '💬';
          }
        } as any;
      }
      this.toggleButton.appendChild(logoEl);
    } else {
      this.toggleButton.appendChild(document.createTextNode('💬'));
    }
  }

  /**
   * Create DOM structure
   */
  private createDOM(): void {
    // Main container
    this.container = createElement('div', {
      class: `agentix-chat-widget-container ${this.config.position} ${this.config.theme}`,
      id: 'agentix-chat-widget'
    });

    // Toggle button
    this.toggleButton = createElement('button', {
      class: 'agentix-chat-toggle-btn',
      id: 'agentix-chat-toggle',
      'aria-label': 'Open chat',
      'aria-expanded': 'false'
    });
    this.refreshToggleButtonLogo();

    this.container.appendChild(this.toggleButton);

    // Chat window
    this.window = createElement('div', {
      class: `agentix-chat-window ${this.config.position}`,
      id: 'agentix-chat-window',
      role: 'dialog',
      'aria-labelledby': 'agentix-chat-title'
    });

    this.window.style.width = `${this.config.width}px`;
    this.window.style.height = `${this.config.height}px`;

    // Header
    const header = createElement('div', {
      class: 'agentix-chat-header'
    });

    if (this.config.logoUrlHeader && this.config.showLogoInHeader) {
      const logoEl = this.createLogoElement('agentix-chat-header-logo', this.config.logoUrlHeader);
      header.appendChild(logoEl);
    }

    const headerInfo = createElement('div', {
      class: 'agentix-chat-header-info'
    });

    const title = createElement('h2', {
      class: 'agentix-chat-header-title',
      id: 'agentix-chat-title'
    }, this.config.title);

    const subtitle = createElement('p', {
      class: 'agentix-chat-header-subtitle'
    }, this.config.subtitle);

    headerInfo.appendChild(title);
    headerInfo.appendChild(subtitle);

    const closeBtn = createElement('button', {
      class: 'agentix-chat-close-btn',
      'aria-label': 'Close chat'
    }, '✕');

    header.appendChild(headerInfo);
    header.appendChild(closeBtn);

    // Messages container
    this.messagesContainer = createElement('div', {
      class: 'agentix-chat-messages',
      role: 'log',
      'aria-live': 'polite'
    });
    // Smart scroll: detect if user scrolled up
    this.messagesContainer.addEventListener('scroll', () => {
      const nearBottom = this.messagesContainer!.scrollHeight - this.messagesContainer!.scrollTop - this.messagesContainer!.clientHeight < 100;
      this.userScrolledUpRef = !nearBottom;
    });

    this.showEmptyState();

    // Input area (floating bar like dashboard)
    const inputBar = createElement('div', {
      class: 'agentix-chat-input-bar'
    });
    const inputInner = createElement('div', {
      class: 'agentix-chat-input-inner'
    });

    this.inputField = createElement('textarea', {
      class: 'agentix-chat-input-field',
      placeholder: this.config.inputPlaceholder,
      'aria-label': 'Message input',
      rows: '1'
    }) as HTMLTextAreaElement;

    // Phone mode button
    const phoneButton = createElement('button', {
      class: 'agentix-chat-phone-btn',
      'aria-label': 'Phone mode',
      style: 'display:none',
    });
    const phoneSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    phoneSvg.setAttribute('width', '20');
    phoneSvg.setAttribute('height', '20');
    phoneSvg.setAttribute('viewBox', '0 0 24 24');
    phoneSvg.setAttribute('fill', 'none');
    phoneSvg.setAttribute('stroke', 'currentColor');
    phoneSvg.setAttribute('stroke-width', '2');
    phoneSvg.setAttribute('stroke-linecap', 'round');
    phoneSvg.setAttribute('stroke-linejoin', 'round');
    phoneSvg.innerHTML = '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/>';
    phoneButton.appendChild(phoneSvg);
    this.phoneButton = phoneButton;
    phoneButton.addEventListener('click', () => {
      this.togglePhoneMode();
    });

    this.micButton = createElement('button', {
      class: 'agentix-chat-mic-btn',
      'aria-label': 'Voice input',
      style: 'display:none',
    });
    const micSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    micSvg.setAttribute('width', '20');
    micSvg.setAttribute('height', '20');
    micSvg.setAttribute('viewBox', '0 0 24 24');
    micSvg.setAttribute('fill', 'none');
    micSvg.setAttribute('stroke', 'currentColor');
    micSvg.setAttribute('stroke-width', '2');
    micSvg.setAttribute('stroke-linecap', 'round');
    micSvg.setAttribute('stroke-linejoin', 'round');
    micSvg.innerHTML = '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="23"/><line x1="8" x2="16" y1="23" y2="23"/>';
    this.micButton.appendChild(micSvg);
    this.micButton.addEventListener('click', () => this.toggleRecording());

    // Emotion webcam toggle button
    const emotionButton = createElement('button', {
      class: 'agentix-chat-emotion-btn',
      'aria-label': 'Emotion Detection',
      style: 'display:none',
    });
    const emotionSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    emotionSvg.setAttribute('width', '20');
    emotionSvg.setAttribute('height', '20');
    emotionSvg.setAttribute('viewBox', '0 0 24 24');
    emotionSvg.setAttribute('fill', 'none');
    emotionSvg.setAttribute('stroke', 'currentColor');
    emotionSvg.setAttribute('stroke-width', '2');
    emotionSvg.setAttribute('stroke-linecap', 'round');
    emotionSvg.setAttribute('stroke-linejoin', 'round');
    emotionSvg.innerHTML = '<circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>';
    emotionButton.appendChild(emotionSvg);
    emotionButton.addEventListener('click', () => {
      if (this.webcamActive) {
        this.stopWebcamEmotion();
      } else {
        this.startWebcamEmotion();
      }
      this.updateEmotionButton(emotionButton);
    });

    // Stop button (shown during streaming)
    this.stopButton = createElement('button', {
      class: 'agentix-chat-stop-btn',
      'aria-label': 'Stop generation',
      style: 'display:none',
    });
    const stopPulse = createElement('span', { class: 'agentix-chat-stop-pulse' });
    const stopIcon = createElement('span', { class: 'agentix-chat-stop-icon' });
    const stopSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    stopSvg.setAttribute('width', '14');
    stopSvg.setAttribute('height', '14');
    stopSvg.setAttribute('viewBox', '0 0 24 24');
    stopSvg.setAttribute('fill', 'currentColor');
    stopSvg.setAttribute('stroke', 'none');
    stopSvg.innerHTML = '<rect x="6" y="6" width="12" height="12" rx="2" />';
    stopIcon.appendChild(stopSvg);
    this.stopButton.appendChild(stopPulse);
    this.stopButton.appendChild(stopIcon);
    this.stopButton.addEventListener('click', () => {
      this.abortController?.abort();
    });

    // Send button
    this.sendButton = createElement('button', {
      class: 'agentix-chat-send-btn',
      'aria-label': 'Send message'
    });
    const sendSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    sendSvg.setAttribute('width', '20');
    sendSvg.setAttribute('height', '20');
    sendSvg.setAttribute('viewBox', '0 0 24 24');
    sendSvg.setAttribute('fill', 'none');
    sendSvg.setAttribute('stroke', 'currentColor');
    sendSvg.setAttribute('stroke-width', '2');
    sendSvg.setAttribute('stroke-linecap', 'round');
    sendSvg.setAttribute('stroke-linejoin', 'round');
    sendSvg.innerHTML = '<line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>';
    this.sendButton.appendChild(sendSvg);

    inputInner.appendChild(this.inputField);
    inputInner.appendChild(phoneButton);
    inputInner.appendChild(this.micButton);
    inputInner.appendChild(emotionButton);
    inputInner.appendChild(this.stopButton);
    inputInner.appendChild(this.sendButton);

    // Store reference for visibility updates
    this.emotionButton = emotionButton;
    inputBar.appendChild(inputInner);

    // Resize handle
    const resizeHandle = createElement('div', {
      class: 'agentix-chat-resize-handle',
      'aria-label': 'Resize chat window'
    });

    // Assemble window
    this.window.appendChild(header);
    this.window.appendChild(this.messagesContainer);
    this.window.appendChild(inputBar);
    this.window.appendChild(resizeHandle);

    this.container.appendChild(this.window);
    document.body.appendChild(this.container);

    // Create privacy modal
    this.createPrivacyModal();
  }

  /**
   * Create privacy consent modal
   */
  private createPrivacyModal(): void {
    this.privacyModal = createElement('div', {
      class: 'agentix-chat-privacy-modal',
      id: 'agentix-chat-privacy-modal',
      role: 'dialog',
      'aria-labelledby': 'agentix-chat-privacy-title'
    });

    const modalContent = createElement('div', {
      class: 'agentix-chat-privacy-content'
    });

    const title = createElement('h3', {
      class: 'agentix-chat-privacy-title',
      id: 'agentix-chat-privacy-title'
    }, WIDGET_I18N[this.config.language].privacyTitle);

    const text = createElement('p', {
      class: 'agentix-chat-privacy-text'
    }, this.config.privacyText);

    const buttonContainer = createElement('div', {
      class: 'agentix-chat-privacy-buttons'
    });

    const acceptBtn = createElement('button', {
      class: 'agentix-chat-privacy-accept',
      id: 'agentix-chat-privacy-accept'
    }, WIDGET_I18N[this.config.language].privacyAccept);

    const declineBtn = createElement('button', {
      class: 'agentix-chat-privacy-decline',
      id: 'agentix-chat-privacy-decline'
    }, WIDGET_I18N[this.config.language].privacyDecline);

    buttonContainer.appendChild(acceptBtn);
    buttonContainer.appendChild(declineBtn);

    modalContent.appendChild(title);
    modalContent.appendChild(text);
    modalContent.appendChild(buttonContainer);

    this.privacyModal.appendChild(modalContent);
    // Append to window instead of container so it overlays inside the chat
    this.window?.appendChild(this.privacyModal);

    // Add event listeners
    addEvent(acceptBtn, 'click', () => this.acceptPrivacy());
    addEvent(declineBtn, 'click', () => this.declinePrivacy());
  }

  /**
   * Accept privacy policy
   */
  private acceptPrivacy(): void {
    this.privacyAccepted = true;
    localStorage.setItem('agentix-chat-privacy-accepted', 'true');
    this.hidePrivacyModal();
    this.enableChat();
    
    // Trigger onOpen callback if window is open
    if (this.isOpen) {
      this.config.onOpen();
      this.sendInitialMessageIfNeeded();
    }
  }

  /**
   * Send initial greeting message if conditions are met
   */
  private sendInitialMessageIfNeeded(): void {
    if (!this.config.initialMessage || this.messages.length > 0 || !this.privacyAccepted) return;
    this.sendUserMessage(this.config.initialMessageContent, { silent: this.config.initialMessageSilent });
  }

  /**
   * Decline privacy policy
   */
  private declinePrivacy(): void {
    this.privacyAccepted = false;
    localStorage.setItem('agentix-chat-privacy-accepted', 'false');
    this.hidePrivacyModal();
    this.disableChat();
  }

  /**
   * Hide privacy modal
   */
  private hidePrivacyModal(): void {
    if (this.privacyModal) {
      this.privacyModal.style.display = 'none';
    }
  }

  /**
   * Enable chat input
   */
  private enableChat(): void {
    if (this.inputField) {
      this.inputField.disabled = false;
      this.inputField.placeholder = this.config.inputPlaceholder;
    }
    if (this.sendButton) {
      this.sendButton.removeAttribute('disabled');
    }
  }

  /**
   * Disable chat input
   */
  private disableChat(): void {
    if (this.inputField) {
      this.inputField.disabled = true;
      this.inputField.placeholder = WIDGET_I18N[this.config.language].privacyDisabledPlaceholder;
    }
    if (this.sendButton) {
      this.sendButton.setAttribute('disabled', '');
    }
  }

  /**
   * Show empty state
   */
  private showEmptyState(): void {
    if (!this.messagesContainer) return;

    this.messagesContainer.innerHTML = `
      <div class="agentix-chat-empty-state">
        <div class="agentix-chat-empty-state-icon">👋</div>
        <div class="agentix-chat-empty-state-text">
          ${WIDGET_I18N[this.config.language].emptyStateText}
        </div>
      </div>
    `;
  }

  /**
   * Attach event listeners
   */
  private attachEventListeners(): void {
    if (!this.toggleButton || !this.window || !this.inputField || !this.sendButton) return;

    // Toggle button
    addEvent(this.toggleButton, 'click', () => this.toggle());

    // Close button
    const closeBtn = this.window.querySelector('.agentix-chat-close-btn') as HTMLElement;
    if (closeBtn) {
      addEvent(closeBtn, 'click', () => this.close());
    }

    // Send button
    addEvent(this.sendButton, 'click', () => this.sendMessage());

    // Input field enter key
    addEvent(this.inputField, 'keydown', (e: Event) => {
      const keyEvent = e as KeyboardEvent;
      if (keyEvent.key === 'Enter' && !keyEvent.shiftKey) {
        keyEvent.preventDefault();
        this.sendMessage();
      }
    });

    // Auto-resize textarea
    addEvent(this.inputField, 'input', () => this.autoResizeTextarea());

    // Resize functionality
    this.setupResizeHandle();

    // Clamp widget size to viewport when browser window resizes
    this.windowResizeHandler = () => {
      if (!this.window) return;
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const currentWidth = this.window.offsetWidth;
      const currentHeight = this.window.offsetHeight;
      if (currentWidth > vw) {
        this.window.style.width = vw + 'px';
      }
      if (currentHeight > vh) {
        this.window.style.height = vh + 'px';
      }
    };
    window.addEventListener('resize', this.windowResizeHandler);
  }

  /**
   * Setup resize handle for the chat window
   */
  private setupResizeHandle(): void {
    if (!this.window) return;

    const resizeHandle = this.window.querySelector('.agentix-chat-resize-handle') as HTMLElement;
    if (!resizeHandle) return;
    // Use Pointer events + rAF to batch DOM writes and avoid layout thrashing
    let startX = 0;
    let startY = 0;
    let startWidth = 0;
    let startHeight = 0;
    let isPointerDown = false;
    let lastX = 0;
    let lastY = 0;
    let frameRequested = false;

    const minWidth = 300;
    const minHeight = 400;
    let maxWidth = window.innerWidth;
    let maxHeight = window.innerHeight;

    const applySize = () => {
      if (!this.window) return;
      const deltaX = lastX - startX;
      const deltaY = lastY - startY;

      // top-left handle: dragging right/down should decrease size
      let newWidth = startWidth - deltaX;
      let newHeight = startHeight - deltaY;

      newWidth = Math.min(Math.max(newWidth, minWidth), maxWidth);
      newHeight = Math.min(Math.max(newHeight, minHeight), maxHeight);

      // Batch DOM writes
      this.window.style.width = newWidth + 'px';
      this.window.style.height = newHeight + 'px';
      frameRequested = false;
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!isPointerDown) return;
      lastX = e.clientX;
      lastY = e.clientY;
      if (!frameRequested) {
        frameRequested = true;
        requestAnimationFrame(applySize);
      }
    };

    const onPointerUp = (e: PointerEvent) => {
      isPointerDown = false;
      this.window?.classList.remove('resizing');
      document.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerup', onPointerUp);
      (e.target as Element).releasePointerCapture?.(e.pointerId);
    };

    const onPointerDown = (e: PointerEvent) => {
      // Only handle primary button
      if (e.button && e.button !== 0) return;
      isPointerDown = true;
      startX = e.clientX;
      startY = e.clientY;
      lastX = startX;
      lastY = startY;
      startWidth = this.window?.offsetWidth || this.config.width;
      startHeight = this.window?.offsetHeight || this.config.height;
      maxWidth = window.innerWidth;
      maxHeight = window.innerHeight;
      this.window?.classList.add('resizing');

      // Capture pointer to continue receiving events
      try {
        (e.target as Element).setPointerCapture?.(e.pointerId);
      } catch (err) {
        // ignore
      }

      document.addEventListener('pointermove', onPointerMove);
      document.addEventListener('pointerup', onPointerUp);
      e.preventDefault();
    };

    resizeHandle.addEventListener('pointerdown', onPointerDown as EventListener, { passive: false });
  }

  /**
   * Show a temporary toast message inside the chat window
   */
  private _showToast(message: string, duration: number = 2500): void {
    const parent = this.inputField?.parentElement;
    if (!parent) return;
    parent.style.position = 'relative';
    const toast = document.createElement('div');
    toast.className = 'agentix-chat-toast';
    toast.textContent = message;
    toast.style.cssText = `
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      background: rgba(220,53,69,0.6);
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 13px;
      border-radius: inherit;
      z-index: 2;
      pointer-events: none;
      text-align: center;
      padding: 0 8px;
      transform: translateX(100%);
      opacity: 0;
      transition: transform 0.4s ease-out, opacity 0.4s ease-out;
    `;
    parent.appendChild(toast);
    requestAnimationFrame(() => {
      toast.style.transform = 'translateX(0)';
      toast.style.opacity = '1';
    });
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  /**
   * Auto-resize textarea based on content
   */
  private autoResizeTextarea(): void {
    if (!this.inputField) return;

    this.inputField.style.height = 'auto';
    const scrollHeight = this.inputField.scrollHeight;
    const newHeight = Math.min(scrollHeight, 100);
    this.inputField.style.height = newHeight + 'px';
    
    // Hide scrollbar when content fits within 4 lines
    if (scrollHeight <= 100) {
      this.inputField.style.overflowY = 'hidden';
    } else {
      this.inputField.style.overflowY = 'auto';
    }
  }

  /**
   * Toggle chat window
   */
  toggle(): void {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }

  /**
   * Open chat window
   */
  open(): void {
    if (!this.window || !this.toggleButton) return;

    this.isOpen = true;
    this.window.classList.add('open');
    this.toggleButton.setAttribute('aria-expanded', 'true');
    this.unreadCount = 0;
    this.updateBadge();
    localStorage.setItem('agentix-chat-open', 'true');
    this.stopBounce();

    // Show privacy modal if not accepted, otherwise focus input and trigger callback
    if (!this.privacyAccepted) {
      if (this.privacyModal) {
        this.privacyModal.style.display = 'flex';
      }
      return;
    }

    this.inputField?.focus();
    this.config.onOpen();
    this.sendInitialMessageIfNeeded();
  }

  /**
   * Close chat window
   */
  close(): void {
    if (!this.window || !this.toggleButton) return;

    this.isOpen = false;
    this.window.classList.remove('open');
    this.toggleButton.setAttribute('aria-expanded', 'false');
    localStorage.setItem('agentix-chat-open', 'false');
    this.startBounce();
  }

  /**
   * Start bounce animation and speech bubble on toggle button
   */
  private startBounce(): void {
    if (!this.toggleButton || this.bounceTimer !== null) return;
    const interval = this.config.bounceInterval || 10000;
    if (interval <= 0) return;

    this.bounceTimer = window.setInterval(() => {
      if (!this.toggleButton || this.isOpen) return;
      this.toggleButton.classList.add('bouncing');
      if (!this.bubbleElement) {
        this.bubbleElement = createElement('div', { class: 'agentix-chat-bubble' });
        this.toggleButton.appendChild(this.bubbleElement);
      }
      this.bubbleElement.textContent = this.config.bounceText || WIDGET_I18N[this.config.language].bounceText;
      this.bubbleElement.classList.add('visible');
      window.setTimeout(() => {
        this.toggleButton?.classList.remove('bouncing');
        this.bubbleElement?.classList.remove('visible');
      }, 3000);
    }, interval);
  }

  /**
   * Stop bounce animation and speech bubble
   */
  private stopBounce(): void {
    if (this.bounceTimer !== null) {
      clearInterval(this.bounceTimer);
      this.bounceTimer = null;
    }
    if (this.toggleButton) {
      this.toggleButton.classList.remove('bouncing');
    }
    if (this.bubbleElement) {
      this.bubbleElement.classList.remove('visible');
    }
  }

  /**
   * Public method to send a message programmatically
   */
  public async sendUserMessage(content: string, options?: { silent?: boolean }): Promise<void> {
    if (!this.messagesContainer || !this.apiClient || this.isLoading) return;

    const sanitizedContent = sanitizeInput(content);
    if (!sanitizedContent.trim()) return;

    // Temporarily set the input value and send
    const originalValue = this.inputField?.value || '';
    if (this.inputField) {
      this.inputField.value = sanitizedContent;
    }
    await this.sendMessage(options?.silent);
    // Restore original value if needed
    if (this.inputField && originalValue) {
      this.inputField.value = originalValue;
    }
  }

  /**
   * Extract a user-friendly German error message from an API error
   */
  private getUserFriendlyErrorMessage(error: Error): string {
    const msg = error.message;
    const statusMatch = msg.match(/API Error \((\d+)\):/);
    const status = statusMatch ? parseInt(statusMatch[1], 10) : null;
    const i18n = WIDGET_I18N[this.config.language];

    switch (status) {
      case 429:
        return i18n.error429;
      case 401:
        return i18n.error401;
      case 403:
        return i18n.error403;
      case 404:
        return i18n.error404;
      case 500:
      case 502:
      case 503:
      case 504:
        return i18n.error500;
      default:
        return i18n.errorDefault;
    }
  }

  /**
   * Send message to API
   */
  private async fetchTTSSegment(text: string, voiceId: string): Promise<Blob> {
    const baseUrl = this.config.apiUrl.replace(/\/v1$/, '');
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
      const res = await fetch(`${baseUrl}/audio/tts/segment`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, voice_id: voiceId }),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(`TTS segment failed: ${res.status}`);
      return res.blob();
    } finally {
      clearTimeout(timeout);
    }
  }

  /**
   * Determine whether sentence-level TTS streaming should be active.
   */
  private shouldStreamTTS(): boolean {
    if (!this.audioCapabilities?.tts?.button_visible || !this.audioCapabilities?.tts?.voice_id) return false;
    // Phone mode always streams when TTS is available
    if (this.phoneMode) return true;
    // Chat auto-read only streams when streaming is explicitly enabled
    if (this.audioCapabilities.tts.auto_read && this.audioCapabilities.tts.streaming_enabled) return true;
    return false;
  }

  private async sendMessage(
    silent: boolean = false,
    text?: string,
    audioEmotions?: { audio_prosody?: EmotionContextPayload['audio_prosody'] | null },
  ): Promise<void> {
    if (!this.messagesContainer || !this.apiClient) return;
    if (!text && (!this.inputField || this.isLoading)) return;

    const userInput = text ? sanitizeInput(text) : sanitizeInput(this.inputField!.value);

    if (!userInput.trim() && !audioEmotions?.audio_prosody) return;

    // Abort any previous streaming TTS
    this.stopAllTTS();

    // Add user message (only if not silent)
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: userInput,
      timestamp: Date.now()
    };

    if (!silent) {
      this.messages.push(userMessage);
      this.renderMessage(userMessage);
    }

    // Clear input
    if (this.inputField) {
      this.inputField.value = '';
      this.autoResizeTextarea();
    }

    // Disable input
    this.isLoading = true;
    if (this.inputField) {
      this.inputField.disabled = true;
    }
    this.sendButton?.setAttribute('disabled', '');
    if (this.sendButton) this.sendButton.style.display = 'none';
    if (this.stopButton) this.stopButton.style.display = this.phoneMode ? 'none' : '';

    // Clear empty state if present
    const emptyState = this.messagesContainer.querySelector('.agentix-chat-empty-state');
    if (emptyState) {
      emptyState.remove();
    }

    // Add assistant streaming message
    const assistantMessage: ChatMessage = {
      id: `msg-${Date.now()}-streaming`,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      isStreaming: true
    };

    this.messages.push(assistantMessage);
    const messageElement = this.renderMessage(assistantMessage);

    try {
      this.abortController = new AbortController();

      // Build API messages - only include user messages and previous assistant messages
      // The last message must be from user for Agentix API
      let apiMessages = this.messages
        .filter(m => m.role !== 'system')
        .filter(m => !(m.role === 'assistant' && m.isStreaming && m.content === ''))
        .map(m => ({
          role: m.role,
          content: m.content
        }));

      // If silent message, add the user input to API messages
      if (silent) {
        apiMessages.push({ role: 'user', content: userInput });
      }

      // Stop continuous emotion polling and flush the last batch
      this.stopEmotionPolling();
      const phases = this.buildEmotionPhases();
      const fused = this.computeOverallFused(phases);

      let fullContent = '';
      let streamingContentElement: HTMLElement | null = null;

      const streamTTS = this.shouldStreamTTS();
      if (streamTTS) {
        this.ttsSegmentPlayer = new TTSSegmentPlayer();
      }

      const voiceId = this.audioCapabilities?.tts?.voice_id || null;

      // Capture pre-send emotion snapshot
      const preSendEmotion = await this.capturePreSendEmotion();
      const emotionPayload = this.buildEmotionPayload();
      if (emotionPayload && preSendEmotion) {
        emotionPayload.visual_pre_send = preSendEmotion;
      }
      if (emotionPayload && audioEmotions?.audio_prosody) {
        emotionPayload.audio_prosody = audioEmotions.audio_prosody;
        if (emotionPayload.consent) {
          emotionPayload.consent.audio_opt_in = true;
        }
      }
      if (emotionPayload) {
        if (fused) emotionPayload.fused = fused;
        if (phases.length > 0) emotionPayload.phases = phases;
        // Extract best audio prosody from continuous audio snapshots
        if (this.audioSnapshotsRef.length > 0) {
          const bestAudio = this.audioSnapshotsRef.reduce((best, curr) =>
            curr.confidence > best.confidence ? curr : best
          );
          emotionPayload.audio_prosody = bestAudio;
          this.audioSnapshotsRef = [];
        }
      }

      let backendEmotion: any = null;
      for await (const chunk of this.apiClient.streamChat(apiMessages, emotionPayload)) {
        if (chunk.emotion) {
          backendEmotion = chunk.emotion;
        }
        if (chunk.content) {
          fullContent += chunk.content;
          assistantMessage.content = fullContent;

          // Phase transition: content arriving means assistant is typing
          if (!this.hasTransitionedToTypingRef) {
            this.switchEmotionPhase('typing');
            this.hasTransitionedToTypingRef = true;
          }

          // Sentence-level TTS streaming
          if (streamTTS && voiceId) {
            this.ttsSentenceBuffer += chunk.content;
            const { chunks, remainder } = extractCompleteSentences(this.ttsSentenceBuffer);
            this.ttsSentenceBuffer = remainder;
            for (const chunk of chunks) {
              const clean = this.cleanTextForTTS(chunk.text);
              if (clean) {
                this.ttsSegmentPlayer!.add(clean, voiceId, this.fetchTTSSegment.bind(this), chunk.pauseAfterMs);
              }
            }
          }
        }
        if (chunk.toolExecution) {
          assistantMessage.terminalEvents = [...(assistantMessage.terminalEvents || []), chunk.toolExecution];
        }

        const contentElement = messageElement?.querySelector('.agentix-chat-message-content');
        if (contentElement) {
          if (!streamingContentElement) {
            streamingContentElement = createElement('div', { class: 'agentix-chat-streaming-content' });
            contentElement.appendChild(streamingContentElement);
          }
          streamingContentElement.innerHTML = renderMarkdown(fullContent);
        }

        this.smartScrollToBottom();
      }

      // Flush any remaining buffered text as a final segment
      if (streamTTS && voiceId && this.ttsSentenceBuffer.trim()) {
        const clean = this.cleanTextForTTS(this.ttsSentenceBuffer.trim());
        if (clean) {
          this.ttsSegmentPlayer!.add(clean, voiceId, this.fetchTTSSegment.bind(this), 0);
        }
        this.ttsSentenceBuffer = '';
      }

      if (backendEmotion) {
        userMessage.emotion = backendEmotion;
      }
      assistantMessage.isStreaming = false;
      if (messageElement) {
        this.addSpeakButton(messageElement, assistantMessage);
      }
      this.saveMessages();
      if (this.webcamActive || this.phoneMode) {
        this.startEmotionPolling();
      }

      // Fade out and remove thinking animation
      if (messageElement) {
        const thinkingWrapper = messageElement.querySelector('.agentix-chat-thinking');
        if (thinkingWrapper) {
          thinkingWrapper.classList.add('fade-out');
          setTimeout(() => {
            if (thinkingWrapper.parentNode) {
              thinkingWrapper.parentNode.removeChild(thinkingWrapper);
            }
          }, 300);
        }
      }

      // Post-process HTML/CSS/JS code blocks after streaming completes
      if (messageElement) {
        const contentEl = messageElement.querySelector('.agentix-chat-message-content') as HTMLElement;
        if (contentEl) {
          this.postProcessHtmlBlocks(contentEl, assistantMessage.content);
        }
      }

      // In phone mode: after assistant response, play TTS then start STT
      if (this.phoneMode && fullContent.trim() && messageElement) {
        // If sentence-level streaming is active (segments queued or already playing),
        // wait for the queue to drain before transitioning back to STT.
        // Never fall back to speakText() when streaming segments exist.
        if (this.ttsSegmentPlayer) {
          const waitForPlayer = () => {
            console.info('[Widget] waitForPlayer poll, queueLength=', this.ttsSegmentPlayer?.queueLength);
            if (this.ttsSegmentPlayer && this.ttsSegmentPlayer.queueLength > 0) {
              setTimeout(waitForPlayer, 200);
            } else {
              console.info('[Widget] waitForPlayer done, restarting recording');
              if (this.phoneMode) {
                this.awaitingUserSpeech = true;
                this.updateMicButton();
                setTimeout(() => this.startRecording(), 300);
              }
            }
          };
          setTimeout(waitForPlayer, 200);
        } else {
          // No streaming player -> fall back to full-response TTS
          setTimeout(() => this.speakText(fullContent.trim(), messageElement), 200);
        }
      }

    } catch (error) {
      this.stopAllTTS();
      // Remove the streaming placeholder from messages and DOM
      const assistantIndex = this.messages.findIndex(m => m.id === assistantMessage.id);
      if (assistantIndex !== -1) {
        this.messages.splice(assistantIndex, 1);
      }
      if (messageElement && messageElement.parentNode) {
        messageElement.parentNode.removeChild(messageElement);
      }

      const friendlyMessage = this.getUserFriendlyErrorMessage(
        error instanceof Error ? error : new Error(String(error))
      );

      // Render a styled error message
      const errorMsgElement = createElement('div', {
        class: 'agentix-chat-error-message'
      });
      const errorIcon = createElement('span', { class: 'agentix-chat-error-icon' });
      errorIcon.textContent = '⚠️';
      const errorText = createElement('span');
      errorText.textContent = friendlyMessage;
      errorMsgElement.appendChild(errorIcon);
      errorMsgElement.appendChild(errorText);
      this.messagesContainer.appendChild(errorMsgElement);
      this.smartScrollToBottom();

      this.config.onError(error instanceof Error ? error : new Error(String(error)));
    } finally {
      this.isLoading = false;
      if (this.inputField) {
        this.inputField.disabled = false;
      }
      this.sendButton?.removeAttribute('disabled');
      if (this.sendButton) this.sendButton.style.display = this.phoneMode ? 'none' : '';
      if (this.stopButton) this.stopButton.style.display = 'none';
      this.smartScrollToBottom();
      if (!this.phoneMode && this.inputField) {
        this.inputField.focus();
      }
      // Clean up TTS player after a grace period so lingering audio
      // doesn't leak between conversations.
      if (this.ttsSegmentPlayer) {
        const player = this.ttsSegmentPlayer;
        setTimeout(() => {
          if (player.queueLength === 0) {
            player.abort();
          }
        }, 5000);
      }
    }
  }

  /**
   * Add a speak button to a completed assistant message element if not already present.
   */
  private addSpeakButton(messageElement: HTMLElement, message: ChatMessage): void {
    if (message.role !== 'assistant' || message.isStreaming || !message.content) return;
    if (messageElement.querySelector('.agentix-chat-message-actions')) return;

    const actionsWrapper = createElement('div', {
      class: 'agentix-chat-message-actions',
      style: this.audioCapabilities?.tts?.button_visible ? '' : 'display:none',
    });
    const speakBtn = createElement('button', {
      class: 'agentix-chat-speak-btn',
      'aria-label': 'Vorlesen',
      title: 'Vorlesen',
    });
    const ring1 = createElement('span', { class: 'speak-ring' });
    const ring2 = createElement('span', { class: 'speak-ring' });
    const ring3 = createElement('span', { class: 'speak-ring' });
    speakBtn.appendChild(ring1);
    speakBtn.appendChild(ring2);
    speakBtn.appendChild(ring3);
    // Play icon (speaker)
    const speakSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    speakSvg.setAttribute('class', 'speak-icon');
    speakSvg.setAttribute('width', '16');
    speakSvg.setAttribute('height', '16');
    speakSvg.setAttribute('viewBox', '0 0 24 24');
    speakSvg.setAttribute('fill', 'none');
    speakSvg.setAttribute('stroke', 'currentColor');
    speakSvg.setAttribute('stroke-width', '2');
    speakSvg.setAttribute('stroke-linecap', 'round');
    speakSvg.setAttribute('stroke-linejoin', 'round');
    speakSvg.innerHTML = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>';
    speakBtn.appendChild(speakSvg);
    // Stop icon (square) – hidden by default
    const stopSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    stopSvg.setAttribute('class', 'speak-stop-icon');
    stopSvg.setAttribute('width', '16');
    stopSvg.setAttribute('height', '16');
    stopSvg.setAttribute('viewBox', '0 0 24 24');
    stopSvg.setAttribute('fill', 'currentColor');
    stopSvg.setAttribute('stroke', 'none');
    stopSvg.innerHTML = '<rect x="6" y="6" width="12" height="12" rx="2"/>';
    stopSvg.style.display = 'none';
    speakBtn.appendChild(stopSvg);
    speakBtn.addEventListener('click', () => {
      this.speakText(message.content, messageElement, speakBtn);
    });
    actionsWrapper.appendChild(speakBtn);
    messageElement.appendChild(actionsWrapper);
  }

  /**
   * Render message in DOM
   */
  private renderMessage(message: ChatMessage): HTMLElement | null {
    if (!this.messagesContainer) return null;

    const messageElement = createElement('div', {
      class: `agentix-chat-message ${message.role} ${message.isStreaming ? 'streaming' : ''}`,
      'data-id': message.id
    });

    const contentElement = createElement('div', {
      class: 'agentix-chat-message-content'
    });

    if (message.role === 'user') {
      contentElement.textContent = message.content;
    } else if (message.content) {
      contentElement.innerHTML = renderMarkdown(message.content);
      if (!message.isStreaming) {
        this.postProcessHtmlBlocks(contentElement, message.content || '');
      }
    } else {
      const thinkingWrapper = createElement('div', { class: 'agentix-chat-thinking' });
      const thinkingLabel = createElement('span', { class: 'agentix-chat-thinking-label' });
      thinkingLabel.textContent = 'Thinking';
      const thinkingDots = createElement('div', { class: 'agentix-chat-thinking-dots' });
      thinkingDots.appendChild(createElement('span'));
      thinkingDots.appendChild(createElement('span'));
      thinkingDots.appendChild(createElement('span'));
      thinkingWrapper.appendChild(thinkingLabel);
      thinkingWrapper.appendChild(thinkingDots);
      contentElement.appendChild(thinkingWrapper);
    }

    messageElement.appendChild(contentElement);

    this.addSpeakButton(messageElement, message);

    this.messagesContainer.appendChild(messageElement);
    this.smartScrollToBottom();

    return messageElement;
  }

  /**
   * Show HTML preview fullscreen modal
   */
  private showHtmlPreviewModal(html: string): void {
    if (!this.window) return;
    const overlay = createElement('div', { class: 'agentix-chat-modal-overlay' });
    overlay.addEventListener('click', () => {
      overlay.remove();
      this.previewHtml = null;
    });
    const box = createElement('div', { class: 'agentix-chat-modal-box preview-modal' });
    box.addEventListener('click', (e) => e.stopPropagation());
    const header = createElement('div', { class: 'agentix-chat-modal-header' });
    const title = createElement('h4', {}, 'HTML Preview');
    const closeBtn = createElement('button', { class: 'agentix-chat-modal-close' }, '✕');
    closeBtn.addEventListener('click', () => {
      overlay.remove();
      this.previewHtml = null;
    });
    header.appendChild(title);
    header.appendChild(closeBtn);
    box.appendChild(header);

    const content = createElement('div', { class: 'agentix-chat-preview-content' });
    const iframe = createElement('iframe', {
      class: 'agentix-chat-preview-frame',
      sandbox: 'allow-scripts',
      srcdoc: html,
      title: 'HTML preview',
    }) as HTMLIFrameElement;
    content.appendChild(iframe);
    box.appendChild(content);

    overlay.appendChild(box);
    this.window.appendChild(overlay);
    this.modalOverlay = overlay;
  }

  /**
   * Post-process HTML/CSS/JS code blocks in a rendered message.
   * Replaces the first HTML block with a toggle + iframe, removes remaining HTML/CSS/JS blocks.
   */
  private postProcessHtmlBlocks(contentElement: HTMLElement, content: string): void {
    const blocks = extractCodeBlocks(content);
    const htmlBlocks = blocks.filter((b) => b.language === 'html');
    if (htmlBlocks.length === 0) return;

    const allPreCodes = contentElement.querySelectorAll('pre > code');
    let firstHtmlPre: HTMLElement | null = null;
    const toRemove: HTMLElement[] = [];

    allPreCodes.forEach((code) => {
      const className = code.getAttribute('class') || '';
      const lang = className.replace('language-', '').trim().toLowerCase();
      const pre = code.parentElement;
      if (!pre) return;

      if (lang === 'html') {
        if (!firstHtmlPre) {
          firstHtmlPre = pre;
        } else {
          toRemove.push(pre as HTMLElement);
        }
      } else if (lang === 'css' || lang === 'javascript' || lang === 'js') {
        toRemove.push(pre as HTMLElement);
      }
    });

    if (!firstHtmlPre) return;

    const i18n = WIDGET_I18N[this.config.language];
    const htmlDoc = buildHtmlDoc(blocks);

    // Build render box
    const wrapper = createElement('div', { class: 'agentix-chat-html-renderbox' });

    const toggle = createElement('div', { class: 'agentix-chat-code-toggle' });
    const codeBtn = createElement('button', { class: 'active' }, i18n.codeView);
    const webBtn = createElement('button', {}, i18n.webView);
    toggle.appendChild(codeBtn);
    toggle.appendChild(webBtn);
    wrapper.appendChild(toggle);

    const iframeWrapper = createElement('div', { class: 'agentix-chat-html-iframe-wrapper' });
    const iframe = createElement('iframe', {
      class: 'agentix-chat-html-iframe',
      sandbox: 'allow-scripts',
      srcdoc: htmlDoc,
      title: 'HTML preview'
    }) as HTMLIFrameElement;
    iframe.style.display = 'none';
    iframeWrapper.appendChild(iframe);

    const maxBtn = createElement('button', { class: 'agentix-chat-html-maximize', title: 'Fullscreen' });
    maxBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>';
    maxBtn.style.display = 'none';
    maxBtn.addEventListener('click', () => {
      this.showHtmlPreviewModal(htmlDoc);
    });
    iframeWrapper.appendChild(maxBtn);
    wrapper.appendChild(iframeWrapper);

    // Move the original pre into the wrapper so it can be toggled
    if (firstHtmlPre) {
      const preEl = firstHtmlPre as HTMLElement;
      const parent = preEl.parentElement;
      if (parent) {
        parent.insertBefore(wrapper, preEl);
        wrapper.appendChild(preEl);
      }
    }

    // Toggle handlers
    codeBtn.addEventListener('click', () => {
      codeBtn.classList.add('active');
      webBtn.classList.remove('active');
      firstHtmlPre!.style.display = '';
      iframe.style.display = 'none';
      maxBtn.style.display = 'none';
    });
    webBtn.addEventListener('click', () => {
      webBtn.classList.add('active');
      codeBtn.classList.remove('active');
      firstHtmlPre!.style.display = 'none';
      iframe.style.display = '';
      maxBtn.style.display = '';
    });

    // Remove extra blocks
    toRemove.forEach((el) => el.remove());
  }

  /**
   * Update badge with unread count
   */
  private updateBadge(): void {
    if (!this.toggleButton) return;

    const badge = this.toggleButton.querySelector('.agentix-chat-badge');

    if (this.unreadCount > 0 && !this.isOpen) {
      if (badge) {
        badge.textContent = String(this.unreadCount);
      } else {
        const newBadge = createElement('div', {
          class: 'agentix-chat-badge'
        }, String(this.unreadCount));
        this.toggleButton.appendChild(newBadge);
      }
    } else if (badge) {
      badge.remove();
    }
  }

  /**
   * Save messages to localStorage
   */
  private saveMessages(): void {
    try {
      const messagesToSave = this.messages.filter(m => m.role !== 'system');
      localStorage.setItem('agentix-chat-messages', JSON.stringify(messagesToSave));
    } catch (error) {
      console.warn('Could not save messages to localStorage:', error);
    }
  }

  /**
   * Load messages from localStorage
   */
  private loadMessages(): void {
    try {
      const saved = localStorage.getItem('agentix-chat-messages');
      if (saved) {
        this.messages = JSON.parse(saved);
        if (this.messagesContainer && this.messages.length > 0) {
          this.messagesContainer.innerHTML = '';
          this.messages.forEach(msg => this.renderMessage(msg));
        }
      }
    } catch (error) {
      console.warn('Could not load messages from localStorage:', error);
    }
  }

  /**
   * Get message count
   */
  getMessageCount(): number {
    return this.messages.filter(m => m.role !== 'system').length;
  }

  /**
   * Clear messages
   */
  clearMessages(): void {
    this.messages = [];
    this.unreadCount = 0;
    this.updateBadge();
    this.saveMessages();
    this.stopWebcamEmotion();
    this.stopEmotionPolling();
    if (this.messagesContainer) {
      this.showEmptyState();
    }
  }

  /**
   * Destroy widget
   */
  destroy(): void {
    this.stopWebcamEmotion();
    this.stopEmotionPolling();
    if (this.windowResizeHandler) {
      window.removeEventListener('resize', this.windowResizeHandler);
      this.windowResizeHandler = null;
    }
    if (this.container) {
      this.container.remove();
    }
    if (this.styleElement) {
      this.styleElement.remove();
    }
    this.abortController?.abort();
  }

  /**
   * Reset conversation id so the next request uses a fresh session.
   */
  resetConversationId(): void {
    sessionStorage.removeItem('agentix-conversation-id');
    this.setupAPIClient();
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<ChatWidgetConfig>): void {
    if (newConfig.language as string === 'auto') {
      newConfig = { ...newConfig, language: detectLanguage() };
    }
    this.config = { ...this.config, ...newConfig };
    // Re-apply logo URL fallbacks so empty strings don't wipe out defaults
    if (!this.config.logoUrlButton) {
      this.config.logoUrlButton = this.config.logoUrl || WIDGET_FAVICON_BASE64;
    }
    if (!this.config.logoUrlHeader) {
      this.config.logoUrlHeader = this.config.logoUrl || WIDGET_LOGO_BASE64;
    }
    this.validateConfig();
    this.setupAPIClient();

    // Update title
    if (newConfig.title !== undefined) {
      const titleEl = this.container?.querySelector('.agentix-chat-header-title');
      if (titleEl) {
        titleEl.textContent = this.config.title;
      }
    }

    // Update subtitle
    if (newConfig.subtitle !== undefined) {
      const subtitleEl = this.container?.querySelector('.agentix-chat-header-subtitle');
      if (subtitleEl) {
        subtitleEl.textContent = this.config.subtitle;
      }
    }

    // Update language or localized text overrides → recalculate i18n strings and refresh visible text
    const localizedKeys: (keyof ChatWidgetConfig)[] = [
      'privacyText', 'privacyTextDe', 'privacyTextEn',
      'initialMessageContent', 'initialMessageContentDe', 'initialMessageContentEn',
      'bounceText', 'bounceTextDe', 'bounceTextEn',
      'inputPlaceholder', 'inputPlaceholderDe', 'inputPlaceholderEn',
    ];
    if (newConfig.language !== undefined || localizedKeys.some((k) => newConfig[k] !== undefined)) {
      const lang = this.config.language;
      const i18n = WIDGET_I18N[lang];
      const privacyOverride = lang === 'de' ? this.config.privacyTextDe : this.config.privacyTextEn;
      const initialOverride = lang === 'de' ? this.config.initialMessageContentDe : this.config.initialMessageContentEn;
      const bounceOverride = lang === 'de' ? this.config.bounceTextDe : this.config.bounceTextEn;
      const placeholderOverride = lang === 'de' ? this.config.inputPlaceholderDe : this.config.inputPlaceholderEn;
      this.config.privacyText = privacyOverride || this.config.privacyText || i18n.privacyText;
      this.config.initialMessageContent = initialOverride || this.config.initialMessageContent || i18n.initialMessageContent;
      this.config.bounceText = bounceOverride || this.config.bounceText || i18n.bounceText;
      this.config.inputPlaceholder = placeholderOverride || i18n.inputPlaceholder;

      // Update input placeholder
      if (this.inputField) {
        this.inputField.placeholder = this.privacyAccepted ? this.config.inputPlaceholder : i18n.privacyDisabledPlaceholder;
      }
      // Update privacy modal elements
      const privacyTitleEl = this.container?.querySelector('.agentix-chat-privacy-title');
      if (privacyTitleEl) {
        privacyTitleEl.textContent = i18n.privacyTitle;
      }
      const privacyTextEl = this.container?.querySelector('.agentix-chat-privacy-text');
      if (privacyTextEl) {
        privacyTextEl.textContent = this.config.privacyText;
      }
      const privacyAcceptEl = this.container?.querySelector('.agentix-chat-privacy-accept');
      if (privacyAcceptEl) {
        privacyAcceptEl.textContent = i18n.privacyAccept;
      }
      const privacyDeclineEl = this.container?.querySelector('.agentix-chat-privacy-decline');
      if (privacyDeclineEl) {
        privacyDeclineEl.textContent = i18n.privacyDecline;
      }
      // Update empty state text
      const emptyStateEl = this.container?.querySelector('.agentix-chat-empty-state-text');
      if (emptyStateEl) {
        emptyStateEl.innerHTML = i18n.emptyStateText;
      }
      // Update bounce bubble text
      if (this.bubbleElement) {
        this.bubbleElement.textContent = this.config.bounceText;
      }
    }

    // Update model → re-fetch audio capabilities
    if (newConfig.model !== undefined) {
      this.fetchAudioCapabilities();
    }

    // Update colors
    if (newConfig.colors) {
      this.applyColorVariables();
    }

    // Update showLogoInButton + logoGlow (toggle button)
    if (newConfig.showLogoInButton !== undefined || newConfig.logoBorder !== undefined || newConfig.logoGlow !== undefined || newConfig.logoBackground !== undefined) {
      this.refreshToggleButtonLogo();
    }

    // Update logo (logoUrlButton, logoUrlHeader, showLogoInHeader, logoBorder, logoGlow, or logoBackground)
    if (newConfig.logoUrlButton !== undefined || newConfig.logoUrlHeader !== undefined || newConfig.logoUrl !== undefined || newConfig.showLogoInHeader !== undefined || newConfig.logoBorder !== undefined || newConfig.logoGlow !== undefined || newConfig.logoBackground !== undefined) {
      this.refreshHeaderLogo();
      // Also update toggle button if logoBorder/logoGlow/logoBackground changed but logoUrl didn't
      if (newConfig.logoBorder !== undefined || newConfig.logoGlow !== undefined || newConfig.logoBackground !== undefined) {
        this.refreshToggleButtonLogo();
      }
    }

    // Update bounce settings
    if (newConfig.bounceInterval !== undefined || newConfig.bounceText !== undefined) {
      if (!this.isOpen) {
        this.stopBounce();
        this.startBounce();
      }
    }

    // Update position/theme classes on container and window
    if (newConfig.position !== undefined || newConfig.theme !== undefined) {
      if (this.container) {
        this.container.className = `agentix-chat-widget-container ${this.config.position} ${this.config.theme}`;
      }
      if (this.window && newConfig.position !== undefined) {
        this.window.className = `agentix-chat-window ${this.config.position}`;
      }
    }

    // Update window size
    if (newConfig.width !== undefined || newConfig.height !== undefined) {
      if (this.window) {
        this.window.style.width = `${this.config.width}px`;
        this.window.style.height = `${this.config.height}px`;
      }
    }

    // privacyEnabled flipped off → accept + enable + hide modal immediately
    if (newConfig.privacyEnabled === false && !this.privacyAccepted) {
      this.privacyAccepted = true;
      this.enableChat();
      if (this.privacyModal) {
        this.privacyModal.style.display = 'none';
      }
    }
  }
}
