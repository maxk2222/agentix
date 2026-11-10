import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { detectLanguage, ChatWidget } from './widget';

describe('detectLanguage', () => {
  let originalLang: string;
  let originalLanguage: PropertyDescriptor | undefined;
  let originalLanguages: PropertyDescriptor | undefined;

  beforeEach(() => {
    originalLang = document.documentElement.lang;
    originalLanguage = Object.getOwnPropertyDescriptor(window.navigator, 'language');
    originalLanguages = Object.getOwnPropertyDescriptor(window.navigator, 'languages');
  });

  afterEach(() => {
    document.documentElement.lang = originalLang;
    if (originalLanguage) {
      Object.defineProperty(window.navigator, 'language', originalLanguage);
    }
    if (originalLanguages) {
      Object.defineProperty(window.navigator, 'languages', originalLanguages);
    }
  });

  function setNavigatorLanguage(lang: string | undefined) {
    Object.defineProperty(window.navigator, 'language', {
      value: lang,
      writable: true,
      configurable: true,
    });
  }

  function setNavigatorLanguages(langs: string[] | undefined) {
    Object.defineProperty(window.navigator, 'languages', {
      value: langs,
      writable: true,
      configurable: true,
    });
  }

  it('returns "de" from navigator.language even if html lang is "en"', () => {
    setNavigatorLanguage('de-DE');
    document.documentElement.lang = 'en';
    expect(detectLanguage()).toBe('de');
  });

  it('returns "en" from navigator.language even if html lang is "de"', () => {
    setNavigatorLanguage('en-US');
    document.documentElement.lang = 'de';
    expect(detectLanguage()).toBe('en');
  });

  it('returns "de" from navigator.language "de-AT"', () => {
    setNavigatorLanguage('de-AT');
    document.documentElement.lang = 'en-US';
    expect(detectLanguage()).toBe('de');
  });

  it('falls back to html lang when navigator.language is absent', () => {
    document.documentElement.lang = 'de';
    setNavigatorLanguage(undefined);
    setNavigatorLanguages(undefined);
    expect(detectLanguage()).toBe('de');
  });

  it('falls back to navigator.languages when navigator.language is absent', () => {
    document.documentElement.lang = '';
    setNavigatorLanguage(undefined);
    setNavigatorLanguages(['en-US', 'de-DE']);
    expect(detectLanguage()).toBe('de');
  });

  it('prefers de over en in navigator.languages', () => {
    document.documentElement.lang = '';
    setNavigatorLanguage(undefined);
    setNavigatorLanguages(['fr-FR', 'de-AT']);
    expect(detectLanguage()).toBe('de');
  });

  it('returns "en" from navigator.languages when only en is present', () => {
    document.documentElement.lang = '';
    setNavigatorLanguage(undefined);
    setNavigatorLanguages(['en-US', 'fr-FR']);
    expect(detectLanguage()).toBe('en');
  });

  it('returns "de" as ultimate fallback when nothing matches', () => {
    document.documentElement.lang = '';
    setNavigatorLanguage(undefined);
    setNavigatorLanguages(['fr-FR', 'es-ES']);
    expect(detectLanguage()).toBe('de');
  });

  it('falls back to html lang when navigator properties are undefined', () => {
    document.documentElement.lang = 'en';
    setNavigatorLanguage(undefined);
    setNavigatorLanguages(undefined);
    expect(detectLanguage()).toBe('en');
  });

  it('returns "de" when everything is absent', () => {
    document.documentElement.lang = '';
    setNavigatorLanguage(undefined);
    setNavigatorLanguages(undefined);
    expect(detectLanguage()).toBe('de');
  });
});

describe('ChatWidget language "auto" handling', () => {
  let originalLang: string;
  let originalLanguage: PropertyDescriptor | undefined;
  let originalLanguages: PropertyDescriptor | undefined;

  beforeEach(() => {
    originalLang = document.documentElement.lang;
    originalLanguage = Object.getOwnPropertyDescriptor(window.navigator, 'language');
    originalLanguages = Object.getOwnPropertyDescriptor(window.navigator, 'languages');
  });

  afterEach(() => {
    document.documentElement.lang = originalLang;
    if (originalLanguage) {
      Object.defineProperty(window.navigator, 'language', originalLanguage);
    }
    if (originalLanguages) {
      Object.defineProperty(window.navigator, 'languages', originalLanguages);
    }
  });

  function setNavigatorLanguage(lang: string | undefined) {
    Object.defineProperty(window.navigator, 'language', {
      value: lang,
      writable: true,
      configurable: true,
    });
  }

  function setNavigatorLanguages(langs: string[] | undefined) {
    Object.defineProperty(window.navigator, 'languages', {
      value: langs,
      writable: true,
      configurable: true,
    });
  }

  it('falls back to detectLanguage when language is "auto"', () => {
    setNavigatorLanguage('en-US');
    setNavigatorLanguages(['en-US']);
    document.documentElement.lang = '';
    const widget = new ChatWidget({ language: 'auto' as any, apiUrl: 'http://localhost/v1' });
    expect((widget as any).config.language).toBe('en');
  });

  it('falls back to detectLanguage when language is omitted', () => {
    setNavigatorLanguage('de-DE');
    setNavigatorLanguages(['de-DE']);
    document.documentElement.lang = '';
    const widget = new ChatWidget({ apiUrl: 'http://localhost/v1' });
    expect((widget as any).config.language).toBe('de');
  });

  it('resolves "auto" in updateConfig to detectLanguage', () => {
    setNavigatorLanguage('en-US');
    setNavigatorLanguages(['en-US']);
    document.documentElement.lang = '';
    const widget = new ChatWidget({ language: 'de', apiUrl: 'http://localhost/v1' });
    expect((widget as any).config.language).toBe('de');
    widget.updateConfig({ language: 'auto' as any });
    expect((widget as any).config.language).toBe('en');
  });
});

describe('ChatWidget header logo updates', () => {
  function makeWidget() {
    const widget = new ChatWidget({
      apiUrl: 'http://localhost/v1',
      logoUrlHeader: 'https://example.test/logo1.png',
      showLogoInHeader: true,
      logoBorder: { enabled: true } as any,
    });
    const header = (widget as any).container.querySelector('.agentix-chat-header');
    return { widget, header };
  }

  it('leaves at most one header logo node after repeated updates', () => {
    const { widget, header } = makeWidget();
    widget.updateConfig({ logoUrlHeader: 'https://example.test/logo2.png' });
    widget.updateConfig({ logoUrlHeader: 'https://example.test/logo3.png' });
    expect(header.querySelectorAll('.agentix-chat-header-logo').length).toBe(1);
    // The wrapper is re-created by createLogoElement when effects are on —
    // but no stale wrappers may accumulate (this was the "white dot" bug).
    expect(header.querySelectorAll('.agentix-chat-logo-wrapper').length).toBeLessThanOrEqual(1);
  });

  it('falls back to the builtin logo on empty url without stacking wrappers', () => {
    const { widget, header } = makeWidget();
    // '' is falsy → updateConfig re-defaults to the embedded logo (same
    // semantic as mergeServerDefaults); hiding is done via showLogoInHeader.
    widget.updateConfig({ logoUrlHeader: '' });
    expect(header.querySelectorAll('.agentix-chat-header-logo').length).toBe(1);
    expect(header.querySelectorAll('.agentix-chat-logo-wrapper').length).toBeLessThanOrEqual(1);
  });

  it('removes logo and wrapper entirely when logo is hidden', () => {
    const { widget, header } = makeWidget();
    widget.updateConfig({ showLogoInHeader: false });
    expect(header.querySelectorAll('.agentix-chat-header-logo').length).toBe(0);
    expect(header.querySelectorAll('.agentix-chat-logo-wrapper').length).toBe(0);
  });
});
