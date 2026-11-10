import { describe, it, expect } from 'vitest';
import { mergeServerDefaults, sanitizeServerDefaults } from './defaults';

describe('mergeServerDefaults', () => {
  it('applies server values for keys the user did not provide', () => {
    const result = mergeServerDefaults({ title: 'My Site' }, { title: 'Server', theme: 'dark' });
    expect(result).toEqual({ theme: 'dark' });
  });

  it('lets explicit user values win over server defaults', () => {
    const result = mergeServerDefaults(
      { title: 'My Site', width: 400 },
      { title: 'Server Title', width: 800, height: 600 },
    );
    expect(result).toEqual({ height: 600 });
  });

  it('merges object keys server-under-user (sub-key level)', () => {
    const result = mergeServerDefaults(
      { colors: { primary: '#ff0000' } },
      { colors: { primary: '#0000ff', background: '#00ff00' } },
    );
    expect(result).toEqual({ colors: { primary: '#ff0000', background: '#00ff00' } });
  });

  it('takes server object when user did not provide one', () => {
    const result = mergeServerDefaults({}, { logoGlow: { enabled: false, opacity: 0.2 } });
    expect(result).toEqual({ logoGlow: { enabled: false, opacity: 0.2 } });
  });

  it('never applies secrets or callbacks from server defaults', () => {
    const result = mergeServerDefaults(
      {},
      { apiKey: 'sk-leak', apiSecret: 'x', onOpen: 'fn', onInit: 'fn', onError: 'fn', title: 'T' },
    );
    expect(result).toEqual({ title: 'T' });
  });

  it("drops language 'auto' but keeps explicit languages", () => {
    expect(mergeServerDefaults({}, { language: 'auto' })).toEqual({});
    expect(mergeServerDefaults({}, { language: 'de' })).toEqual({ language: 'de' });
  });

  it('skips undefined values', () => {
    expect(mergeServerDefaults({}, { title: undefined, theme: 'dark' })).toEqual({ theme: 'dark' });
  });

  it('user-provided object wins entirely for non-merge keys', () => {
    const result = mergeServerDefaults(
      { logoBorder: { enabled: true } },
      { logoBorder: { enabled: false, color: '#123456' } },
    );
    expect(result).toEqual({ logoBorder: { enabled: true, color: '#123456' } });
  });

  it('returns empty object when nothing applies', () => {
    expect(mergeServerDefaults({ title: 'x' }, { title: 'y' })).toEqual({});
    expect(mergeServerDefaults({}, {})).toEqual({});
  });
});

describe('sanitizeServerDefaults', () => {
  it('strips secrets and callbacks', () => {
    const result = sanitizeServerDefaults({
      apiKey: 'sk',
      apiSecret: 's',
      onOpen: 'fn',
      onInit: 'fn',
      onError: 'fn',
      title: 'Bot',
    });
    expect(result).toEqual({ title: 'Bot' });
  });

  it("drops language 'auto'", () => {
    expect(sanitizeServerDefaults({ language: 'auto', title: 'T' })).toEqual({ title: 'T' });
  });
});
