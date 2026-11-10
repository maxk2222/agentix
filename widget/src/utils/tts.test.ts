import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { extractCompleteSentences, TTSSegmentPlayer, type TTSChunk } from './tts';

describe('extractCompleteSentences', () => {
  it('returns empty for empty buffer', () => {
    const result = extractCompleteSentences('');
    expect(result.chunks).toEqual([]);
    expect(result.remainder).toBe('');
  });

  it('returns empty for whitespace-only buffer', () => {
    const result = extractCompleteSentences('   \n\t  ');
    expect(result.chunks).toEqual([]);
    expect(result.remainder).toBe('   \n\t  ');
  });

  it('extracts a single complete sentence', () => {
    const result = extractCompleteSentences('Hello world.');
    expect(result.chunks).toEqual<TTSChunk[]>([{ text: 'Hello world.', pauseAfterMs: 150 }]);
    expect(result.remainder).toBe('');
  });

  it('buffers an incomplete sentence', () => {
    const result = extractCompleteSentences('Hello wor');
    expect(result.chunks).toEqual([]);
    expect(result.remainder).toBe('Hello wor');
  });

  it('extracts multiple sentences and keeps remainder', () => {
    const result = extractCompleteSentences('First sentence. Second sentence. Third par');
    expect(result.chunks).toEqual<TTSChunk[]>([
      { text: 'First sentence.', pauseAfterMs: 150 },
      { text: 'Second sentence.', pauseAfterMs: 150 },
    ]);
    expect(result.remainder).toBe('Third par');
  });

  it('handles exclamation and question marks', () => {
    const result = extractCompleteSentences('What? Yes! Great.');
    expect(result.chunks).toEqual<TTSChunk[]>([
      { text: 'What?', pauseAfterMs: 150 },
      { text: 'Yes!', pauseAfterMs: 150 },
      { text: 'Great.', pauseAfterMs: 150 },
    ]);
    expect(result.remainder).toBe('');
  });

  it('respects German abbreviations (z.B.)', () => {
    const result = extractCompleteSentences('Das ist z.B. ein Test. Nächster Satz.');
    expect(result.chunks).toEqual<TTSChunk[]>([
      { text: 'Das ist z.B. ein Test.', pauseAfterMs: 150 },
      { text: 'Nächster Satz.', pauseAfterMs: 150 },
    ]);
    expect(result.remainder).toBe('');
  });

  it('respects German abbreviations (u.a.)', () => {
    const result = extractCompleteSentences('Dabei sind u.a. die folgenden Punkte. Fertig.');
    expect(result.chunks).toEqual<TTSChunk[]>([
      { text: 'Dabei sind u.a. die folgenden Punkte.', pauseAfterMs: 150 },
      { text: 'Fertig.', pauseAfterMs: 150 },
    ]);
    expect(result.remainder).toBe('');
  });

  it('respects German abbreviations (d.h.)', () => {
    const result = extractCompleteSentences('Das ist d.h. ein Beispiel. Ende.');
    expect(result.chunks).toEqual<TTSChunk[]>([
      { text: 'Das ist d.h. ein Beispiel.', pauseAfterMs: 150 },
      { text: 'Ende.', pauseAfterMs: 150 },
    ]);
    expect(result.remainder).toBe('');
  });

  it('respects English abbreviations (Dr.)', () => {
    const result = extractCompleteSentences('Dr. Smith lives here. Next sentence.');
    expect(result.chunks).toEqual<TTSChunk[]>([
      { text: 'Dr. Smith lives here.', pauseAfterMs: 150 },
      { text: 'Next sentence.', pauseAfterMs: 150 },
    ]);
    expect(result.remainder).toBe('');
  });

  it('respects English abbreviations (i.e.)', () => {
    const result = extractCompleteSentences('Use i.e. for examples. Done.');
    expect(result.chunks).toEqual<TTSChunk[]>([
      { text: 'Use i.e. for examples.', pauseAfterMs: 150 },
      { text: 'Done.', pauseAfterMs: 150 },
    ]);
    expect(result.remainder).toBe('');
  });

  it('respects English abbreviations (e.g.)', () => {
    const result = extractCompleteSentences('Fruits e.g. apples and oranges. Next.');
    expect(result.chunks).toEqual<TTSChunk[]>([
      { text: 'Fruits e.g. apples and oranges.', pauseAfterMs: 150 },
      { text: 'Next.', pauseAfterMs: 150 },
    ]);
    expect(result.remainder).toBe('');
  });

  it('estimates 300 ms pause after single newline', () => {
    const result = extractCompleteSentences('First line.\nSecond line.');
    expect(result.chunks).toEqual<TTSChunk[]>([
      { text: 'First line.', pauseAfterMs: 300 },
      { text: 'Second line.', pauseAfterMs: 150 },
    ]);
  });

  it('estimates 600 ms pause after paragraph break', () => {
    const result = extractCompleteSentences('First line.\n\nSecond line.');
    expect(result.chunks).toEqual<TTSChunk[]>([
      { text: 'First line.', pauseAfterMs: 600 },
      { text: 'Second line.', pauseAfterMs: 150 },
    ]);
  });

  it('estimates 150 ms pause after plain space', () => {
    const result = extractCompleteSentences('Hello. World.');
    expect(result.chunks).toEqual<TTSChunk[]>([
      { text: 'Hello.', pauseAfterMs: 150 },
      { text: 'World.', pauseAfterMs: 150 },
    ]);
  });

  it('does not split on terminators not followed by whitespace', () => {
    const result = extractCompleteSentences('Value is 1.5 meters. Done.');
    expect(result.chunks).toEqual<TTSChunk[]>([
      { text: 'Value is 1.5 meters.', pauseAfterMs: 150 },
      { text: 'Done.', pauseAfterMs: 150 },
    ]);
    expect(result.remainder).toBe('');
  });

  it('handles ellipsis as sentence boundary', () => {
    const result = extractCompleteSentences('Wait... no. Done.');
    expect(result.chunks).toEqual<TTSChunk[]>([
      { text: 'Wait...', pauseAfterMs: 150 },
      { text: 'no.', pauseAfterMs: 150 },
      { text: 'Done.', pauseAfterMs: 150 },
    ]);
    expect(result.remainder).toBe('');
  });

  it('skips very short candidates (< 3 chars)', () => {
    const result = extractCompleteSentences('A. This is real.');
    expect(result.chunks).toEqual<TTSChunk[]>([{ text: 'This is real.', pauseAfterMs: 150 }]);
    expect(result.remainder.trim()).toBe('');
  });

  it('handles chunked input progressively', () => {
    let buffer = 'First sent';
    let r1 = extractCompleteSentences(buffer);
    expect(r1.chunks).toEqual([]);
    expect(r1.remainder).toBe('First sent');

    buffer = r1.remainder + 'ence. Second s';
    let r2 = extractCompleteSentences(buffer);
    expect(r2.chunks).toEqual<TTSChunk[]>([{ text: 'First sentence.', pauseAfterMs: 150 }]);
    expect(r2.remainder).toBe('Second s');

    buffer = r2.remainder + 'entence.';
    let r3 = extractCompleteSentences(buffer);
    expect(r3.chunks).toEqual<TTSChunk[]>([{ text: 'Second sentence.', pauseAfterMs: 150 }]);
    expect(r3.remainder).toBe('');
  });
});

describe('TTSSegmentPlayer', () => {
  let originalAudio: typeof globalThis.Audio;
  let originalCreateObjectURL: typeof URL.createObjectURL;
  let originalRevokeObjectURL: typeof URL.revokeObjectURL;

  beforeEach(() => {
    vi.useFakeTimers();
    originalAudio = globalThis.Audio;
    originalCreateObjectURL = URL.createObjectURL;
    originalRevokeObjectURL = URL.revokeObjectURL;

    let playResolve: (() => void) | null = null;

    globalThis.Audio = vi.fn().mockImplementation(() => {
      const audio = {
        src: '',
        onended: null as (() => void) | null,
        onerror: null as (() => void) | null,
        play: vi.fn().mockImplementation(() => {
          return new Promise<void>((resolve) => {
            playResolve = resolve;
            queueMicrotask(() => {
              if (audio.onended) {
                audio.onended();
              }
            });
          });
        }),
        pause: vi.fn(),
      };
      return audio;
    }) as unknown as typeof globalThis.Audio;

    URL.createObjectURL = vi.fn(() => 'blob:mock-url');
    URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    vi.useRealTimers();
    globalThis.Audio = originalAudio;
    URL.createObjectURL = originalCreateObjectURL;
    URL.revokeObjectURL = originalRevokeObjectURL;
  });

  it('plays segments sequentially', async () => {
    const player = new TTSSegmentPlayer();
    const fetchFn = vi.fn().mockResolvedValue(new Blob(['audio1']));

    player.add('Hello.', 'voice1', fetchFn);
    player.add('World.', 'voice1', fetchFn);

    await vi.advanceTimersByTimeAsync(50);

    expect(fetchFn).toHaveBeenCalledTimes(2);
    expect(fetchFn).toHaveBeenNthCalledWith(1, 'Hello.', 'voice1');
    expect(fetchFn).toHaveBeenNthCalledWith(2, 'World.', 'voice1');
    expect(player.hasPlayedAny()).toBe(true);
  });

  it('skips failed segments and continues', async () => {
    const player = new TTSSegmentPlayer();
    const fetchFn = vi.fn()
      .mockRejectedValueOnce(new Error('Network error'))
      .mockResolvedValueOnce(new Blob(['audio2']));

    player.add('Fail.', 'voice1', fetchFn);
    player.add('Success.', 'voice1', fetchFn);

    await vi.advanceTimersByTimeAsync(50);

    expect(fetchFn).toHaveBeenCalledTimes(2);
    expect(player.hasPlayedAny()).toBe(true);
  });

  it('hasPlayedAny returns false before any successful play', async () => {
    const player = new TTSSegmentPlayer();
    expect(player.hasPlayedAny()).toBe(false);

    const fetchFn = vi.fn().mockRejectedValue(new Error('fail'));
    player.add('Fail.', 'voice1', fetchFn);

    await vi.advanceTimersByTimeAsync(50);
    expect(player.hasPlayedAny()).toBe(false);
  });

  it('abort stops playback and clears queue', async () => {
    const player = new TTSSegmentPlayer();
    const fetchFn = vi.fn().mockImplementation(() => new Promise(() => {}));

    player.add('Hello.', 'voice1', fetchFn);
    player.abort();

    expect(player.queueLength).toBe(0);
    expect(player.hasPlayedAny()).toBe(false);
  });

  it('revokes object URLs after playback', async () => {
    const player = new TTSSegmentPlayer();
    const fetchFn = vi.fn().mockResolvedValue(new Blob(['audio']));

    player.add('Hello.', 'voice1', fetchFn);
    await vi.advanceTimersByTimeAsync(50);

    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
  });

  it('inserts pause between segments when pauseAfterMs is set', async () => {
    const player = new TTSSegmentPlayer();
    const fetchFn = vi.fn().mockResolvedValue(new Blob(['audio']));

    player.add('Hello.', 'voice1', fetchFn, 200);
    player.add('World.', 'voice1', fetchFn, 0);

    await vi.advanceTimersByTimeAsync(10);
    expect(player.hasPlayedAny()).toBe(true);

    await vi.advanceTimersByTimeAsync(200);
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it('does not pause after the last segment', async () => {
    const player = new TTSSegmentPlayer();
    const fetchFn = vi.fn().mockResolvedValue(new Blob(['audio']));

    player.add('Hello.', 'voice1', fetchFn, 200);

    await vi.advanceTimersByTimeAsync(50);
    expect(player.hasPlayedAny()).toBe(true);
    expect(URL.revokeObjectURL).toHaveBeenCalled();
  });
});
