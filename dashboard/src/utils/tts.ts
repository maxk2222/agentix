/**
 * TTS streaming utilities for sentence-level audio playback.
 *
 * extractCompleteSentences  – detects sentence boundaries in a text
 *                              buffer while respecting abbreviations.
 * TTSSegmentPlayer            – queues audio segments and plays them
 *                              sequentially, pre-fetching the next
 *                              blob while the current one is playing.
 */

/* ------------------------------------------------------------------ */
/*  Sentence extraction                                                */
/* ------------------------------------------------------------------ */

const ABBREVIATIONS = new Set([
  // German
  'z.b', 'u.a', 'etc', 'dr', 'prof', 'nr', 'bzw', 'd.h', 'ca', 'usw',
  'inkl', 'exkl', 'vgl', 'siehe', 'v.a', 'ab', 'abk', 'allg', 'anm',
  'anon', 'anst', 'a.ö', 'Bd', 'begr', 'bes', 'beisp', 'betr', 'bez',
  'bh', 'bibl', 'biol', 'bk', 'bl', 'chem', 'co', 'd.h', 'd.i', 'd.j',
  'd.m', 'd.r', 'd.s', 'd.v', 'dat', 'dgl', 'dipl', 'diss', 'dt',
  'e.h', 'e.v', 'ebd', 'ed', 'eig', 'eigtl', 'eing', 'engl', 'entspr',
  'erw', 'etw', 'evtl', 'exkl', 'f', 'fa', 'fam', 'ff', 'fig', 'fr',
  'g.g.a', 'g.u', 'geb', 'geh', 'gen', 'ges', 'ggf', 'gr', 'hbf',
  'hebr', 'hg', 'hin', 'hist', 'hiv', 'hl', 'hr', 'hrsg', 'i.a', 'i.d',
  'i.e', 'i.v', 'i.w.s', 'ibid', 'ig', 'ii', 'incl', 'inkl', 'int',
  'i.a', 'jhd', 'jhdt', 'jr', 'jur', 'k.u.k', 'kath', 'kfm', 'kg',
  'kl', 'k.o', 'k.u.k', 'lab', 'lat', 'lfd', 'lit', 'lt', 'm.e',
  'm.m.n', 'm.w.n', 'med', 'mev', 'min', 'mm', 'mio', 'mrd', 'mtl',
  'n.chr', 'n.d', 'n.f', 'n.j', 'n.m', 'n.r', 'n.w', 'n.z', 'nachf',
  'nat', 'nd', 'nr', 'o.a', 'o.ä', 'o.g', 'o.k', 'od', 'ohg',
  'op', 'ord', 'orig', 'p.a', 'p.m', 'p.p', 'p.s', 'pa', 'pag', 'pers',
  'pf', 'phil', 'pl', 'pol', 'pp', 'pr', 'priv', 'prof', 'prot', 'q.v',
  'r.k', 'rd', 'reg', 'resp', 'rev', 'röm', 's.a', 's.b', 's.o', 's.u',
  's.v', 'sa', 'sel', 'sen', 'seit', 'so', 'soz', 'sp', 'spp', 'sr',
  'ss', 'st', 'std', 'str', 'täg', 'tab', 'techn', 'tel', 'theol',
  'tr', 'u.a', 'u.ä', 'u.s.w', 'u.v.a', 'u.v.m', 'u.z', 'ult', 'univ',
  'urspr', 'usw', 'u.z', 'v.a', 'v.chr', 'v.g', 'v.h', 'v.m', 'v.o',
  'v.r', 'v.t', 'v.u.z', 'verh', 'verm', 'vers', 'vgl', 'viz', 'vol',
  'vollst', 'vorbem', 'vors', 'w.e', 'w.o', 'w.u', 'wiss', 'wohl',
  'z.b', 'z.t', 'z.u', 'zus', 'zz', 'zzt',
  // English
  'i.e', 'e.g', 'vs', 'dr', 'mr', 'mrs', 'ms', 'prof', 'sr', 'jr',
  'st', 'ave', 'blvd', 'rd', 'ln', 'ct', ' approx', 'inc', 'ltd',
  'co', 'corp', 'llc', 'plc', 'et al', 'vol', 'vols', 'rev', 'est',
  'fig', 'figs', 'gov', 'dept', 'dept', 'nos', 'art', 'arts',
  'pp', 'par', 'pars', 'eq', 'eqs', 'ch', 'chs', 'ed', 'eds',
  'trans', 'vol', 'vols', 'bk', 'bks', 'pt', 'pts', 'sec', 'secs',
  'nos', 'circa', 'ca', 'a.m', 'p.m', 'b.c', 'a.d', 'ph.d',
  'm.d', 'b.a', 'm.a', 'd.d.s', 'd.v.m', 'esq', 'hon', 'm.b.a',
  'm.d', 'ph.d', 'r.n', 'rep', 'reps', 'sen', 'sens', 'col', 'cols',
  'gen', 'gens', 'lt', 'lts', 'maj', 'sgt', 'cpl', 'capt', 'cmdr',
  'adm', 'rev', 'revs', 'hon', 'hons', 'assoc', 'asst', 'assts',
  'ext', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep',
  'sept', 'oct', 'nov', 'dec', 'mon', 'tue', 'wed', 'thu', 'fri',
  'sat', 'sun',
]);

/** Lower-case and strip trailing punctuation from a word for abbreviation lookup. */
function normWord(w: string): string {
  return w
    .toLowerCase()
    .replace(/[.,;:!?]+$/, '')
    .replace(/^\W+/, '');
}

/**
 * Estimate pause duration (in ms) based on the whitespace that
 * follows a sentence terminator.
 */
function estimatePause(whitespace: string): number {
  if (whitespace.includes('\n\n')) return 600;
  if (whitespace.includes('\n')) return 300;
  return 150;
}

/**
 * Extract complete sentences from a growing text buffer.
 *
 * Returns the extracted sentences (with estimated pause after each),
 * and the leftover remainder that should be kept in the buffer for
 * the next chunk.
 */
export function extractCompleteSentences(buffer: string): { chunks: TTSChunk[]; remainder: string } {
  if (!buffer.trim()) return { chunks: [], remainder: buffer };

  const chunks: TTSChunk[] = [];
  let lastSplit = 0;

  // We scan character by character looking for sentence terminators.
  // A terminator followed by whitespace (space, newline, tab) marks
  // a potential split, unless the preceding word is a known abbreviation.
  for (let i = 0; i < buffer.length; i++) {
    const ch = buffer[i];
    if (ch !== '.' && ch !== '!' && ch !== '?') continue;

    // Look ahead: must be followed by whitespace or end-of-string
    const next = buffer[i + 1];
    if (next !== undefined && next !== ' ' && next !== '\n' && next !== '\t' && next !== '\r') {
      continue;
    }

    // Extract the candidate sentence text (from lastSplit up to and including terminator)
    const candidate = buffer.slice(lastSplit, i + 1).trim();
    if (candidate.length < 3) {
      // Too short – could be an ellipsis fragment or stray punctuation.
      // Still advance lastSplit so we don't get stuck.
      lastSplit = i + 1;
      continue;
    }

    // Check the word immediately before the terminator
    const before = buffer.slice(lastSplit, i).trimEnd();
    const words = before.split(/\s+/);
    const lastWord = words[words.length - 1] || '';
    const norm = normWord(lastWord);

    if (ABBREVIATIONS.has(norm)) {
      // This is an abbreviation, do NOT split here
      continue;
    }

    // Capture the whitespace after the terminator to estimate pause length
    let wsEnd = i + 1;
    while (wsEnd < buffer.length && /\s/.test(buffer[wsEnd])) {
      wsEnd++;
    }
    const whitespace = buffer.slice(i + 1, wsEnd);

    chunks.push({ text: candidate, pauseAfterMs: estimatePause(whitespace) });
    lastSplit = wsEnd;
  }

  const remainder = buffer.slice(lastSplit);
  return { chunks, remainder };
}

/* ------------------------------------------------------------------ */
/*  Segment player                                                     */
/* ------------------------------------------------------------------ */

export interface TTSChunk {
  text: string;
  pauseAfterMs: number;
}

export interface SegmentItem {
  text: string;
  voiceId: string;
  blobPromise: Promise<Blob>;
  pauseAfterMs?: number;
}

/**
 * Queues TTS audio segments and plays them sequentially.
 *
 * When a segment is added its audio blob is fetched immediately
 * (parallel).  Playback happens in order: the player waits for the
 * first blob, plays it, then moves to the next.  This overlaps
 * network I/O with playback so audio keeps flowing.
 */
export class TTSSegmentPlayer {
  private queue: SegmentItem[] = [];
  private isPlaying = false;
  private currentAudio: HTMLAudioElement | null = null;
  private _playedAny = false;

  /** Number of segments currently queued (including the one playing). */
  get queueLength(): number {
    return this.queue.length;
  }

  /** True if at least one segment started playing successfully. */
  hasPlayedAny(): boolean {
    return this._playedAny;
  }

  /**
   * Add a sentence to the queue.  `fetchSegment` must return a
   * Promise<Blob> (e.g. a fetch to /audio/tts/segment).
   */
  add(text: string, voiceId: string, fetchSegment: (text: string, voiceId: string) => Promise<Blob>, pauseAfterMs?: number): void {
    const blobPromise = fetchSegment(text, voiceId);
    this.queue.push({ text, voiceId, blobPromise, pauseAfterMs });
    if (!this.isPlaying) {
      this.playNext();
    }
  }

  /** Stop playback, clear the queue, and revoke any object URLs. */
  abort(): void {
    if (this.currentAudio) {
      this.currentAudio.pause();
      const src = this.currentAudio.src;
      if (src && src.startsWith('blob:')) {
        URL.revokeObjectURL(src);
      }
      this.currentAudio = null;
    }
    // Revoke URLs for items still in the queue whose promises have
    // already resolved but were never played.
    this.queue = [];
    this.isPlaying = false;
  }

  private async playNext(): Promise<void> {
    if (this.isPlaying) return;
    this.isPlaying = true;

    while (this.queue.length > 0) {
      const item = this.queue.shift()!;
      try {
        const blob = await item.blobPromise;
        if (this.queue.length === 0 && !this.isPlaying) {
          // aborted while waiting
          URL.revokeObjectURL(URL.createObjectURL(blob));
          break;
        }
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        this.currentAudio = audio;

        await new Promise<void>((resolve) => {
          audio.onended = () => {
            URL.revokeObjectURL(url);
            this.currentAudio = null;
            this._playedAny = true;
            resolve();
          };
          audio.onerror = () => {
            URL.revokeObjectURL(url);
            this.currentAudio = null;
            console.warn('[TTSSegmentPlayer] Audio playback error, skipping segment');
            resolve();
          };
          audio.play().catch(() => {
            URL.revokeObjectURL(url);
            this.currentAudio = null;
            console.warn('[TTSSegmentPlayer] Audio play() rejected, skipping segment');
            resolve();
          });
        });

        // Pause between segments to preserve natural rhythm
        if (item.pauseAfterMs && item.pauseAfterMs > 0 && this.queue.length > 0) {
          await new Promise<void>((resolve) => setTimeout(resolve, item.pauseAfterMs));
        }
      } catch (err) {
        console.warn('[TTSSegmentPlayer] Failed to fetch segment, skipping:', err);
        // continue to next queued segment
      }
    }

    this.isPlaying = false;
  }
}
