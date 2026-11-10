var __defProp = Object.defineProperty;
var __typeError = (msg) => {
  throw TypeError(msg);
};
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
var __accessCheck = (obj, member, msg) => member.has(obj) || __typeError("Cannot " + msg);
var __privateAdd = (obj, member, value) => member.has(obj) ? __typeError("Cannot add the same private member more than once") : member instanceof WeakSet ? member.add(obj) : member.set(obj, value);
var __privateMethod = (obj, member, method) => (__accessCheck(obj, member, "access private method"), method);
var _Marked_instances, parseMarkdown_fn, onError_fn;
class OpenAIClient {
  constructor(apiKey, apiUrl = "https://api.openai.com/v1", model = "gpt-4-turbo", conversationId) {
    __publicField(this, "apiKey");
    __publicField(this, "apiUrl");
    __publicField(this, "model");
    __publicField(this, "conversationId");
    this.apiKey = apiKey;
    this.apiUrl = apiUrl;
    this.model = model;
    this.conversationId = conversationId;
  }
  async getErrorMessage(response) {
    var _a;
    const text = await response.text();
    if (!text) return `HTTP ${response.status}`;
    try {
      const error = JSON.parse(text);
      return ((_a = error.error) == null ? void 0 : _a.message) || error.message || text;
    } catch {
      return text;
    }
  }
  /**
   * Stream chat completion from OpenAI
   */
  async *streamChat(messages, emotionContext) {
    var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j;
    const baseMessages = messages.map((m) => ({ role: m.role, content: m.content }));
    const request = {
      model: this.model,
      messages: baseMessages,
      temperature: 0.7,
      stream: true,
      ...this.conversationId ? { conversation_id: this.conversationId } : {},
      ...emotionContext ? { emotion_context: emotionContext } : {}
    };
    try {
      const headers = { "Content-Type": "application/json" };
      if (this.apiKey && this.apiKey.trim() !== "") {
        headers["Authorization"] = `Bearer ${this.apiKey}`;
      }
      const response = await fetch(`${this.apiUrl}/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify(request),
        credentials: "include"
      });
      if (!response.ok) {
        const errorMessage = await this.getErrorMessage(response);
        throw new Error(`API Error (${response.status}): ${errorMessage}`);
      }
      if (!response.body) {
        throw new Error("No response body from API");
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") {
              break;
            }
            try {
              const chunk = JSON.parse(data);
              const content = (_b = (_a = chunk.choices[0]) == null ? void 0 : _a.delta) == null ? void 0 : _b.content;
              const reasoning = (_d = (_c = chunk.choices[0]) == null ? void 0 : _c.delta) == null ? void 0 : _d.reasoning_content;
              const toolExecution = (_f = (_e = chunk.choices[0]) == null ? void 0 : _e.delta) == null ? void 0 : _f.tool_execution;
              const toolCalls = (_h = (_g = chunk.choices[0]) == null ? void 0 : _g.delta) == null ? void 0 : _h.tool_calls;
              const emotion = (_j = (_i = chunk.choices[0]) == null ? void 0 : _i.delta) == null ? void 0 : _j.emotion;
              if (content || reasoning || toolExecution || toolCalls || emotion) {
                yield { content, reasoning, toolExecution, toolCalls, emotion };
              }
            } catch (e) {
              continue;
            }
          }
        }
      }
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error("Unknown API error occurred");
    }
  }
  /**
   * Get non-streaming chat completion
   */
  async chat(messages) {
    var _a, _b;
    const baseMessages = messages.map((m) => ({ role: m.role, content: m.content }));
    const request = {
      model: this.model,
      messages: baseMessages,
      temperature: 0.7,
      stream: false,
      ...this.conversationId ? { conversation_id: this.conversationId } : {}
    };
    try {
      const headers = { "Content-Type": "application/json" };
      if (this.apiKey && this.apiKey.trim() !== "") {
        headers["Authorization"] = `Bearer ${this.apiKey}`;
      }
      const response = await fetch(`${this.apiUrl}/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify(request),
        credentials: "include"
      });
      if (!response.ok) {
        const errorMessage = await this.getErrorMessage(response);
        throw new Error(`API Error (${response.status}): ${errorMessage}`);
      }
      const data = await response.json();
      return ((_b = (_a = data.choices[0]) == null ? void 0 : _a.message) == null ? void 0 : _b.content) || "No response received";
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error("Unknown API error occurred");
    }
  }
}
function _getDefaults() {
  return {
    async: false,
    breaks: false,
    extensions: null,
    gfm: true,
    hooks: null,
    pedantic: false,
    renderer: null,
    silent: false,
    tokenizer: null,
    walkTokens: null
  };
}
let _defaults = _getDefaults();
function changeDefaults(newDefaults) {
  _defaults = newDefaults;
}
const escapeTest = /[&<>"']/;
const escapeReplace = new RegExp(escapeTest.source, "g");
const escapeTestNoEncode = /[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/;
const escapeReplaceNoEncode = new RegExp(escapeTestNoEncode.source, "g");
const escapeReplacements = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;"
};
const getEscapeReplacement = (ch) => escapeReplacements[ch];
function escape$1(html2, encode) {
  if (encode) {
    if (escapeTest.test(html2)) {
      return html2.replace(escapeReplace, getEscapeReplacement);
    }
  } else {
    if (escapeTestNoEncode.test(html2)) {
      return html2.replace(escapeReplaceNoEncode, getEscapeReplacement);
    }
  }
  return html2;
}
const unescapeTest = /&(#(?:\d+)|(?:#x[0-9A-Fa-f]+)|(?:\w+));?/ig;
function unescape(html2) {
  return html2.replace(unescapeTest, (_, n) => {
    n = n.toLowerCase();
    if (n === "colon")
      return ":";
    if (n.charAt(0) === "#") {
      return n.charAt(1) === "x" ? String.fromCharCode(parseInt(n.substring(2), 16)) : String.fromCharCode(+n.substring(1));
    }
    return "";
  });
}
const caret = /(^|[^\[])\^/g;
function edit(regex, opt) {
  let source = typeof regex === "string" ? regex : regex.source;
  opt = opt || "";
  const obj = {
    replace: (name, val) => {
      let valSource = typeof val === "string" ? val : val.source;
      valSource = valSource.replace(caret, "$1");
      source = source.replace(name, valSource);
      return obj;
    },
    getRegex: () => {
      return new RegExp(source, opt);
    }
  };
  return obj;
}
function cleanUrl(href) {
  try {
    href = encodeURI(href).replace(/%25/g, "%");
  } catch (e) {
    return null;
  }
  return href;
}
const noopTest = { exec: () => null };
function splitCells(tableRow, count) {
  const row = tableRow.replace(/\|/g, (match, offset, str) => {
    let escaped = false;
    let curr = offset;
    while (--curr >= 0 && str[curr] === "\\")
      escaped = !escaped;
    if (escaped) {
      return "|";
    } else {
      return " |";
    }
  }), cells = row.split(/ \|/);
  let i = 0;
  if (!cells[0].trim()) {
    cells.shift();
  }
  if (cells.length > 0 && !cells[cells.length - 1].trim()) {
    cells.pop();
  }
  if (count) {
    if (cells.length > count) {
      cells.splice(count);
    } else {
      while (cells.length < count)
        cells.push("");
    }
  }
  for (; i < cells.length; i++) {
    cells[i] = cells[i].trim().replace(/\\\|/g, "|");
  }
  return cells;
}
function rtrim(str, c, invert) {
  const l = str.length;
  if (l === 0) {
    return "";
  }
  let suffLen = 0;
  while (suffLen < l) {
    const currChar = str.charAt(l - suffLen - 1);
    if (currChar === c && true) {
      suffLen++;
    } else {
      break;
    }
  }
  return str.slice(0, l - suffLen);
}
function findClosingBracket(str, b) {
  if (str.indexOf(b[1]) === -1) {
    return -1;
  }
  let level = 0;
  for (let i = 0; i < str.length; i++) {
    if (str[i] === "\\") {
      i++;
    } else if (str[i] === b[0]) {
      level++;
    } else if (str[i] === b[1]) {
      level--;
      if (level < 0) {
        return i;
      }
    }
  }
  return -1;
}
function outputLink(cap, link2, raw, lexer) {
  const href = link2.href;
  const title = link2.title ? escape$1(link2.title) : null;
  const text = cap[1].replace(/\\([\[\]])/g, "$1");
  if (cap[0].charAt(0) !== "!") {
    lexer.state.inLink = true;
    const token = {
      type: "link",
      raw,
      href,
      title,
      text,
      tokens: lexer.inlineTokens(text)
    };
    lexer.state.inLink = false;
    return token;
  }
  return {
    type: "image",
    raw,
    href,
    title,
    text: escape$1(text)
  };
}
function indentCodeCompensation(raw, text) {
  const matchIndentToCode = raw.match(/^(\s+)(?:```)/);
  if (matchIndentToCode === null) {
    return text;
  }
  const indentToCode = matchIndentToCode[1];
  return text.split("\n").map((node) => {
    const matchIndentInNode = node.match(/^\s+/);
    if (matchIndentInNode === null) {
      return node;
    }
    const [indentInNode] = matchIndentInNode;
    if (indentInNode.length >= indentToCode.length) {
      return node.slice(indentToCode.length);
    }
    return node;
  }).join("\n");
}
class _Tokenizer {
  // set by the lexer
  constructor(options) {
    __publicField(this, "options");
    __publicField(this, "rules");
    // set by the lexer
    __publicField(this, "lexer");
    this.options = options || _defaults;
  }
  space(src) {
    const cap = this.rules.block.newline.exec(src);
    if (cap && cap[0].length > 0) {
      return {
        type: "space",
        raw: cap[0]
      };
    }
  }
  code(src) {
    const cap = this.rules.block.code.exec(src);
    if (cap) {
      const text = cap[0].replace(/^ {1,4}/gm, "");
      return {
        type: "code",
        raw: cap[0],
        codeBlockStyle: "indented",
        text: !this.options.pedantic ? rtrim(text, "\n") : text
      };
    }
  }
  fences(src) {
    const cap = this.rules.block.fences.exec(src);
    if (cap) {
      const raw = cap[0];
      const text = indentCodeCompensation(raw, cap[3] || "");
      return {
        type: "code",
        raw,
        lang: cap[2] ? cap[2].trim().replace(this.rules.inline.anyPunctuation, "$1") : cap[2],
        text
      };
    }
  }
  heading(src) {
    const cap = this.rules.block.heading.exec(src);
    if (cap) {
      let text = cap[2].trim();
      if (/#$/.test(text)) {
        const trimmed = rtrim(text, "#");
        if (this.options.pedantic) {
          text = trimmed.trim();
        } else if (!trimmed || / $/.test(trimmed)) {
          text = trimmed.trim();
        }
      }
      return {
        type: "heading",
        raw: cap[0],
        depth: cap[1].length,
        text,
        tokens: this.lexer.inline(text)
      };
    }
  }
  hr(src) {
    const cap = this.rules.block.hr.exec(src);
    if (cap) {
      return {
        type: "hr",
        raw: cap[0]
      };
    }
  }
  blockquote(src) {
    const cap = this.rules.block.blockquote.exec(src);
    if (cap) {
      const text = rtrim(cap[0].replace(/^ *>[ \t]?/gm, ""), "\n");
      const top = this.lexer.state.top;
      this.lexer.state.top = true;
      const tokens = this.lexer.blockTokens(text);
      this.lexer.state.top = top;
      return {
        type: "blockquote",
        raw: cap[0],
        tokens,
        text
      };
    }
  }
  list(src) {
    let cap = this.rules.block.list.exec(src);
    if (cap) {
      let bull = cap[1].trim();
      const isordered = bull.length > 1;
      const list2 = {
        type: "list",
        raw: "",
        ordered: isordered,
        start: isordered ? +bull.slice(0, -1) : "",
        loose: false,
        items: []
      };
      bull = isordered ? `\\d{1,9}\\${bull.slice(-1)}` : `\\${bull}`;
      if (this.options.pedantic) {
        bull = isordered ? bull : "[*+-]";
      }
      const itemRegex = new RegExp(`^( {0,3}${bull})((?:[	 ][^\\n]*)?(?:\\n|$))`);
      let raw = "";
      let itemContents = "";
      let endsWithBlankLine = false;
      while (src) {
        let endEarly = false;
        if (!(cap = itemRegex.exec(src))) {
          break;
        }
        if (this.rules.block.hr.test(src)) {
          break;
        }
        raw = cap[0];
        src = src.substring(raw.length);
        let line = cap[2].split("\n", 1)[0].replace(/^\t+/, (t) => " ".repeat(3 * t.length));
        let nextLine = src.split("\n", 1)[0];
        let indent = 0;
        if (this.options.pedantic) {
          indent = 2;
          itemContents = line.trimStart();
        } else {
          indent = cap[2].search(/[^ ]/);
          indent = indent > 4 ? 1 : indent;
          itemContents = line.slice(indent);
          indent += cap[1].length;
        }
        let blankLine = false;
        if (!line && /^ *$/.test(nextLine)) {
          raw += nextLine + "\n";
          src = src.substring(nextLine.length + 1);
          endEarly = true;
        }
        if (!endEarly) {
          const nextBulletRegex = new RegExp(`^ {0,${Math.min(3, indent - 1)}}(?:[*+-]|\\d{1,9}[.)])((?:[ 	][^\\n]*)?(?:\\n|$))`);
          const hrRegex = new RegExp(`^ {0,${Math.min(3, indent - 1)}}((?:- *){3,}|(?:_ *){3,}|(?:\\* *){3,})(?:\\n+|$)`);
          const fencesBeginRegex = new RegExp(`^ {0,${Math.min(3, indent - 1)}}(?:\`\`\`|~~~)`);
          const headingBeginRegex = new RegExp(`^ {0,${Math.min(3, indent - 1)}}#`);
          while (src) {
            const rawLine = src.split("\n", 1)[0];
            nextLine = rawLine;
            if (this.options.pedantic) {
              nextLine = nextLine.replace(/^ {1,4}(?=( {4})*[^ ])/g, "  ");
            }
            if (fencesBeginRegex.test(nextLine)) {
              break;
            }
            if (headingBeginRegex.test(nextLine)) {
              break;
            }
            if (nextBulletRegex.test(nextLine)) {
              break;
            }
            if (hrRegex.test(src)) {
              break;
            }
            if (nextLine.search(/[^ ]/) >= indent || !nextLine.trim()) {
              itemContents += "\n" + nextLine.slice(indent);
            } else {
              if (blankLine) {
                break;
              }
              if (line.search(/[^ ]/) >= 4) {
                break;
              }
              if (fencesBeginRegex.test(line)) {
                break;
              }
              if (headingBeginRegex.test(line)) {
                break;
              }
              if (hrRegex.test(line)) {
                break;
              }
              itemContents += "\n" + nextLine;
            }
            if (!blankLine && !nextLine.trim()) {
              blankLine = true;
            }
            raw += rawLine + "\n";
            src = src.substring(rawLine.length + 1);
            line = nextLine.slice(indent);
          }
        }
        if (!list2.loose) {
          if (endsWithBlankLine) {
            list2.loose = true;
          } else if (/\n *\n *$/.test(raw)) {
            endsWithBlankLine = true;
          }
        }
        let istask = null;
        let ischecked;
        if (this.options.gfm) {
          istask = /^\[[ xX]\] /.exec(itemContents);
          if (istask) {
            ischecked = istask[0] !== "[ ] ";
            itemContents = itemContents.replace(/^\[[ xX]\] +/, "");
          }
        }
        list2.items.push({
          type: "list_item",
          raw,
          task: !!istask,
          checked: ischecked,
          loose: false,
          text: itemContents,
          tokens: []
        });
        list2.raw += raw;
      }
      list2.items[list2.items.length - 1].raw = raw.trimEnd();
      list2.items[list2.items.length - 1].text = itemContents.trimEnd();
      list2.raw = list2.raw.trimEnd();
      for (let i = 0; i < list2.items.length; i++) {
        this.lexer.state.top = false;
        list2.items[i].tokens = this.lexer.blockTokens(list2.items[i].text, []);
        if (!list2.loose) {
          const spacers = list2.items[i].tokens.filter((t) => t.type === "space");
          const hasMultipleLineBreaks = spacers.length > 0 && spacers.some((t) => /\n.*\n/.test(t.raw));
          list2.loose = hasMultipleLineBreaks;
        }
      }
      if (list2.loose) {
        for (let i = 0; i < list2.items.length; i++) {
          list2.items[i].loose = true;
        }
      }
      return list2;
    }
  }
  html(src) {
    const cap = this.rules.block.html.exec(src);
    if (cap) {
      const token = {
        type: "html",
        block: true,
        raw: cap[0],
        pre: cap[1] === "pre" || cap[1] === "script" || cap[1] === "style",
        text: cap[0]
      };
      return token;
    }
  }
  def(src) {
    const cap = this.rules.block.def.exec(src);
    if (cap) {
      const tag2 = cap[1].toLowerCase().replace(/\s+/g, " ");
      const href = cap[2] ? cap[2].replace(/^<(.*)>$/, "$1").replace(this.rules.inline.anyPunctuation, "$1") : "";
      const title = cap[3] ? cap[3].substring(1, cap[3].length - 1).replace(this.rules.inline.anyPunctuation, "$1") : cap[3];
      return {
        type: "def",
        tag: tag2,
        raw: cap[0],
        href,
        title
      };
    }
  }
  table(src) {
    const cap = this.rules.block.table.exec(src);
    if (!cap) {
      return;
    }
    if (!/[:|]/.test(cap[2])) {
      return;
    }
    const headers = splitCells(cap[1]);
    const aligns = cap[2].replace(/^\||\| *$/g, "").split("|");
    const rows = cap[3] && cap[3].trim() ? cap[3].replace(/\n[ \t]*$/, "").split("\n") : [];
    const item = {
      type: "table",
      raw: cap[0],
      header: [],
      align: [],
      rows: []
    };
    if (headers.length !== aligns.length) {
      return;
    }
    for (const align of aligns) {
      if (/^ *-+: *$/.test(align)) {
        item.align.push("right");
      } else if (/^ *:-+: *$/.test(align)) {
        item.align.push("center");
      } else if (/^ *:-+ *$/.test(align)) {
        item.align.push("left");
      } else {
        item.align.push(null);
      }
    }
    for (const header of headers) {
      item.header.push({
        text: header,
        tokens: this.lexer.inline(header)
      });
    }
    for (const row of rows) {
      item.rows.push(splitCells(row, item.header.length).map((cell) => {
        return {
          text: cell,
          tokens: this.lexer.inline(cell)
        };
      }));
    }
    return item;
  }
  lheading(src) {
    const cap = this.rules.block.lheading.exec(src);
    if (cap) {
      return {
        type: "heading",
        raw: cap[0],
        depth: cap[2].charAt(0) === "=" ? 1 : 2,
        text: cap[1],
        tokens: this.lexer.inline(cap[1])
      };
    }
  }
  paragraph(src) {
    const cap = this.rules.block.paragraph.exec(src);
    if (cap) {
      const text = cap[1].charAt(cap[1].length - 1) === "\n" ? cap[1].slice(0, -1) : cap[1];
      return {
        type: "paragraph",
        raw: cap[0],
        text,
        tokens: this.lexer.inline(text)
      };
    }
  }
  text(src) {
    const cap = this.rules.block.text.exec(src);
    if (cap) {
      return {
        type: "text",
        raw: cap[0],
        text: cap[0],
        tokens: this.lexer.inline(cap[0])
      };
    }
  }
  escape(src) {
    const cap = this.rules.inline.escape.exec(src);
    if (cap) {
      return {
        type: "escape",
        raw: cap[0],
        text: escape$1(cap[1])
      };
    }
  }
  tag(src) {
    const cap = this.rules.inline.tag.exec(src);
    if (cap) {
      if (!this.lexer.state.inLink && /^<a /i.test(cap[0])) {
        this.lexer.state.inLink = true;
      } else if (this.lexer.state.inLink && /^<\/a>/i.test(cap[0])) {
        this.lexer.state.inLink = false;
      }
      if (!this.lexer.state.inRawBlock && /^<(pre|code|kbd|script)(\s|>)/i.test(cap[0])) {
        this.lexer.state.inRawBlock = true;
      } else if (this.lexer.state.inRawBlock && /^<\/(pre|code|kbd|script)(\s|>)/i.test(cap[0])) {
        this.lexer.state.inRawBlock = false;
      }
      return {
        type: "html",
        raw: cap[0],
        inLink: this.lexer.state.inLink,
        inRawBlock: this.lexer.state.inRawBlock,
        block: false,
        text: cap[0]
      };
    }
  }
  link(src) {
    const cap = this.rules.inline.link.exec(src);
    if (cap) {
      const trimmedUrl = cap[2].trim();
      if (!this.options.pedantic && /^</.test(trimmedUrl)) {
        if (!/>$/.test(trimmedUrl)) {
          return;
        }
        const rtrimSlash = rtrim(trimmedUrl.slice(0, -1), "\\");
        if ((trimmedUrl.length - rtrimSlash.length) % 2 === 0) {
          return;
        }
      } else {
        const lastParenIndex = findClosingBracket(cap[2], "()");
        if (lastParenIndex > -1) {
          const start = cap[0].indexOf("!") === 0 ? 5 : 4;
          const linkLen = start + cap[1].length + lastParenIndex;
          cap[2] = cap[2].substring(0, lastParenIndex);
          cap[0] = cap[0].substring(0, linkLen).trim();
          cap[3] = "";
        }
      }
      let href = cap[2];
      let title = "";
      if (this.options.pedantic) {
        const link2 = /^([^'"]*[^\s])\s+(['"])(.*)\2/.exec(href);
        if (link2) {
          href = link2[1];
          title = link2[3];
        }
      } else {
        title = cap[3] ? cap[3].slice(1, -1) : "";
      }
      href = href.trim();
      if (/^</.test(href)) {
        if (this.options.pedantic && !/>$/.test(trimmedUrl)) {
          href = href.slice(1);
        } else {
          href = href.slice(1, -1);
        }
      }
      return outputLink(cap, {
        href: href ? href.replace(this.rules.inline.anyPunctuation, "$1") : href,
        title: title ? title.replace(this.rules.inline.anyPunctuation, "$1") : title
      }, cap[0], this.lexer);
    }
  }
  reflink(src, links) {
    let cap;
    if ((cap = this.rules.inline.reflink.exec(src)) || (cap = this.rules.inline.nolink.exec(src))) {
      const linkString = (cap[2] || cap[1]).replace(/\s+/g, " ");
      const link2 = links[linkString.toLowerCase()];
      if (!link2) {
        const text = cap[0].charAt(0);
        return {
          type: "text",
          raw: text,
          text
        };
      }
      return outputLink(cap, link2, cap[0], this.lexer);
    }
  }
  emStrong(src, maskedSrc, prevChar = "") {
    let match = this.rules.inline.emStrongLDelim.exec(src);
    if (!match)
      return;
    if (match[3] && prevChar.match(/[\p{L}\p{N}]/u))
      return;
    const nextChar = match[1] || match[2] || "";
    if (!nextChar || !prevChar || this.rules.inline.punctuation.exec(prevChar)) {
      const lLength = [...match[0]].length - 1;
      let rDelim, rLength, delimTotal = lLength, midDelimTotal = 0;
      const endReg = match[0][0] === "*" ? this.rules.inline.emStrongRDelimAst : this.rules.inline.emStrongRDelimUnd;
      endReg.lastIndex = 0;
      maskedSrc = maskedSrc.slice(-1 * src.length + lLength);
      while ((match = endReg.exec(maskedSrc)) != null) {
        rDelim = match[1] || match[2] || match[3] || match[4] || match[5] || match[6];
        if (!rDelim)
          continue;
        rLength = [...rDelim].length;
        if (match[3] || match[4]) {
          delimTotal += rLength;
          continue;
        } else if (match[5] || match[6]) {
          if (lLength % 3 && !((lLength + rLength) % 3)) {
            midDelimTotal += rLength;
            continue;
          }
        }
        delimTotal -= rLength;
        if (delimTotal > 0)
          continue;
        rLength = Math.min(rLength, rLength + delimTotal + midDelimTotal);
        const lastCharLength = [...match[0]][0].length;
        const raw = src.slice(0, lLength + match.index + lastCharLength + rLength);
        if (Math.min(lLength, rLength) % 2) {
          const text2 = raw.slice(1, -1);
          return {
            type: "em",
            raw,
            text: text2,
            tokens: this.lexer.inlineTokens(text2)
          };
        }
        const text = raw.slice(2, -2);
        return {
          type: "strong",
          raw,
          text,
          tokens: this.lexer.inlineTokens(text)
        };
      }
    }
  }
  codespan(src) {
    const cap = this.rules.inline.code.exec(src);
    if (cap) {
      let text = cap[2].replace(/\n/g, " ");
      const hasNonSpaceChars = /[^ ]/.test(text);
      const hasSpaceCharsOnBothEnds = /^ /.test(text) && / $/.test(text);
      if (hasNonSpaceChars && hasSpaceCharsOnBothEnds) {
        text = text.substring(1, text.length - 1);
      }
      text = escape$1(text, true);
      return {
        type: "codespan",
        raw: cap[0],
        text
      };
    }
  }
  br(src) {
    const cap = this.rules.inline.br.exec(src);
    if (cap) {
      return {
        type: "br",
        raw: cap[0]
      };
    }
  }
  del(src) {
    const cap = this.rules.inline.del.exec(src);
    if (cap) {
      return {
        type: "del",
        raw: cap[0],
        text: cap[2],
        tokens: this.lexer.inlineTokens(cap[2])
      };
    }
  }
  autolink(src) {
    const cap = this.rules.inline.autolink.exec(src);
    if (cap) {
      let text, href;
      if (cap[2] === "@") {
        text = escape$1(cap[1]);
        href = "mailto:" + text;
      } else {
        text = escape$1(cap[1]);
        href = text;
      }
      return {
        type: "link",
        raw: cap[0],
        text,
        href,
        tokens: [
          {
            type: "text",
            raw: text,
            text
          }
        ]
      };
    }
  }
  url(src) {
    var _a;
    let cap;
    if (cap = this.rules.inline.url.exec(src)) {
      let text, href;
      if (cap[2] === "@") {
        text = escape$1(cap[0]);
        href = "mailto:" + text;
      } else {
        let prevCapZero;
        do {
          prevCapZero = cap[0];
          cap[0] = ((_a = this.rules.inline._backpedal.exec(cap[0])) == null ? void 0 : _a[0]) ?? "";
        } while (prevCapZero !== cap[0]);
        text = escape$1(cap[0]);
        if (cap[1] === "www.") {
          href = "http://" + cap[0];
        } else {
          href = cap[0];
        }
      }
      return {
        type: "link",
        raw: cap[0],
        text,
        href,
        tokens: [
          {
            type: "text",
            raw: text,
            text
          }
        ]
      };
    }
  }
  inlineText(src) {
    const cap = this.rules.inline.text.exec(src);
    if (cap) {
      let text;
      if (this.lexer.state.inRawBlock) {
        text = cap[0];
      } else {
        text = escape$1(cap[0]);
      }
      return {
        type: "text",
        raw: cap[0],
        text
      };
    }
  }
}
const newline = /^(?: *(?:\n|$))+/;
const blockCode = /^( {4}[^\n]+(?:\n(?: *(?:\n|$))*)?)+/;
const fences = /^ {0,3}(`{3,}(?=[^`\n]*(?:\n|$))|~{3,})([^\n]*)(?:\n|$)(?:|([\s\S]*?)(?:\n|$))(?: {0,3}\1[~`]* *(?=\n|$)|$)/;
const hr = /^ {0,3}((?:-[\t ]*){3,}|(?:_[ \t]*){3,}|(?:\*[ \t]*){3,})(?:\n+|$)/;
const heading = /^ {0,3}(#{1,6})(?=\s|$)(.*)(?:\n+|$)/;
const bullet = /(?:[*+-]|\d{1,9}[.)])/;
const lheading = edit(/^(?!bull )((?:.|\n(?!\s*?\n|bull ))+?)\n {0,3}(=+|-+) *(?:\n+|$)/).replace(/bull/g, bullet).getRegex();
const _paragraph = /^([^\n]+(?:\n(?!hr|heading|lheading|blockquote|fences|list|html|table| +\n)[^\n]+)*)/;
const blockText = /^[^\n]+/;
const _blockLabel = /(?!\s*\])(?:\\.|[^\[\]\\])+/;
const def = edit(/^ {0,3}\[(label)\]: *(?:\n *)?([^<\s][^\s]*|<.*?>)(?:(?: +(?:\n *)?| *\n *)(title))? *(?:\n+|$)/).replace("label", _blockLabel).replace("title", /(?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))/).getRegex();
const list = edit(/^( {0,3}bull)([ \t][^\n]+?)?(?:\n|$)/).replace(/bull/g, bullet).getRegex();
const _tag = "address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul";
const _comment = /<!--(?!-?>)[\s\S]*?(?:-->|$)/;
const html = edit("^ {0,3}(?:<(script|pre|style|textarea)[\\s>][\\s\\S]*?(?:</\\1>[^\\n]*\\n+|$)|comment[^\\n]*(\\n+|$)|<\\?[\\s\\S]*?(?:\\?>\\n*|$)|<![A-Z][\\s\\S]*?(?:>\\n*|$)|<!\\[CDATA\\[[\\s\\S]*?(?:\\]\\]>\\n*|$)|</?(tag)(?: +|\\n|/?>)[\\s\\S]*?(?:(?:\\n *)+\\n|$)|<(?!script|pre|style|textarea)([a-z][\\w-]*)(?:attribute)*? */?>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n *)+\\n|$)|</(?!script|pre|style|textarea)[a-z][\\w-]*\\s*>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n *)+\\n|$))", "i").replace("comment", _comment).replace("tag", _tag).replace("attribute", / +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?/).getRegex();
const paragraph = edit(_paragraph).replace("hr", hr).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("|lheading", "").replace("|table", "").replace("blockquote", " {0,3}>").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", _tag).getRegex();
const blockquote = edit(/^( {0,3}> ?(paragraph|[^\n]*)(?:\n|$))+/).replace("paragraph", paragraph).getRegex();
const blockNormal = {
  blockquote,
  code: blockCode,
  def,
  fences,
  heading,
  hr,
  html,
  lheading,
  list,
  newline,
  paragraph,
  table: noopTest,
  text: blockText
};
const gfmTable = edit("^ *([^\\n ].*)\\n {0,3}((?:\\| *)?:?-+:? *(?:\\| *:?-+:? *)*(?:\\| *)?)(?:\\n((?:(?! *\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\n|$))*)\\n*|$)").replace("hr", hr).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("blockquote", " {0,3}>").replace("code", " {4}[^\\n]").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", _tag).getRegex();
const blockGfm = {
  ...blockNormal,
  table: gfmTable,
  paragraph: edit(_paragraph).replace("hr", hr).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("|lheading", "").replace("table", gfmTable).replace("blockquote", " {0,3}>").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)]) ").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", _tag).getRegex()
};
const blockPedantic = {
  ...blockNormal,
  html: edit(`^ *(?:comment *(?:\\n|\\s*$)|<(tag)[\\s\\S]+?</\\1> *(?:\\n{2,}|\\s*$)|<tag(?:"[^"]*"|'[^']*'|\\s[^'"/>\\s]*)*?/?> *(?:\\n{2,}|\\s*$))`).replace("comment", _comment).replace(/tag/g, "(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\b)\\w+(?!:|[^\\w\\s@]*@)\\b").getRegex(),
  def: /^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +(["(][^\n]+[")]))? *(?:\n+|$)/,
  heading: /^(#{1,6})(.*)(?:\n+|$)/,
  fences: noopTest,
  // fences not supported
  lheading: /^(.+?)\n {0,3}(=+|-+) *(?:\n+|$)/,
  paragraph: edit(_paragraph).replace("hr", hr).replace("heading", " *#{1,6} *[^\n]").replace("lheading", lheading).replace("|table", "").replace("blockquote", " {0,3}>").replace("|fences", "").replace("|list", "").replace("|html", "").replace("|tag", "").getRegex()
};
const escape = /^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/;
const inlineCode = /^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/;
const br = /^( {2,}|\\)\n(?!\s*$)/;
const inlineText = /^(`+|[^`])(?:(?= {2,}\n)|[\s\S]*?(?:(?=[\\<!\[`*_]|\b_|$)|[^ ](?= {2,}\n)))/;
const _punctuation = "\\p{P}$+<=>`^|~";
const punctuation = edit(/^((?![*_])[\spunctuation])/, "u").replace(/punctuation/g, _punctuation).getRegex();
const blockSkip = /\[[^[\]]*?\]\([^\(\)]*?\)|`[^`]*?`|<[^<>]*?>/g;
const emStrongLDelim = edit(/^(?:\*+(?:((?!\*)[punct])|[^\s*]))|^_+(?:((?!_)[punct])|([^\s_]))/, "u").replace(/punct/g, _punctuation).getRegex();
const emStrongRDelimAst = edit("^[^_*]*?__[^_*]*?\\*[^_*]*?(?=__)|[^*]+(?=[^*])|(?!\\*)[punct](\\*+)(?=[\\s]|$)|[^punct\\s](\\*+)(?!\\*)(?=[punct\\s]|$)|(?!\\*)[punct\\s](\\*+)(?=[^punct\\s])|[\\s](\\*+)(?!\\*)(?=[punct])|(?!\\*)[punct](\\*+)(?!\\*)(?=[punct])|[^punct\\s](\\*+)(?=[^punct\\s])", "gu").replace(/punct/g, _punctuation).getRegex();
const emStrongRDelimUnd = edit("^[^_*]*?\\*\\*[^_*]*?_[^_*]*?(?=\\*\\*)|[^_]+(?=[^_])|(?!_)[punct](_+)(?=[\\s]|$)|[^punct\\s](_+)(?!_)(?=[punct\\s]|$)|(?!_)[punct\\s](_+)(?=[^punct\\s])|[\\s](_+)(?!_)(?=[punct])|(?!_)[punct](_+)(?!_)(?=[punct])", "gu").replace(/punct/g, _punctuation).getRegex();
const anyPunctuation = edit(/\\([punct])/, "gu").replace(/punct/g, _punctuation).getRegex();
const autolink = edit(/^<(scheme:[^\s\x00-\x1f<>]*|email)>/).replace("scheme", /[a-zA-Z][a-zA-Z0-9+.-]{1,31}/).replace("email", /[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/).getRegex();
const _inlineComment = edit(_comment).replace("(?:-->|$)", "-->").getRegex();
const tag = edit("^comment|^</[a-zA-Z][\\w:-]*\\s*>|^<[a-zA-Z][\\w-]*(?:attribute)*?\\s*/?>|^<\\?[\\s\\S]*?\\?>|^<![a-zA-Z]+\\s[\\s\\S]*?>|^<!\\[CDATA\\[[\\s\\S]*?\\]\\]>").replace("comment", _inlineComment).replace("attribute", /\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?/).getRegex();
const _inlineLabel = /(?:\[(?:\\.|[^\[\]\\])*\]|\\.|`[^`]*`|[^\[\]\\`])*?/;
const link = edit(/^!?\[(label)\]\(\s*(href)(?:\s+(title))?\s*\)/).replace("label", _inlineLabel).replace("href", /<(?:\\.|[^\n<>\\])+>|[^\s\x00-\x1f]*/).replace("title", /"(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)/).getRegex();
const reflink = edit(/^!?\[(label)\]\[(ref)\]/).replace("label", _inlineLabel).replace("ref", _blockLabel).getRegex();
const nolink = edit(/^!?\[(ref)\](?:\[\])?/).replace("ref", _blockLabel).getRegex();
const reflinkSearch = edit("reflink|nolink(?!\\()", "g").replace("reflink", reflink).replace("nolink", nolink).getRegex();
const inlineNormal = {
  _backpedal: noopTest,
  // only used for GFM url
  anyPunctuation,
  autolink,
  blockSkip,
  br,
  code: inlineCode,
  del: noopTest,
  emStrongLDelim,
  emStrongRDelimAst,
  emStrongRDelimUnd,
  escape,
  link,
  nolink,
  punctuation,
  reflink,
  reflinkSearch,
  tag,
  text: inlineText,
  url: noopTest
};
const inlinePedantic = {
  ...inlineNormal,
  link: edit(/^!?\[(label)\]\((.*?)\)/).replace("label", _inlineLabel).getRegex(),
  reflink: edit(/^!?\[(label)\]\s*\[([^\]]*)\]/).replace("label", _inlineLabel).getRegex()
};
const inlineGfm = {
  ...inlineNormal,
  escape: edit(escape).replace("])", "~|])").getRegex(),
  url: edit(/^((?:ftp|https?):\/\/|www\.)(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*|^email/, "i").replace("email", /[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/).getRegex(),
  _backpedal: /(?:[^?!.,:;*_'"~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_'"~)]+(?!$))+/,
  del: /^(~~?)(?=[^\s~])([\s\S]*?[^\s~])\1(?=[^~]|$)/,
  text: /^([`~]+|[^`~])(?:(?= {2,}\n)|(?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)|[\s\S]*?(?:(?=[\\<!\[`*~_]|\b_|https?:\/\/|ftp:\/\/|www\.|$)|[^ ](?= {2,}\n)|[^a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-](?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)))/
};
const inlineBreaks = {
  ...inlineGfm,
  br: edit(br).replace("{2,}", "*").getRegex(),
  text: edit(inlineGfm.text).replace("\\b_", "\\b_| {2,}\\n").replace(/\{2,\}/g, "*").getRegex()
};
const block = {
  normal: blockNormal,
  gfm: blockGfm,
  pedantic: blockPedantic
};
const inline = {
  normal: inlineNormal,
  gfm: inlineGfm,
  breaks: inlineBreaks,
  pedantic: inlinePedantic
};
class _Lexer {
  constructor(options) {
    __publicField(this, "tokens");
    __publicField(this, "options");
    __publicField(this, "state");
    __publicField(this, "tokenizer");
    __publicField(this, "inlineQueue");
    this.tokens = [];
    this.tokens.links = /* @__PURE__ */ Object.create(null);
    this.options = options || _defaults;
    this.options.tokenizer = this.options.tokenizer || new _Tokenizer();
    this.tokenizer = this.options.tokenizer;
    this.tokenizer.options = this.options;
    this.tokenizer.lexer = this;
    this.inlineQueue = [];
    this.state = {
      inLink: false,
      inRawBlock: false,
      top: true
    };
    const rules = {
      block: block.normal,
      inline: inline.normal
    };
    if (this.options.pedantic) {
      rules.block = block.pedantic;
      rules.inline = inline.pedantic;
    } else if (this.options.gfm) {
      rules.block = block.gfm;
      if (this.options.breaks) {
        rules.inline = inline.breaks;
      } else {
        rules.inline = inline.gfm;
      }
    }
    this.tokenizer.rules = rules;
  }
  /**
   * Expose Rules
   */
  static get rules() {
    return {
      block,
      inline
    };
  }
  /**
   * Static Lex Method
   */
  static lex(src, options) {
    const lexer = new _Lexer(options);
    return lexer.lex(src);
  }
  /**
   * Static Lex Inline Method
   */
  static lexInline(src, options) {
    const lexer = new _Lexer(options);
    return lexer.inlineTokens(src);
  }
  /**
   * Preprocessing
   */
  lex(src) {
    src = src.replace(/\r\n|\r/g, "\n");
    this.blockTokens(src, this.tokens);
    let next;
    while (next = this.inlineQueue.shift()) {
      this.inlineTokens(next.src, next.tokens);
    }
    return this.tokens;
  }
  blockTokens(src, tokens = []) {
    if (this.options.pedantic) {
      src = src.replace(/\t/g, "    ").replace(/^ +$/gm, "");
    } else {
      src = src.replace(/^( *)(\t+)/gm, (_, leading, tabs) => {
        return leading + "    ".repeat(tabs.length);
      });
    }
    let token;
    let lastToken;
    let cutSrc;
    let lastParagraphClipped;
    while (src) {
      if (this.options.extensions && this.options.extensions.block && this.options.extensions.block.some((extTokenizer) => {
        if (token = extTokenizer.call({ lexer: this }, src, tokens)) {
          src = src.substring(token.raw.length);
          tokens.push(token);
          return true;
        }
        return false;
      })) {
        continue;
      }
      if (token = this.tokenizer.space(src)) {
        src = src.substring(token.raw.length);
        if (token.raw.length === 1 && tokens.length > 0) {
          tokens[tokens.length - 1].raw += "\n";
        } else {
          tokens.push(token);
        }
        continue;
      }
      if (token = this.tokenizer.code(src)) {
        src = src.substring(token.raw.length);
        lastToken = tokens[tokens.length - 1];
        if (lastToken && (lastToken.type === "paragraph" || lastToken.type === "text")) {
          lastToken.raw += "\n" + token.raw;
          lastToken.text += "\n" + token.text;
          this.inlineQueue[this.inlineQueue.length - 1].src = lastToken.text;
        } else {
          tokens.push(token);
        }
        continue;
      }
      if (token = this.tokenizer.fences(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.heading(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.hr(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.blockquote(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.list(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.html(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.def(src)) {
        src = src.substring(token.raw.length);
        lastToken = tokens[tokens.length - 1];
        if (lastToken && (lastToken.type === "paragraph" || lastToken.type === "text")) {
          lastToken.raw += "\n" + token.raw;
          lastToken.text += "\n" + token.raw;
          this.inlineQueue[this.inlineQueue.length - 1].src = lastToken.text;
        } else if (!this.tokens.links[token.tag]) {
          this.tokens.links[token.tag] = {
            href: token.href,
            title: token.title
          };
        }
        continue;
      }
      if (token = this.tokenizer.table(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.lheading(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      cutSrc = src;
      if (this.options.extensions && this.options.extensions.startBlock) {
        let startIndex = Infinity;
        const tempSrc = src.slice(1);
        let tempStart;
        this.options.extensions.startBlock.forEach((getStartIndex) => {
          tempStart = getStartIndex.call({ lexer: this }, tempSrc);
          if (typeof tempStart === "number" && tempStart >= 0) {
            startIndex = Math.min(startIndex, tempStart);
          }
        });
        if (startIndex < Infinity && startIndex >= 0) {
          cutSrc = src.substring(0, startIndex + 1);
        }
      }
      if (this.state.top && (token = this.tokenizer.paragraph(cutSrc))) {
        lastToken = tokens[tokens.length - 1];
        if (lastParagraphClipped && lastToken.type === "paragraph") {
          lastToken.raw += "\n" + token.raw;
          lastToken.text += "\n" + token.text;
          this.inlineQueue.pop();
          this.inlineQueue[this.inlineQueue.length - 1].src = lastToken.text;
        } else {
          tokens.push(token);
        }
        lastParagraphClipped = cutSrc.length !== src.length;
        src = src.substring(token.raw.length);
        continue;
      }
      if (token = this.tokenizer.text(src)) {
        src = src.substring(token.raw.length);
        lastToken = tokens[tokens.length - 1];
        if (lastToken && lastToken.type === "text") {
          lastToken.raw += "\n" + token.raw;
          lastToken.text += "\n" + token.text;
          this.inlineQueue.pop();
          this.inlineQueue[this.inlineQueue.length - 1].src = lastToken.text;
        } else {
          tokens.push(token);
        }
        continue;
      }
      if (src) {
        const errMsg = "Infinite loop on byte: " + src.charCodeAt(0);
        if (this.options.silent) {
          console.error(errMsg);
          break;
        } else {
          throw new Error(errMsg);
        }
      }
    }
    this.state.top = true;
    return tokens;
  }
  inline(src, tokens = []) {
    this.inlineQueue.push({ src, tokens });
    return tokens;
  }
  /**
   * Lexing/Compiling
   */
  inlineTokens(src, tokens = []) {
    let token, lastToken, cutSrc;
    let maskedSrc = src;
    let match;
    let keepPrevChar, prevChar;
    if (this.tokens.links) {
      const links = Object.keys(this.tokens.links);
      if (links.length > 0) {
        while ((match = this.tokenizer.rules.inline.reflinkSearch.exec(maskedSrc)) != null) {
          if (links.includes(match[0].slice(match[0].lastIndexOf("[") + 1, -1))) {
            maskedSrc = maskedSrc.slice(0, match.index) + "[" + "a".repeat(match[0].length - 2) + "]" + maskedSrc.slice(this.tokenizer.rules.inline.reflinkSearch.lastIndex);
          }
        }
      }
    }
    while ((match = this.tokenizer.rules.inline.blockSkip.exec(maskedSrc)) != null) {
      maskedSrc = maskedSrc.slice(0, match.index) + "[" + "a".repeat(match[0].length - 2) + "]" + maskedSrc.slice(this.tokenizer.rules.inline.blockSkip.lastIndex);
    }
    while ((match = this.tokenizer.rules.inline.anyPunctuation.exec(maskedSrc)) != null) {
      maskedSrc = maskedSrc.slice(0, match.index) + "++" + maskedSrc.slice(this.tokenizer.rules.inline.anyPunctuation.lastIndex);
    }
    while (src) {
      if (!keepPrevChar) {
        prevChar = "";
      }
      keepPrevChar = false;
      if (this.options.extensions && this.options.extensions.inline && this.options.extensions.inline.some((extTokenizer) => {
        if (token = extTokenizer.call({ lexer: this }, src, tokens)) {
          src = src.substring(token.raw.length);
          tokens.push(token);
          return true;
        }
        return false;
      })) {
        continue;
      }
      if (token = this.tokenizer.escape(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.tag(src)) {
        src = src.substring(token.raw.length);
        lastToken = tokens[tokens.length - 1];
        if (lastToken && token.type === "text" && lastToken.type === "text") {
          lastToken.raw += token.raw;
          lastToken.text += token.text;
        } else {
          tokens.push(token);
        }
        continue;
      }
      if (token = this.tokenizer.link(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.reflink(src, this.tokens.links)) {
        src = src.substring(token.raw.length);
        lastToken = tokens[tokens.length - 1];
        if (lastToken && token.type === "text" && lastToken.type === "text") {
          lastToken.raw += token.raw;
          lastToken.text += token.text;
        } else {
          tokens.push(token);
        }
        continue;
      }
      if (token = this.tokenizer.emStrong(src, maskedSrc, prevChar)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.codespan(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.br(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.del(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (token = this.tokenizer.autolink(src)) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      if (!this.state.inLink && (token = this.tokenizer.url(src))) {
        src = src.substring(token.raw.length);
        tokens.push(token);
        continue;
      }
      cutSrc = src;
      if (this.options.extensions && this.options.extensions.startInline) {
        let startIndex = Infinity;
        const tempSrc = src.slice(1);
        let tempStart;
        this.options.extensions.startInline.forEach((getStartIndex) => {
          tempStart = getStartIndex.call({ lexer: this }, tempSrc);
          if (typeof tempStart === "number" && tempStart >= 0) {
            startIndex = Math.min(startIndex, tempStart);
          }
        });
        if (startIndex < Infinity && startIndex >= 0) {
          cutSrc = src.substring(0, startIndex + 1);
        }
      }
      if (token = this.tokenizer.inlineText(cutSrc)) {
        src = src.substring(token.raw.length);
        if (token.raw.slice(-1) !== "_") {
          prevChar = token.raw.slice(-1);
        }
        keepPrevChar = true;
        lastToken = tokens[tokens.length - 1];
        if (lastToken && lastToken.type === "text") {
          lastToken.raw += token.raw;
          lastToken.text += token.text;
        } else {
          tokens.push(token);
        }
        continue;
      }
      if (src) {
        const errMsg = "Infinite loop on byte: " + src.charCodeAt(0);
        if (this.options.silent) {
          console.error(errMsg);
          break;
        } else {
          throw new Error(errMsg);
        }
      }
    }
    return tokens;
  }
}
class _Renderer {
  constructor(options) {
    __publicField(this, "options");
    this.options = options || _defaults;
  }
  code(code, infostring, escaped) {
    var _a;
    const lang = (_a = (infostring || "").match(/^\S*/)) == null ? void 0 : _a[0];
    code = code.replace(/\n$/, "") + "\n";
    if (!lang) {
      return "<pre><code>" + (escaped ? code : escape$1(code, true)) + "</code></pre>\n";
    }
    return '<pre><code class="language-' + escape$1(lang) + '">' + (escaped ? code : escape$1(code, true)) + "</code></pre>\n";
  }
  blockquote(quote) {
    return `<blockquote>
${quote}</blockquote>
`;
  }
  html(html2, block2) {
    return html2;
  }
  heading(text, level, raw) {
    return `<h${level}>${text}</h${level}>
`;
  }
  hr() {
    return "<hr>\n";
  }
  list(body, ordered, start) {
    const type = ordered ? "ol" : "ul";
    const startatt = ordered && start !== 1 ? ' start="' + start + '"' : "";
    return "<" + type + startatt + ">\n" + body + "</" + type + ">\n";
  }
  listitem(text, task, checked) {
    return `<li>${text}</li>
`;
  }
  checkbox(checked) {
    return "<input " + (checked ? 'checked="" ' : "") + 'disabled="" type="checkbox">';
  }
  paragraph(text) {
    return `<p>${text}</p>
`;
  }
  table(header, body) {
    if (body)
      body = `<tbody>${body}</tbody>`;
    return "<table>\n<thead>\n" + header + "</thead>\n" + body + "</table>\n";
  }
  tablerow(content) {
    return `<tr>
${content}</tr>
`;
  }
  tablecell(content, flags) {
    const type = flags.header ? "th" : "td";
    const tag2 = flags.align ? `<${type} align="${flags.align}">` : `<${type}>`;
    return tag2 + content + `</${type}>
`;
  }
  /**
   * span level renderer
   */
  strong(text) {
    return `<strong>${text}</strong>`;
  }
  em(text) {
    return `<em>${text}</em>`;
  }
  codespan(text) {
    return `<code>${text}</code>`;
  }
  br() {
    return "<br>";
  }
  del(text) {
    return `<del>${text}</del>`;
  }
  link(href, title, text) {
    const cleanHref = cleanUrl(href);
    if (cleanHref === null) {
      return text;
    }
    href = cleanHref;
    let out = '<a href="' + href + '"';
    if (title) {
      out += ' title="' + title + '"';
    }
    out += ">" + text + "</a>";
    return out;
  }
  image(href, title, text) {
    const cleanHref = cleanUrl(href);
    if (cleanHref === null) {
      return text;
    }
    href = cleanHref;
    let out = `<img src="${href}" alt="${text}"`;
    if (title) {
      out += ` title="${title}"`;
    }
    out += ">";
    return out;
  }
  text(text) {
    return text;
  }
}
class _TextRenderer {
  // no need for block level renderers
  strong(text) {
    return text;
  }
  em(text) {
    return text;
  }
  codespan(text) {
    return text;
  }
  del(text) {
    return text;
  }
  html(text) {
    return text;
  }
  text(text) {
    return text;
  }
  link(href, title, text) {
    return "" + text;
  }
  image(href, title, text) {
    return "" + text;
  }
  br() {
    return "";
  }
}
class _Parser {
  constructor(options) {
    __publicField(this, "options");
    __publicField(this, "renderer");
    __publicField(this, "textRenderer");
    this.options = options || _defaults;
    this.options.renderer = this.options.renderer || new _Renderer();
    this.renderer = this.options.renderer;
    this.renderer.options = this.options;
    this.textRenderer = new _TextRenderer();
  }
  /**
   * Static Parse Method
   */
  static parse(tokens, options) {
    const parser = new _Parser(options);
    return parser.parse(tokens);
  }
  /**
   * Static Parse Inline Method
   */
  static parseInline(tokens, options) {
    const parser = new _Parser(options);
    return parser.parseInline(tokens);
  }
  /**
   * Parse Loop
   */
  parse(tokens, top = true) {
    let out = "";
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      if (this.options.extensions && this.options.extensions.renderers && this.options.extensions.renderers[token.type]) {
        const genericToken = token;
        const ret = this.options.extensions.renderers[genericToken.type].call({ parser: this }, genericToken);
        if (ret !== false || !["space", "hr", "heading", "code", "table", "blockquote", "list", "html", "paragraph", "text"].includes(genericToken.type)) {
          out += ret || "";
          continue;
        }
      }
      switch (token.type) {
        case "space": {
          continue;
        }
        case "hr": {
          out += this.renderer.hr();
          continue;
        }
        case "heading": {
          const headingToken = token;
          out += this.renderer.heading(this.parseInline(headingToken.tokens), headingToken.depth, unescape(this.parseInline(headingToken.tokens, this.textRenderer)));
          continue;
        }
        case "code": {
          const codeToken = token;
          out += this.renderer.code(codeToken.text, codeToken.lang, !!codeToken.escaped);
          continue;
        }
        case "table": {
          const tableToken = token;
          let header = "";
          let cell = "";
          for (let j = 0; j < tableToken.header.length; j++) {
            cell += this.renderer.tablecell(this.parseInline(tableToken.header[j].tokens), { header: true, align: tableToken.align[j] });
          }
          header += this.renderer.tablerow(cell);
          let body = "";
          for (let j = 0; j < tableToken.rows.length; j++) {
            const row = tableToken.rows[j];
            cell = "";
            for (let k = 0; k < row.length; k++) {
              cell += this.renderer.tablecell(this.parseInline(row[k].tokens), { header: false, align: tableToken.align[k] });
            }
            body += this.renderer.tablerow(cell);
          }
          out += this.renderer.table(header, body);
          continue;
        }
        case "blockquote": {
          const blockquoteToken = token;
          const body = this.parse(blockquoteToken.tokens);
          out += this.renderer.blockquote(body);
          continue;
        }
        case "list": {
          const listToken = token;
          const ordered = listToken.ordered;
          const start = listToken.start;
          const loose = listToken.loose;
          let body = "";
          for (let j = 0; j < listToken.items.length; j++) {
            const item = listToken.items[j];
            const checked = item.checked;
            const task = item.task;
            let itemBody = "";
            if (item.task) {
              const checkbox = this.renderer.checkbox(!!checked);
              if (loose) {
                if (item.tokens.length > 0 && item.tokens[0].type === "paragraph") {
                  item.tokens[0].text = checkbox + " " + item.tokens[0].text;
                  if (item.tokens[0].tokens && item.tokens[0].tokens.length > 0 && item.tokens[0].tokens[0].type === "text") {
                    item.tokens[0].tokens[0].text = checkbox + " " + item.tokens[0].tokens[0].text;
                  }
                } else {
                  item.tokens.unshift({
                    type: "text",
                    text: checkbox + " "
                  });
                }
              } else {
                itemBody += checkbox + " ";
              }
            }
            itemBody += this.parse(item.tokens, loose);
            body += this.renderer.listitem(itemBody, task, !!checked);
          }
          out += this.renderer.list(body, ordered, start);
          continue;
        }
        case "html": {
          const htmlToken = token;
          out += this.renderer.html(htmlToken.text, htmlToken.block);
          continue;
        }
        case "paragraph": {
          const paragraphToken = token;
          out += this.renderer.paragraph(this.parseInline(paragraphToken.tokens));
          continue;
        }
        case "text": {
          let textToken = token;
          let body = textToken.tokens ? this.parseInline(textToken.tokens) : textToken.text;
          while (i + 1 < tokens.length && tokens[i + 1].type === "text") {
            textToken = tokens[++i];
            body += "\n" + (textToken.tokens ? this.parseInline(textToken.tokens) : textToken.text);
          }
          out += top ? this.renderer.paragraph(body) : body;
          continue;
        }
        default: {
          const errMsg = 'Token with "' + token.type + '" type was not found.';
          if (this.options.silent) {
            console.error(errMsg);
            return "";
          } else {
            throw new Error(errMsg);
          }
        }
      }
    }
    return out;
  }
  /**
   * Parse Inline Tokens
   */
  parseInline(tokens, renderer) {
    renderer = renderer || this.renderer;
    let out = "";
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      if (this.options.extensions && this.options.extensions.renderers && this.options.extensions.renderers[token.type]) {
        const ret = this.options.extensions.renderers[token.type].call({ parser: this }, token);
        if (ret !== false || !["escape", "html", "link", "image", "strong", "em", "codespan", "br", "del", "text"].includes(token.type)) {
          out += ret || "";
          continue;
        }
      }
      switch (token.type) {
        case "escape": {
          const escapeToken = token;
          out += renderer.text(escapeToken.text);
          break;
        }
        case "html": {
          const tagToken = token;
          out += renderer.html(tagToken.text);
          break;
        }
        case "link": {
          const linkToken = token;
          out += renderer.link(linkToken.href, linkToken.title, this.parseInline(linkToken.tokens, renderer));
          break;
        }
        case "image": {
          const imageToken = token;
          out += renderer.image(imageToken.href, imageToken.title, imageToken.text);
          break;
        }
        case "strong": {
          const strongToken = token;
          out += renderer.strong(this.parseInline(strongToken.tokens, renderer));
          break;
        }
        case "em": {
          const emToken = token;
          out += renderer.em(this.parseInline(emToken.tokens, renderer));
          break;
        }
        case "codespan": {
          const codespanToken = token;
          out += renderer.codespan(codespanToken.text);
          break;
        }
        case "br": {
          out += renderer.br();
          break;
        }
        case "del": {
          const delToken = token;
          out += renderer.del(this.parseInline(delToken.tokens, renderer));
          break;
        }
        case "text": {
          const textToken = token;
          out += renderer.text(textToken.text);
          break;
        }
        default: {
          const errMsg = 'Token with "' + token.type + '" type was not found.';
          if (this.options.silent) {
            console.error(errMsg);
            return "";
          } else {
            throw new Error(errMsg);
          }
        }
      }
    }
    return out;
  }
}
class _Hooks {
  constructor(options) {
    __publicField(this, "options");
    this.options = options || _defaults;
  }
  /**
   * Process markdown before marked
   */
  preprocess(markdown) {
    return markdown;
  }
  /**
   * Process HTML after marked is finished
   */
  postprocess(html2) {
    return html2;
  }
  /**
   * Process all tokens before walk tokens
   */
  processAllTokens(tokens) {
    return tokens;
  }
}
__publicField(_Hooks, "passThroughHooks", /* @__PURE__ */ new Set([
  "preprocess",
  "postprocess",
  "processAllTokens"
]));
class Marked {
  constructor(...args) {
    __privateAdd(this, _Marked_instances);
    __publicField(this, "defaults", _getDefaults());
    __publicField(this, "options", this.setOptions);
    __publicField(this, "parse", __privateMethod(this, _Marked_instances, parseMarkdown_fn).call(this, _Lexer.lex, _Parser.parse));
    __publicField(this, "parseInline", __privateMethod(this, _Marked_instances, parseMarkdown_fn).call(this, _Lexer.lexInline, _Parser.parseInline));
    __publicField(this, "Parser", _Parser);
    __publicField(this, "Renderer", _Renderer);
    __publicField(this, "TextRenderer", _TextRenderer);
    __publicField(this, "Lexer", _Lexer);
    __publicField(this, "Tokenizer", _Tokenizer);
    __publicField(this, "Hooks", _Hooks);
    this.use(...args);
  }
  /**
   * Run callback for every token
   */
  walkTokens(tokens, callback) {
    var _a, _b;
    let values = [];
    for (const token of tokens) {
      values = values.concat(callback.call(this, token));
      switch (token.type) {
        case "table": {
          const tableToken = token;
          for (const cell of tableToken.header) {
            values = values.concat(this.walkTokens(cell.tokens, callback));
          }
          for (const row of tableToken.rows) {
            for (const cell of row) {
              values = values.concat(this.walkTokens(cell.tokens, callback));
            }
          }
          break;
        }
        case "list": {
          const listToken = token;
          values = values.concat(this.walkTokens(listToken.items, callback));
          break;
        }
        default: {
          const genericToken = token;
          if ((_b = (_a = this.defaults.extensions) == null ? void 0 : _a.childTokens) == null ? void 0 : _b[genericToken.type]) {
            this.defaults.extensions.childTokens[genericToken.type].forEach((childTokens) => {
              values = values.concat(this.walkTokens(genericToken[childTokens], callback));
            });
          } else if (genericToken.tokens) {
            values = values.concat(this.walkTokens(genericToken.tokens, callback));
          }
        }
      }
    }
    return values;
  }
  use(...args) {
    const extensions = this.defaults.extensions || { renderers: {}, childTokens: {} };
    args.forEach((pack) => {
      const opts = { ...pack };
      opts.async = this.defaults.async || opts.async || false;
      if (pack.extensions) {
        pack.extensions.forEach((ext) => {
          if (!ext.name) {
            throw new Error("extension name required");
          }
          if ("renderer" in ext) {
            const prevRenderer = extensions.renderers[ext.name];
            if (prevRenderer) {
              extensions.renderers[ext.name] = function(...args2) {
                let ret = ext.renderer.apply(this, args2);
                if (ret === false) {
                  ret = prevRenderer.apply(this, args2);
                }
                return ret;
              };
            } else {
              extensions.renderers[ext.name] = ext.renderer;
            }
          }
          if ("tokenizer" in ext) {
            if (!ext.level || ext.level !== "block" && ext.level !== "inline") {
              throw new Error("extension level must be 'block' or 'inline'");
            }
            const extLevel = extensions[ext.level];
            if (extLevel) {
              extLevel.unshift(ext.tokenizer);
            } else {
              extensions[ext.level] = [ext.tokenizer];
            }
            if (ext.start) {
              if (ext.level === "block") {
                if (extensions.startBlock) {
                  extensions.startBlock.push(ext.start);
                } else {
                  extensions.startBlock = [ext.start];
                }
              } else if (ext.level === "inline") {
                if (extensions.startInline) {
                  extensions.startInline.push(ext.start);
                } else {
                  extensions.startInline = [ext.start];
                }
              }
            }
          }
          if ("childTokens" in ext && ext.childTokens) {
            extensions.childTokens[ext.name] = ext.childTokens;
          }
        });
        opts.extensions = extensions;
      }
      if (pack.renderer) {
        const renderer = this.defaults.renderer || new _Renderer(this.defaults);
        for (const prop in pack.renderer) {
          if (!(prop in renderer)) {
            throw new Error(`renderer '${prop}' does not exist`);
          }
          if (prop === "options") {
            continue;
          }
          const rendererProp = prop;
          const rendererFunc = pack.renderer[rendererProp];
          const prevRenderer = renderer[rendererProp];
          renderer[rendererProp] = (...args2) => {
            let ret = rendererFunc.apply(renderer, args2);
            if (ret === false) {
              ret = prevRenderer.apply(renderer, args2);
            }
            return ret || "";
          };
        }
        opts.renderer = renderer;
      }
      if (pack.tokenizer) {
        const tokenizer = this.defaults.tokenizer || new _Tokenizer(this.defaults);
        for (const prop in pack.tokenizer) {
          if (!(prop in tokenizer)) {
            throw new Error(`tokenizer '${prop}' does not exist`);
          }
          if (["options", "rules", "lexer"].includes(prop)) {
            continue;
          }
          const tokenizerProp = prop;
          const tokenizerFunc = pack.tokenizer[tokenizerProp];
          const prevTokenizer = tokenizer[tokenizerProp];
          tokenizer[tokenizerProp] = (...args2) => {
            let ret = tokenizerFunc.apply(tokenizer, args2);
            if (ret === false) {
              ret = prevTokenizer.apply(tokenizer, args2);
            }
            return ret;
          };
        }
        opts.tokenizer = tokenizer;
      }
      if (pack.hooks) {
        const hooks = this.defaults.hooks || new _Hooks();
        for (const prop in pack.hooks) {
          if (!(prop in hooks)) {
            throw new Error(`hook '${prop}' does not exist`);
          }
          if (prop === "options") {
            continue;
          }
          const hooksProp = prop;
          const hooksFunc = pack.hooks[hooksProp];
          const prevHook = hooks[hooksProp];
          if (_Hooks.passThroughHooks.has(prop)) {
            hooks[hooksProp] = (arg) => {
              if (this.defaults.async) {
                return Promise.resolve(hooksFunc.call(hooks, arg)).then((ret2) => {
                  return prevHook.call(hooks, ret2);
                });
              }
              const ret = hooksFunc.call(hooks, arg);
              return prevHook.call(hooks, ret);
            };
          } else {
            hooks[hooksProp] = (...args2) => {
              let ret = hooksFunc.apply(hooks, args2);
              if (ret === false) {
                ret = prevHook.apply(hooks, args2);
              }
              return ret;
            };
          }
        }
        opts.hooks = hooks;
      }
      if (pack.walkTokens) {
        const walkTokens = this.defaults.walkTokens;
        const packWalktokens = pack.walkTokens;
        opts.walkTokens = function(token) {
          let values = [];
          values.push(packWalktokens.call(this, token));
          if (walkTokens) {
            values = values.concat(walkTokens.call(this, token));
          }
          return values;
        };
      }
      this.defaults = { ...this.defaults, ...opts };
    });
    return this;
  }
  setOptions(opt) {
    this.defaults = { ...this.defaults, ...opt };
    return this;
  }
  lexer(src, options) {
    return _Lexer.lex(src, options ?? this.defaults);
  }
  parser(tokens, options) {
    return _Parser.parse(tokens, options ?? this.defaults);
  }
}
_Marked_instances = new WeakSet();
parseMarkdown_fn = function(lexer, parser) {
  return (src, options) => {
    const origOpt = { ...options };
    const opt = { ...this.defaults, ...origOpt };
    if (this.defaults.async === true && origOpt.async === false) {
      if (!opt.silent) {
        console.warn("marked(): The async option was set to true by an extension. The async: false option sent to parse will be ignored.");
      }
      opt.async = true;
    }
    const throwError = __privateMethod(this, _Marked_instances, onError_fn).call(this, !!opt.silent, !!opt.async);
    if (typeof src === "undefined" || src === null) {
      return throwError(new Error("marked(): input parameter is undefined or null"));
    }
    if (typeof src !== "string") {
      return throwError(new Error("marked(): input parameter is of type " + Object.prototype.toString.call(src) + ", string expected"));
    }
    if (opt.hooks) {
      opt.hooks.options = opt;
    }
    if (opt.async) {
      return Promise.resolve(opt.hooks ? opt.hooks.preprocess(src) : src).then((src2) => lexer(src2, opt)).then((tokens) => opt.hooks ? opt.hooks.processAllTokens(tokens) : tokens).then((tokens) => opt.walkTokens ? Promise.all(this.walkTokens(tokens, opt.walkTokens)).then(() => tokens) : tokens).then((tokens) => parser(tokens, opt)).then((html2) => opt.hooks ? opt.hooks.postprocess(html2) : html2).catch(throwError);
    }
    try {
      if (opt.hooks) {
        src = opt.hooks.preprocess(src);
      }
      let tokens = lexer(src, opt);
      if (opt.hooks) {
        tokens = opt.hooks.processAllTokens(tokens);
      }
      if (opt.walkTokens) {
        this.walkTokens(tokens, opt.walkTokens);
      }
      let html2 = parser(tokens, opt);
      if (opt.hooks) {
        html2 = opt.hooks.postprocess(html2);
      }
      return html2;
    } catch (e) {
      return throwError(e);
    }
  };
};
onError_fn = function(silent, async) {
  return (e) => {
    e.message += "\nPlease report this to https://github.com/markedjs/marked.";
    if (silent) {
      const msg = "<p>An error occurred:</p><pre>" + escape$1(e.message + "", true) + "</pre>";
      if (async) {
        return Promise.resolve(msg);
      }
      return msg;
    }
    if (async) {
      return Promise.reject(e);
    }
    throw e;
  };
};
const markedInstance = new Marked();
function marked(src, opt) {
  return markedInstance.parse(src, opt);
}
marked.options = marked.setOptions = function(options) {
  markedInstance.setOptions(options);
  marked.defaults = markedInstance.defaults;
  changeDefaults(marked.defaults);
  return marked;
};
marked.getDefaults = _getDefaults;
marked.defaults = _defaults;
marked.use = function(...args) {
  markedInstance.use(...args);
  marked.defaults = markedInstance.defaults;
  changeDefaults(marked.defaults);
  return marked;
};
marked.walkTokens = function(tokens, callback) {
  return markedInstance.walkTokens(tokens, callback);
};
marked.parseInline = markedInstance.parseInline;
marked.Parser = _Parser;
marked.parser = _Parser.parse;
marked.Renderer = _Renderer;
marked.TextRenderer = _TextRenderer;
marked.Lexer = _Lexer;
marked.lexer = _Lexer.lex;
marked.Tokenizer = _Tokenizer;
marked.Hooks = _Hooks;
marked.parse = marked;
marked.options;
marked.setOptions;
marked.use;
marked.walkTokens;
marked.parseInline;
_Parser.parse;
_Lexer.lex;
function extractCodeBlocks(markdown) {
  const blocks = [];
  const regex = /```(\w*)\n([\s\S]*?)```/g;
  let match;
  while ((match = regex.exec(markdown)) !== null) {
    blocks.push({ language: match[1].trim().toLowerCase(), content: match[2] });
  }
  return blocks;
}
function buildHtmlDoc(blocks) {
  const css = blocks.filter((b) => b.language === "css").map((b) => b.content).join("\n");
  const js = blocks.filter((b) => b.language === "javascript" || b.language === "js").map((b) => b.content).join("\n");
  const html2 = blocks.filter((b) => b.language === "html").map((b) => b.content).join("\n");
  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
${css}
</style>
<script>
${js}
<\/script>
</head>
<body>
${html2}
</body>
</html>`;
}
function configureMarked() {
  marked.setOptions({
    breaks: false,
    gfm: true
  });
}
function renderMarkdown(text) {
  configureMarked();
  const html2 = marked(text);
  return html2.replace(/on\w+="[^"]*"/g, "").replace(/javascript:/g, "").replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "");
}
function sanitizeInput(input) {
  return input.trim().replace(/\x00/g, "").replace(/[\x80-\xFF]/g, (char) => {
    const code = char.charCodeAt(0);
    return code >= 32 ? char : "";
  }).substring(0, 4e3);
}
const OBJECT_MERGE_KEYS = /* @__PURE__ */ new Set(["colors", "logoBorder", "logoGlow", "logoBackground"]);
const IGNORED_DEFAULT_KEYS = /* @__PURE__ */ new Set(["apiKey", "apiSecret", "onOpen", "onInit", "onError"]);
function isPlainObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function mergeServerDefaults(userConfig, serverDefaults) {
  const merged = {};
  for (const [key, value] of Object.entries(serverDefaults)) {
    if (value === void 0 || IGNORED_DEFAULT_KEYS.has(key)) {
      continue;
    }
    if (key === "language" && value === "auto") {
      continue;
    }
    const userValue = userConfig[key];
    if (userValue === void 0) {
      merged[key] = value;
      continue;
    }
    if (OBJECT_MERGE_KEYS.has(key) && isPlainObject(value) && isPlainObject(userValue)) {
      merged[key] = { ...value, ...userValue };
    }
  }
  return merged;
}
function sanitizeServerDefaults(serverDefaults) {
  const sanitized = {};
  for (const [key, value] of Object.entries(serverDefaults)) {
    if (value === void 0 || IGNORED_DEFAULT_KEYS.has(key)) {
      continue;
    }
    if (key === "language" && value === "auto") {
      continue;
    }
    sanitized[key] = value;
  }
  return sanitized;
}
function createElement(tag2, attrs = {}, html2 = "") {
  const element = document.createElement(tag2);
  Object.entries(attrs).forEach(([key, value]) => {
    if (key === "class") {
      element.className = value;
    } else if (key === "id") {
      element.id = value;
    } else if (key.startsWith("aria-")) {
      element.setAttribute(key, value);
    } else if (key.startsWith("data-")) {
      element.setAttribute(key, value);
    } else {
      element.setAttribute(key, value);
    }
  });
  if (html2) {
    element.innerHTML = html2;
  }
  return element;
}
function addEvent(element, event, handler, options) {
  element.addEventListener(event, handler, options);
  return () => element.removeEventListener(event, handler);
}
function scrollToBottom(element) {
  element.scrollTop = element.scrollHeight;
}
const ABBREVIATIONS = /* @__PURE__ */ new Set([
  // German
  "z.b",
  "u.a",
  "etc",
  "dr",
  "prof",
  "nr",
  "bzw",
  "d.h",
  "ca",
  "usw",
  "inkl",
  "exkl",
  "vgl",
  "siehe",
  "v.a",
  "ab",
  "abk",
  "allg",
  "anm",
  "anon",
  "anst",
  "a.ö",
  "Bd",
  "begr",
  "bes",
  "beisp",
  "betr",
  "bez",
  "bh",
  "bibl",
  "biol",
  "bk",
  "bl",
  "chem",
  "co",
  "d.h",
  "d.i",
  "d.j",
  "d.m",
  "d.r",
  "d.s",
  "d.v",
  "dat",
  "dgl",
  "dipl",
  "diss",
  "dt",
  "e.h",
  "e.v",
  "ebd",
  "ed",
  "eig",
  "eigtl",
  "eing",
  "engl",
  "entspr",
  "erw",
  "etw",
  "evtl",
  "exkl",
  "f",
  "fa",
  "fam",
  "ff",
  "fig",
  "fr",
  "g.g.a",
  "g.u",
  "geb",
  "geh",
  "gen",
  "ges",
  "ggf",
  "gr",
  "hbf",
  "hebr",
  "hg",
  "hin",
  "hist",
  "hiv",
  "hl",
  "hr",
  "hrsg",
  "i.a",
  "i.d",
  "i.e",
  "i.v",
  "i.w.s",
  "ibid",
  "ig",
  "ii",
  "incl",
  "inkl",
  "int",
  "i.a",
  "jhd",
  "jhdt",
  "jr",
  "jur",
  "k.u.k",
  "kath",
  "kfm",
  "kg",
  "kl",
  "k.o",
  "k.u.k",
  "lab",
  "lat",
  "lfd",
  "lit",
  "lt",
  "m.e",
  "m.m.n",
  "m.w.n",
  "med",
  "mev",
  "min",
  "mm",
  "mio",
  "mrd",
  "mtl",
  "n.chr",
  "n.d",
  "n.f",
  "n.j",
  "n.m",
  "n.r",
  "n.w",
  "n.z",
  "nachf",
  "nat",
  "nd",
  "nr",
  "o.a",
  "o.ä",
  "o.g",
  "o.k",
  "od",
  "ohg",
  "op",
  "ord",
  "orig",
  "p.a",
  "p.m",
  "p.p",
  "p.s",
  "pa",
  "pag",
  "pers",
  "pf",
  "phil",
  "pl",
  "pol",
  "pp",
  "pr",
  "priv",
  "prof",
  "prot",
  "q.v",
  "r.k",
  "rd",
  "reg",
  "resp",
  "rev",
  "röm",
  "s.a",
  "s.b",
  "s.o",
  "s.u",
  "s.v",
  "sa",
  "sel",
  "sen",
  "seit",
  "so",
  "soz",
  "sp",
  "spp",
  "sr",
  "ss",
  "st",
  "std",
  "str",
  "täg",
  "tab",
  "techn",
  "tel",
  "theol",
  "tr",
  "u.a",
  "u.ä",
  "u.s.w",
  "u.v.a",
  "u.v.m",
  "u.z",
  "ult",
  "univ",
  "urspr",
  "usw",
  "u.z",
  "v.a",
  "v.chr",
  "v.g",
  "v.h",
  "v.m",
  "v.o",
  "v.r",
  "v.t",
  "v.u.z",
  "verh",
  "verm",
  "vers",
  "vgl",
  "viz",
  "vol",
  "vollst",
  "vorbem",
  "vors",
  "w.e",
  "w.o",
  "w.u",
  "wiss",
  "wohl",
  "z.b",
  "z.t",
  "z.u",
  "zus",
  "zz",
  "zzt",
  // English
  "i.e",
  "e.g",
  "vs",
  "dr",
  "mr",
  "mrs",
  "ms",
  "prof",
  "sr",
  "jr",
  "st",
  "ave",
  "blvd",
  "rd",
  "ln",
  "ct",
  " approx",
  "inc",
  "ltd",
  "co",
  "corp",
  "llc",
  "plc",
  "et al",
  "vol",
  "vols",
  "rev",
  "est",
  "fig",
  "figs",
  "gov",
  "dept",
  "dept",
  "nos",
  "art",
  "arts",
  "pp",
  "par",
  "pars",
  "eq",
  "eqs",
  "ch",
  "chs",
  "ed",
  "eds",
  "trans",
  "vol",
  "vols",
  "bk",
  "bks",
  "pt",
  "pts",
  "sec",
  "secs",
  "nos",
  "circa",
  "ca",
  "a.m",
  "p.m",
  "b.c",
  "a.d",
  "ph.d",
  "m.d",
  "b.a",
  "m.a",
  "d.d.s",
  "d.v.m",
  "esq",
  "hon",
  "m.b.a",
  "m.d",
  "ph.d",
  "r.n",
  "rep",
  "reps",
  "sen",
  "sens",
  "col",
  "cols",
  "gen",
  "gens",
  "lt",
  "lts",
  "maj",
  "sgt",
  "cpl",
  "capt",
  "cmdr",
  "adm",
  "rev",
  "revs",
  "hon",
  "hons",
  "assoc",
  "asst",
  "assts",
  "ext",
  "jan",
  "feb",
  "mar",
  "apr",
  "jun",
  "jul",
  "aug",
  "sep",
  "sept",
  "oct",
  "nov",
  "dec",
  "mon",
  "tue",
  "wed",
  "thu",
  "fri",
  "sat",
  "sun"
]);
function normWord(w) {
  return w.toLowerCase().replace(/[.,;:!?]+$/, "").replace(/^\W+/, "");
}
function estimatePause(whitespace) {
  if (whitespace.includes("\n\n")) return 600;
  if (whitespace.includes("\n")) return 300;
  return 150;
}
function extractCompleteSentences(buffer) {
  if (!buffer.trim()) return { chunks: [], remainder: buffer };
  const chunks = [];
  let lastSplit = 0;
  for (let i = 0; i < buffer.length; i++) {
    const ch = buffer[i];
    if (ch !== "." && ch !== "!" && ch !== "?") continue;
    const next = buffer[i + 1];
    if (next !== void 0 && next !== " " && next !== "\n" && next !== "	" && next !== "\r") {
      continue;
    }
    const candidate = buffer.slice(lastSplit, i + 1).trim();
    if (candidate.length < 3) {
      lastSplit = i + 1;
      continue;
    }
    const before = buffer.slice(lastSplit, i).trimEnd();
    const words = before.split(/\s+/);
    const lastWord = words[words.length - 1] || "";
    const norm = normWord(lastWord);
    if (ABBREVIATIONS.has(norm)) {
      continue;
    }
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
class TTSSegmentPlayer {
  constructor() {
    __publicField(this, "queue", []);
    __publicField(this, "isPlaying", false);
    __publicField(this, "currentAudio", null);
    __publicField(this, "_playedAny", false);
  }
  /** Number of segments currently queued (including the one playing). */
  get queueLength() {
    return this.queue.length;
  }
  /** True if at least one segment started playing successfully. */
  hasPlayedAny() {
    return this._playedAny;
  }
  /**
   * Add a sentence to the queue.  `fetchSegment` must return a
   * Promise<Blob> (e.g. a fetch to /audio/tts/segment).
   */
  add(text, voiceId, fetchSegment, pauseAfterMs) {
    const blobPromise = fetchSegment(text, voiceId);
    this.queue.push({ text, voiceId, blobPromise, pauseAfterMs });
    if (!this.isPlaying) {
      this.playNext();
    }
  }
  /** Stop playback, clear the queue, and revoke any object URLs. */
  abort() {
    if (this.currentAudio) {
      this.currentAudio.pause();
      const src = this.currentAudio.src;
      if (src && src.startsWith("blob:")) {
        URL.revokeObjectURL(src);
      }
      this.currentAudio = null;
    }
    this.queue = [];
    this.isPlaying = false;
  }
  async playNext() {
    if (this.isPlaying) return;
    this.isPlaying = true;
    while (this.queue.length > 0) {
      const item = this.queue.shift();
      try {
        const blob = await item.blobPromise;
        if (this.queue.length === 0 && !this.isPlaying) {
          URL.revokeObjectURL(URL.createObjectURL(blob));
          break;
        }
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        this.currentAudio = audio;
        await new Promise((resolve) => {
          audio.onended = () => {
            URL.revokeObjectURL(url);
            this.currentAudio = null;
            this._playedAny = true;
            resolve();
          };
          audio.onerror = () => {
            URL.revokeObjectURL(url);
            this.currentAudio = null;
            console.warn("[TTSSegmentPlayer] Audio playback error, skipping segment");
            resolve();
          };
          audio.play().catch(() => {
            URL.revokeObjectURL(url);
            this.currentAudio = null;
            console.warn("[TTSSegmentPlayer] Audio play() rejected, skipping segment");
            resolve();
          });
        });
        if (item.pauseAfterMs && item.pauseAfterMs > 0 && this.queue.length > 0) {
          await new Promise((resolve) => setTimeout(resolve, item.pauseAfterMs));
        }
      } catch (err) {
        console.warn("[TTSSegmentPlayer] Failed to fetch segment, skipping:", err);
      }
    }
    this.isPlaying = false;
  }
}
function audioBufferToWav(buffer) {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1;
  const bitDepth = 16;
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const dataLength = buffer.length * numChannels * bytesPerSample;
  const bufferLength = 44 + dataLength;
  const arrayBuffer = new ArrayBuffer(bufferLength);
  const view = new DataView(arrayBuffer);
  function writeString(offset2, str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset2 + i, str.charCodeAt(i));
    }
  }
  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(36, "data");
  view.setUint32(40, dataLength, true);
  const channels = [];
  for (let i = 0; i < numChannels; i++) {
    channels.push(buffer.getChannelData(i));
  }
  let offset = 44;
  for (let i = 0; i < buffer.length; i++) {
    for (let c = 0; c < numChannels; c++) {
      const sample = Math.max(-1, Math.min(1, channels[c][i]));
      view.setInt16(offset, sample < 0 ? sample * 32768 : sample * 32767, true);
      offset += 2;
    }
  }
  return new Blob([arrayBuffer], { type: "audio/wav" });
}
async function blobToWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new AudioContext();
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    return audioBufferToWav(audioBuffer);
  } finally {
    await audioContext.close();
  }
}
const WIDGET_STYLES = `
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
const WIDGET_FAVICON_BASE64 = "data:image/png;base64,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";
const WIDGET_LOGO_BASE64 = "data:image/png;base64,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";
const ort$1 = () => window.ort;
(() => {
  const o = window.ort;
  if (o) {
    o.env.wasm.numThreads = 1;
    o.env.wasm.simd = true;
  }
})();
class FaceDetector {
  constructor() {
    __publicField(this, "session", null);
    __publicField(this, "inputSize", 320);
    __publicField(this, "modelUrl", null);
  }
  async load(modelUrl) {
    if (this.session && this.modelUrl === modelUrl) return true;
    try {
      const eps = ["webgpu", "wasm"].filter((p) => {
        if (p === "webgpu" && "gpu" in navigator) return true;
        return p === "wasm";
      });
      this.session = await ort$1().InferenceSession.create(modelUrl, {
        executionProviders: eps.length > 0 ? eps : ["wasm"]
      });
      this.modelUrl = modelUrl;
      return true;
    } catch (err) {
      console.warn("[Emotion] Face detector load failed:", err);
      this.session = null;
      this.modelUrl = null;
      return false;
    }
  }
  isReady() {
    return this.session !== null;
  }
  async detect(imageData) {
    if (!this.session) return [];
    const { width, height } = imageData;
    const canvas = document.createElement("canvas");
    canvas.width = this.inputSize;
    canvas.height = this.inputSize;
    const ctx = canvas.getContext("2d");
    const scale = Math.min(this.inputSize / width, this.inputSize / height);
    const newW = Math.round(width * scale);
    const newH = Math.round(height * scale);
    const dx = (this.inputSize - newW) / 2;
    const dy = (this.inputSize - newH) / 2;
    const imgCanvas = document.createElement("canvas");
    imgCanvas.width = width;
    imgCanvas.height = height;
    const imgCtx = imgCanvas.getContext("2d");
    imgCtx.putImageData(imageData, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, this.inputSize, this.inputSize);
    ctx.drawImage(imgCanvas, 0, 0, width, height, dx, dy, newW, newH);
    const resized = ctx.getImageData(0, 0, this.inputSize, this.inputSize);
    const floatData = new Float32Array(this.inputSize * this.inputSize * 3);
    for (let y = 0; y < this.inputSize; y++) {
      for (let x = 0; x < this.inputSize; x++) {
        const srcIdx = (y * this.inputSize + x) * 4;
        const dstIdx = y * this.inputSize + x;
        floatData[dstIdx] = resized.data[srcIdx] / 255;
        floatData[this.inputSize * this.inputSize + dstIdx] = resized.data[srcIdx + 1] / 255;
        floatData[2 * this.inputSize * this.inputSize + dstIdx] = resized.data[srcIdx + 2] / 255;
      }
    }
    try {
      const tensor = new (ort$1()).Tensor("float32", floatData, [1, 3, this.inputSize, this.inputSize]);
      const feeds = { input: tensor };
      const results = await this.session.run(feeds);
      return this._postprocess(results, scale, dx, dy, width, height);
    } catch (err) {
      console.warn("[Emotion] Face detection inference failed:", err);
      return [];
    }
  }
  _postprocess(results, scale, dx, dy, origW, origH) {
    const output = results.output || results["output0"] || results["detections"] || Object.values(results)[0];
    if (!output || !(output instanceof ort$1().Tensor)) return [];
    const data = output.data;
    const dims = output.dims;
    const boxes = [];
    if (dims.length === 3 && dims[2] >= 5) {
      const [, numBoxes, boxDim] = dims;
      const confThreshold = 0.5;
      const nmsThreshold = 0.4;
      const candidates = [];
      for (let i = 0; i < numBoxes; i++) {
        const offset = i * boxDim;
        const conf = data[offset + 4];
        if (conf > confThreshold) {
          const x1 = data[offset];
          const y1 = data[offset + 1];
          const x2 = data[offset + 2];
          const y2 = data[offset + 3];
          candidates.push([x1, y1, x2, y2, conf]);
        }
      }
      candidates.sort((a, b) => b[4] - a[4]);
      const picked = [];
      for (const box of candidates) {
        let shouldPick = true;
        for (const p of picked) {
          if (this._iou(box, p) > nmsThreshold) {
            shouldPick = false;
            break;
          }
        }
        if (shouldPick) picked.push(box);
      }
      for (const [x1, y1, x2, y2, conf] of picked) {
        const cx = (x1 - dx) / scale;
        const cy = (y1 - dy) / scale;
        const cw = (x2 - x1) / scale;
        const ch = (y2 - y1) / scale;
        boxes.push({
          x: Math.max(0, Math.round(cx)),
          y: Math.max(0, Math.round(cy)),
          w: Math.min(origW, Math.round(cw)),
          h: Math.min(origH, Math.round(ch)),
          conf
        });
      }
    }
    return boxes.slice(0, 5);
  }
  _iou(a, b) {
    const x1 = Math.max(a[0], b[0]);
    const y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[2], b[2]);
    const y2 = Math.min(a[3], b[3]);
    const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a[2] - a[0]) * (a[3] - a[1]);
    const areaB = (b[2] - b[0]) * (b[3] - b[1]);
    return interArea / (areaA + areaB - interArea + 1e-6);
  }
}
const ort = () => window.ort;
(() => {
  const o = window.ort;
  if (o) {
    o.env.wasm.numThreads = 1;
    o.env.wasm.simd = true;
  }
})();
const FER_LABELS = [
  "neutral",
  "happy",
  "sad",
  "surprised",
  "fear",
  "disgust",
  "angry"
];
const LABEL_MAP = {
  neutral: "neutral",
  happy: "satisfied",
  sad: "sad",
  surprised: "surprised",
  fear: "anxious",
  disgust: "frustrated",
  angry: "angry"
};
class ExpressionClassifier {
  constructor() {
    __publicField(this, "session", null);
    __publicField(this, "inputSize", 48);
    // FER+ standard size
    __publicField(this, "modelUrl", null);
  }
  async load(modelUrl) {
    if (this.session && this.modelUrl === modelUrl) return true;
    try {
      const eps = ["webgpu", "wasm"].filter((p) => {
        if (p === "webgpu" && "gpu" in navigator) return true;
        return p === "wasm";
      });
      this.session = await ort().InferenceSession.create(modelUrl, {
        executionProviders: eps.length > 0 ? eps : ["wasm"]
      });
      this.modelUrl = modelUrl;
      return true;
    } catch (err) {
      console.warn("[Emotion] Expression classifier load failed:", err);
      this.session = null;
      this.modelUrl = null;
      return false;
    }
  }
  isReady() {
    return this.session !== null;
  }
  async classify(croppedFace) {
    if (!this.session) return { emotion: "unknown", confidence: 0 };
    const canvas = document.createElement("canvas");
    canvas.width = this.inputSize;
    canvas.height = this.inputSize;
    const ctx = canvas.getContext("2d");
    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = croppedFace.width;
    srcCanvas.height = croppedFace.height;
    const srcCtx = srcCanvas.getContext("2d");
    srcCtx.putImageData(croppedFace, 0, 0);
    ctx.drawImage(srcCanvas, 0, 0, croppedFace.width, croppedFace.height, 0, 0, this.inputSize, this.inputSize);
    const resized = ctx.getImageData(0, 0, this.inputSize, this.inputSize);
    const grayData = new Float32Array(this.inputSize * this.inputSize);
    for (let i = 0; i < this.inputSize * this.inputSize; i++) {
      const r = resized.data[i * 4];
      const g = resized.data[i * 4 + 1];
      const b = resized.data[i * 4 + 2];
      grayData[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    }
    try {
      const tensor = new (ort()).Tensor("float32", grayData, [1, 1, this.inputSize, this.inputSize]);
      const feeds = { input: tensor };
      const results = await this.session.run(feeds);
      const output = results.output || results["output0"] || results["probabilities"] || Object.values(results)[0];
      if (!output || !(output instanceof ort().Tensor)) {
        return { emotion: "unknown", confidence: 0 };
      }
      const probs = output.data;
      let maxIdx = 0;
      let maxVal = probs[0];
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > maxVal) {
          maxVal = probs[i];
          maxIdx = i;
        }
      }
      const rawLabel = FER_LABELS[maxIdx] || "unknown";
      const mappedEmotion = LABEL_MAP[rawLabel] || rawLabel;
      return { emotion: mappedEmotion, confidence: Math.round(maxVal * 100) / 100 };
    } catch (err) {
      console.warn("[Emotion] Expression classification inference failed:", err);
      return { emotion: "unknown", confidence: 0 };
    }
  }
}
function aggregateResults(results, sourceLabel) {
  var _a;
  if (results.length === 0) return null;
  if (results.length === 1) return results[0];
  const counts = {};
  const confidences = {};
  for (const r of results) {
    const emo = r.primary_emotion;
    counts[emo] = (counts[emo] || 0) + 1;
    if (!confidences[emo]) confidences[emo] = [];
    confidences[emo].push(r.confidence);
  }
  const primary = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  const avgConfidence = confidences[primary].reduce((a, b) => a + b, 0) / confidences[primary].length;
  const intensityCounts = {};
  for (const r of results) {
    if (r.primary_emotion === primary) {
      const int = r.intensity;
      intensityCounts[int] = (intensityCounts[int] || 0) + 1;
    }
  }
  const intensity = ((_a = Object.entries(intensityCounts).sort((a, b) => b[1] - a[1])[0]) == null ? void 0 : _a[0]) || "low";
  const validFaceFrames = results.reduce((sum, r) => sum + (r.valid_face_frames || 0), 0);
  const sampledFrames = results.reduce((sum, r) => sum + (r.sampled_frames || 1), 0);
  return {
    primary_emotion: primary,
    confidence: Math.round(avgConfidence * 100) / 100,
    intensity,
    source: sourceLabel || results[0].source || "aggregate",
    timestamp_ms: Date.now(),
    sampled_frames: sampledFrames,
    valid_face_frames: validFaceFrames,
    face_detection_rate: sampledFrames > 0 ? Math.round(validFaceFrames / sampledFrames * 100) / 100 : 0
  };
}
function aggregateFrameResults(results) {
  const aggregated = aggregateResults(results, "webcam_browser_onnx");
  if (!aggregated) return null;
  return aggregated;
}
function fuseEmotions(results, weights) {
  if (results.length === 0) return null;
  if (results.length === 1) return results[0];
  const defaultWeights = [
    { source: "webcam_browser_onnx", weight: 1 },
    { source: "audio_prosody_browser", weight: 0.8 },
    { source: "text_rules", weight: 0.6 }
  ];
  const weightMap = /* @__PURE__ */ new Map();
  for (const w of defaultWeights) {
    weightMap.set(w.source, w.weight);
  }
  const scores = {};
  const intensities = {};
  for (const r of results) {
    const w = weightMap.get(r.source) ?? 0.5;
    const emo = r.primary_emotion;
    scores[emo] = (scores[emo] || 0) + r.confidence * w;
    if (!intensities[emo]) intensities[emo] = { low: 0, medium: 0, high: 0 };
    intensities[emo][r.intensity] += w;
  }
  const primary = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
  const totalWeightForPrimary = results.filter((r) => r.primary_emotion === primary).reduce((sum, r) => sum + (weightMap.get(r.source) ?? 0.5), 0);
  const rawConfidence = scores[primary];
  const confidence = totalWeightForPrimary > 0 ? Math.round(Math.min(rawConfidence / totalWeightForPrimary, 1) * 100) / 100 : 0;
  const primaryIntensities = intensities[primary] || { low: 1, medium: 0, high: 0 };
  const intensity = Object.entries(primaryIntensities).sort((a, b) => b[1] - a[1])[0][0] || "low";
  const allSignals = results.flatMap((r) => r.signals || []);
  return {
    primary_emotion: primary,
    confidence,
    intensity,
    source: "multimodal_fusion",
    timestamp_ms: Date.now(),
    signals: Array.from(new Set(allSignals)).slice(0, 8)
  };
}
class SnippetBufferImpl {
  constructor(config) {
    __publicField(this, "video", null);
    __publicField(this, "stream", null);
    __publicField(this, "canvas");
    __publicField(this, "ctx");
    __publicField(this, "intervalId", null);
    __publicField(this, "rollingBuffer", []);
    __publicField(this, "config");
    __publicField(this, "active", false);
    __publicField(this, "faceDetector");
    __publicField(this, "expressionClassifier");
    __publicField(this, "modelsLoaded", false);
    this.config = config;
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.faceDetector = new FaceDetector();
    this.expressionClassifier = new ExpressionClassifier();
  }
  async start() {
    if (this.active) return;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, frameRate: { ideal: 10 } },
        audio: false
      });
      this.video = document.createElement("video");
      this.video.srcObject = this.stream;
      this.video.play();
      this.video.muted = true;
      this.video.playsInline = true;
      await new Promise((resolve) => {
        if (!this.video) return;
        this.video.onloadedmetadata = () => resolve();
        setTimeout(resolve, 1e3);
      });
      const [faceOk, exprOk] = await Promise.all([
        this.faceDetector.load(this.config.faceModelUrl),
        this.expressionClassifier.load(this.config.expressionModelUrl)
      ]);
      this.modelsLoaded = faceOk && exprOk;
      if (!this.modelsLoaded) {
        console.warn("[Emotion] ONNX models failed to load, falling back to placeholder");
      }
      this.active = true;
      const sampleIntervalMs = 1e3 / this.config.sampleFps;
      this.intervalId = setInterval(() => this._sampleFrame(), sampleIntervalMs);
    } catch (err) {
      console.warn("[Emotion] Webcam access failed:", err);
      this.active = false;
    }
  }
  stop() {
    this.active = false;
    this.modelsLoaded = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.video) {
      this.video.pause();
      this.video.srcObject = null;
      this.video = null;
    }
    this.rollingBuffer = [];
  }
  isActive() {
    return this.active;
  }
  _cropFace(imageData, box) {
    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = imageData.width;
    srcCanvas.height = imageData.height;
    const srcCtx = srcCanvas.getContext("2d");
    srcCtx.putImageData(imageData, 0, 0);
    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = box.w;
    cropCanvas.height = box.h;
    const cropCtx = cropCanvas.getContext("2d");
    cropCtx.drawImage(srcCanvas, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h);
    return cropCtx.getImageData(0, 0, box.w, box.h);
  }
  async _sampleFrame() {
    if (!this.video || !this.active || this.video.readyState < 2) return;
    const w = this.video.videoWidth || 320;
    const h = this.video.videoHeight || 240;
    this.canvas.width = w;
    this.canvas.height = h;
    this.ctx.drawImage(this.video, 0, 0, w, h);
    const imageData = this.ctx.getImageData(0, 0, w, h);
    let faceResults = [];
    if (this.modelsLoaded) {
      try {
        const faces = await this.faceDetector.detect(imageData);
        for (const face of faces) {
          const cropped = this._cropFace(imageData, face);
          const expr = await this.expressionClassifier.classify(cropped);
          let intensity = "low";
          if (expr.confidence >= 0.7) intensity = "high";
          else if (expr.confidence >= 0.4) intensity = "medium";
          faceResults.push({
            primary_emotion: expr.emotion,
            confidence: expr.confidence,
            intensity,
            source: "webcam_browser_onnx",
            timestamp_ms: Date.now(),
            valid_face_frames: 1,
            sampled_frames: 1,
            face_detection_rate: 1
          });
        }
      } catch (err) {
        console.warn("[Emotion] Frame analysis failed:", err);
      }
    }
    this.rollingBuffer.push({ imageData, timestamp: Date.now(), faceResults });
    const cutoff = Date.now() - this.config.windowMs;
    while (this.rollingBuffer.length > 0 && this.rollingBuffer[0].timestamp < cutoff) {
      this.rollingBuffer.shift();
    }
    while (this.rollingBuffer.length > this.config.maxFrames) {
      this.rollingBuffer.shift();
    }
  }
  async captureSnapshot() {
    if (this.rollingBuffer.length === 0 && this.video && this.active) {
      const w = this.video.videoWidth || 320;
      const h = this.video.videoHeight || 240;
      this.canvas.width = w;
      this.canvas.height = h;
      this.ctx.drawImage(this.video, 0, 0, w, h);
      const imageData = this.ctx.getImageData(0, 0, w, h);
      let faceResults = [];
      if (this.modelsLoaded) {
        try {
          const faces = await this.faceDetector.detect(imageData);
          for (const face of faces) {
            const cropped = this._cropFace(imageData, face);
            const expr = await this.expressionClassifier.classify(cropped);
            let intensity = "low";
            if (expr.confidence >= 0.7) intensity = "high";
            else if (expr.confidence >= 0.4) intensity = "medium";
            faceResults.push({
              primary_emotion: expr.emotion,
              confidence: expr.confidence,
              intensity,
              source: "webcam_browser_onnx",
              timestamp_ms: Date.now(),
              valid_face_frames: 1,
              sampled_frames: 1,
              face_detection_rate: 1
            });
          }
        } catch (err) {
          console.warn("[Emotion] Immediate capture failed:", err);
        }
      }
      this.rollingBuffer.push({ imageData, timestamp: Date.now(), faceResults });
    }
    const allFaceResults = this.rollingBuffer.flatMap((b) => b.faceResults);
    if (allFaceResults.length > 0) {
      return aggregateFrameResults(allFaceResults);
    }
    const validFrames = this.rollingBuffer.length;
    if (validFrames < this.config.minValidFaceFrames) {
      return {
        primary_emotion: "unknown",
        confidence: 0,
        intensity: "low",
        source: "webcam_browser_onnx",
        timestamp_ms: Date.now(),
        sampled_frames: validFrames,
        valid_face_frames: 0,
        face_detection_rate: 0,
        window_ms: this.config.windowMs
      };
    }
    return {
      primary_emotion: "neutral",
      confidence: 0,
      intensity: "low",
      source: "webcam_browser_onnx",
      timestamp_ms: Date.now(),
      sampled_frames: validFrames,
      valid_face_frames: 0,
      face_detection_rate: 0,
      window_ms: this.config.windowMs
    };
  }
}
function createVideoSnippetBuffer(config) {
  return new SnippetBufferImpl(config);
}
function applyHighpassFilter(samples, sampleRate, cutoffHz = 80) {
  const rc = 1 / (2 * Math.PI * cutoffHz);
  const dt = 1 / sampleRate;
  const alpha = rc / (rc + dt);
  const out = new Float32Array(samples.length);
  let prevIn = 0;
  let prevOut = 0;
  for (let i = 0; i < samples.length; i++) {
    const x = samples[i];
    const y = alpha * (prevOut + x - prevIn);
    out[i] = y;
    prevIn = x;
    prevOut = y;
  }
  return out;
}
function computeSNR(signalRMS, noiseFloorRMS) {
  if (noiseFloorRMS <= 0 || signalRMS <= 0) {
    return signalRMS > 0 ? Infinity : 0;
  }
  return 20 * Math.log10(signalRMS / noiseFloorRMS);
}
class AudioSnippetBufferImpl {
  constructor(config) {
    __publicField(this, "audioCtx", null);
    __publicField(this, "analyser", null);
    __publicField(this, "source", null);
    __publicField(this, "intervalId", null);
    __publicField(this, "features", []);
    __publicField(this, "active", false);
    __publicField(this, "config");
    this.config = config;
  }
  start(stream) {
    if (this.active) return;
    try {
      this.audioCtx = new AudioContext();
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 2048;
      this.source = this.audioCtx.createMediaStreamSource(stream);
      this.source.connect(this.analyser);
      this.active = true;
      this.features = [];
      this.intervalId = setInterval(() => {
        if (!this.analyser || !this.active) return;
        const bufferLength = this.analyser.fftSize;
        const dataArray = new Float32Array(bufferLength);
        this.analyser.getFloatTimeDomainData(dataArray);
        const filtered = applyHighpassFilter(dataArray, this.audioCtx.sampleRate, 80);
        const sumSq = filtered.reduce((acc, val) => acc + val * val, 0);
        const rms = Math.sqrt(sumSq / bufferLength);
        let zcr = 0;
        for (let i = 1; i < bufferLength; i++) {
          if (filtered[i - 1] * filtered[i] < 0) {
            zcr++;
          }
        }
        zcr /= bufferLength;
        this.features.push({ rms, zcr, timestamp: Date.now() });
      }, 100);
    } catch (err) {
      console.warn("[Emotion] Audio emotion start failed:", err);
      this.active = false;
    }
  }
  stop() {
    this.active = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }
    if (this.analyser) {
      this.analyser.disconnect();
      this.analyser = null;
    }
    if (this.audioCtx && this.audioCtx.state !== "closed") {
      this.audioCtx.close();
      this.audioCtx = null;
    }
    this.features = [];
  }
  isActive() {
    return this.active;
  }
  async captureSnapshot() {
    if (this.features.length < 10) {
      return null;
    }
    const rmsValues = this.features.map((f) => f.rms);
    const sortedRms = [...rmsValues].sort((a, b) => a - b);
    const noiseFloorRMS = sortedRms[Math.floor(sortedRms.length * 0.1)] || 1e-4;
    const signalRMS = rmsValues.reduce((a, b) => a + b, 0) / rmsValues.length;
    const snrDb = computeSNR(signalRMS, noiseFloorRMS);
    const minSnrDb = this.config.minSnrDb ?? 10;
    if (snrDb < minSnrDb) {
      console.info("[Emotion] Audio SNR too low:", snrDb.toFixed(1), "dB <", minSnrDb, "dB");
      return null;
    }
    const avgRms = this.features.reduce((a, b) => a + b.rms, 0) / this.features.length;
    const avgZcr = this.features.reduce((a, b) => a + b.zcr, 0) / this.features.length;
    const maxRms = Math.max(...this.features.map((f) => f.rms));
    const rmsVariance = this.features.reduce((a, b) => a + Math.pow(b.rms - avgRms, 2), 0) / this.features.length;
    const isHighEnergy = maxRms > 0.12;
    const isVeryHighEnergy = maxRms > 0.25;
    const isHighZcr = avgZcr > 0.12;
    const isVariable = rmsVariance > 4e-3;
    const isLowEnergy = avgRms < 0.015;
    let primaryEmotion = "neutral";
    let confidence = 0.3;
    let intensity = "low";
    if (isVeryHighEnergy && isHighZcr) {
      primaryEmotion = "angry";
      confidence = Math.min(maxRms * 2.5, 0.9);
      intensity = "high";
    } else if (isHighEnergy && isHighZcr && isVariable) {
      primaryEmotion = "frustrated";
      confidence = Math.min(maxRms * 2, 0.75);
      intensity = "medium";
    } else if (isHighEnergy && !isHighZcr) {
      primaryEmotion = "urgent";
      confidence = Math.min(maxRms * 2, 0.7);
      intensity = "high";
    } else if (isLowEnergy && !isHighZcr) {
      primaryEmotion = "sad";
      confidence = Math.min(0.5 + (0.015 - avgRms) * 20, 0.7);
      intensity = "medium";
    } else if (isVariable && avgRms > 0.025) {
      primaryEmotion = "anxious";
      confidence = Math.min(rmsVariance * 40, 0.65);
      intensity = "medium";
    }
    if (confidence < 0.3) {
      return null;
    }
    const validFrames = this.features.length;
    const sampledFrames = validFrames;
    return {
      primary_emotion: primaryEmotion,
      confidence: Math.round(confidence * 100) / 100,
      intensity,
      source: "audio_prosody_browser",
      timestamp_ms: Date.now(),
      sampled_frames: sampledFrames,
      valid_face_frames: validFrames,
      face_detection_rate: 1,
      window_ms: this.config.windowMs
    };
  }
}
function createAudioSnippetBuffer(config) {
  return new AudioSnippetBufferImpl(config);
}
const WIDGET_I18N = {
  de: {
    inputPlaceholder: "Nachricht eingeben...",
    privacyTitle: "Datenschutzinformation",
    privacyAccept: "Ich akzeptiere",
    privacyDecline: "Ablehnen",
    privacyDisabledPlaceholder: "Datenschutz muss akzeptiert werden...",
    emptyStateText: "Hallo! Ich bin dein Agentix CyberChatbot.<br>Wie kann ich dir heute helfen?",
    bounceText: "Hier chatten!",
    initialMessageContent: "Hallo!",
    privacyText: "Willkommen beim Chatbot! Ihre Konversationen werden ausschließlich auf den Servern des Betreibers verarbeitet. Es werden keine personenbezogenen Daten an Dritte weitergegeben.",
    error429: "Zu viele Anfragen. Bitte warten Sie einen Moment und versuchen es erneut.",
    error401: "Nicht autorisiert. Bitte überprüfen Sie Ihre API-Konfiguration.",
    error403: "Zugriff verweigert. Bitte überprüfen Sie Ihre Berechtigungen.",
    error404: "Der angeforderte Dienst wurde nicht gefunden.",
    error500: "Der Server ist vorübergehend nicht erreichbar. Bitte versuchen Sie es später erneut.",
    errorDefault: "Ein Fehler ist aufgetreten. Bitte versuchen Sie es erneut.",
    codeView: "Code",
    webView: "Web",
    sttEmpty: "Ich konnte dich leider nicht verstehen."
  },
  en: {
    inputPlaceholder: "Type your message here...",
    privacyTitle: "Privacy Notice",
    privacyAccept: "Accept",
    privacyDecline: "Decline",
    privacyDisabledPlaceholder: "Please accept the privacy notice to start chatting...",
    emptyStateText: "Hello! I am your Agentix CyberChatbot.<br>How can I help you today?",
    bounceText: "Chat here!",
    initialMessageContent: "Hello! Language: English",
    privacyText: "Welcome to the chatbot! Your conversations are processed exclusively on the operator's servers. No personal data is shared with third parties.",
    error429: "Too many requests. Please wait a moment and try again.",
    error401: "Unauthorized. Please check your API configuration.",
    error403: "Access denied. Please check your permissions.",
    error404: "The requested service was not found.",
    error500: "The server is temporarily unavailable. Please try again later.",
    errorDefault: "An error occurred. Please try again.",
    codeView: "Code",
    webView: "Web",
    sttEmpty: "I could not understand you, sorry."
  }
};
function detectLanguage() {
  const browserLang = navigator.language;
  if (browserLang) {
    const normalized = browserLang.toLowerCase().split("-")[0];
    if (normalized === "de") return "de";
    if (normalized === "en") return "en";
  }
  const browserLangs = navigator.languages;
  if (browserLangs && browserLangs.length > 0) {
    const normalized = browserLangs.map((l) => l.toLowerCase().split("-")[0]);
    if (normalized.includes("de")) return "de";
    if (normalized.includes("en")) return "en";
  }
  const pageLang = document.documentElement.lang;
  if (pageLang) {
    const normalized = pageLang.toLowerCase().split("-")[0];
    if (normalized === "de") return "de";
    if (normalized === "en") return "en";
  }
  return "de";
}
class ChatWidget {
  constructor(config = {}) {
    __publicField(this, "config");
    __publicField(this, "userConfig");
    __publicField(this, "container", null);
    __publicField(this, "window", null);
    __publicField(this, "messagesContainer", null);
    __publicField(this, "inputField", null);
    __publicField(this, "sendButton", null);
    __publicField(this, "stopButton", null);
    __publicField(this, "micButton", null);
    __publicField(this, "phoneButton", null);
    __publicField(this, "emotionButton", null);
    __publicField(this, "toggleButton", null);
    __publicField(this, "isOpen", false);
    __publicField(this, "isLoading", false);
    __publicField(this, "messages", []);
    __publicField(this, "apiClient", null);
    __publicField(this, "abortController", null);
    __publicField(this, "unreadCount", 0);
    __publicField(this, "styleElement", null);
    __publicField(this, "privacyAccepted", false);
    __publicField(this, "privacyModal", null);
    __publicField(this, "bounceTimer", null);
    __publicField(this, "bubbleElement", null);
    __publicField(this, "_warnedNoKey", false);
    __publicField(this, "audioCapabilities", null);
    __publicField(this, "ttsSentenceBuffer", "");
    __publicField(this, "ttsSegmentPlayer", null);
    __publicField(this, "ttsAudio", null);
    __publicField(this, "isRecording", false);
    __publicField(this, "phoneMode", false);
    __publicField(this, "awaitingUserSpeech", false);
    __publicField(this, "vadSpeechDetected", false);
    __publicField(this, "recordingStartTime", 0);
    __publicField(this, "phoneEmptyRetries", 0);
    __publicField(this, "mediaRecorder", null);
    __publicField(this, "audioChunks", []);
    __publicField(this, "audioContext", null);
    __publicField(this, "analyser", null);
    __publicField(this, "vadSilenceStart", 0);
    __publicField(this, "vadRafId", null);
    __publicField(this, "previewHtml", null);
    __publicField(this, "modalOverlay", null);
    __publicField(this, "emotionConfig", null);
    __publicField(this, "webcamConsent", false);
    __publicField(this, "webcamActive", false);
    __publicField(this, "emotionRuntimeStatus", "uninitialized");
    __publicField(this, "snippetBufferRef", null);
    __publicField(this, "audioSnippetBufferRef", null);
    __publicField(this, "emotionTimelineRef", null);
    __publicField(this, "emotionPollIntervalRef", null);
    __publicField(this, "lastEmotionModelRef", null);
    __publicField(this, "continuousAudioStreamRef", null);
    __publicField(this, "hasTransitionedToTypingRef", false);
    __publicField(this, "audioSnapshotsRef", []);
    __publicField(this, "userScrolledUpRef", false);
    __publicField(this, "isStartingAudioRef", false);
    __publicField(this, "isStartingWebcamRef", false);
    __publicField(this, "windowResizeHandler", null);
    this.userConfig = config;
    const lang = config.language && config.language !== "auto" ? config.language : detectLanguage();
    const i18n = WIDGET_I18N[lang];
    const privacyOverride = lang === "de" ? config.privacyTextDe : config.privacyTextEn;
    const initialOverride = lang === "de" ? config.initialMessageContentDe : config.initialMessageContentEn;
    const bounceOverride = lang === "de" ? config.bounceTextDe : config.bounceTextEn;
    const placeholderOverride = lang === "de" ? config.inputPlaceholderDe : config.inputPlaceholderEn;
    this.config = {
      apiKey: config.apiKey || void 0 || "",
      apiUrl: config.apiUrl || void 0 || window.location.origin + "/v1",
      model: config.model || void 0 || "CyberSecurityChatbot",
      position: config.position || "bottom-right",
      theme: config.theme || "light",
      language: lang,
      systemPrompt: config.systemPrompt || "",
      width: config.width || 650,
      height: config.height || 550,
      showOnLoad: config.showOnLoad ?? false,
      title: config.title || "CyberSecurityChatbot",
      subtitle: config.subtitle || "Powered by Agentix",
      logoUrl: config.logoUrl || "",
      logoUrlButton: config.logoUrlButton || config.logoUrl || WIDGET_FAVICON_BASE64,
      logoUrlHeader: config.logoUrlHeader || config.logoUrl || WIDGET_LOGO_BASE64,
      showLogoInHeader: config.showLogoInHeader ?? true,
      showLogoInButton: config.showLogoInButton ?? true,
      logoBorder: config.logoBorder || { enabled: false, color: "#ffffff", opacity: 0.35 },
      logoGlow: config.logoGlow || { enabled: true, color: "#ffffff", opacity: 0.4 },
      logoBackground: config.logoBackground || { enabled: true, color: "#ffffff", opacity: 0.35 },
      privacyText: privacyOverride || config.privacyText || i18n.privacyText,
      privacyTextDe: config.privacyTextDe || "",
      privacyTextEn: config.privacyTextEn || "",
      privacyEnabled: config.privacyEnabled ?? true,
      initialMessage: config.initialMessage ?? true,
      initialMessageContent: initialOverride || config.initialMessageContent || i18n.initialMessageContent,
      initialMessageContentDe: config.initialMessageContentDe || "",
      initialMessageContentEn: config.initialMessageContentEn || "",
      initialMessageSilent: config.initialMessageSilent ?? true,
      bounceInterval: config.bounceInterval ?? 15e3,
      bounceText: bounceOverride || config.bounceText || i18n.bounceText,
      bounceTextDe: config.bounceTextDe || "",
      bounceTextEn: config.bounceTextEn || "",
      inputPlaceholder: placeholderOverride || i18n.inputPlaceholder,
      inputPlaceholderDe: config.inputPlaceholderDe || "",
      inputPlaceholderEn: config.inputPlaceholderEn || "",
      colors: config.colors || {},
      onInit: config.onInit || (() => {
      }),
      onOpen: config.onOpen || (() => {
      }),
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
  async loadServerDefaults() {
    try {
      const baseUrl = this.config.apiUrl.replace(/\/v1$/, "");
      const [defRes, brandRes] = await Promise.allSettled([
        fetch(`${baseUrl}/widget-defaults`, { credentials: "omit" }),
        fetch(`${baseUrl}/branding`, { credentials: "omit" })
      ]);
      let defaults = null;
      if (defRes.status === "fulfilled" && defRes.value.ok) {
        defaults = await defRes.value.json();
      }
      const brandDefaults = {};
      if (brandRes.status === "fulfilled" && brandRes.value.ok) {
        const branding = await brandRes.value.json();
        if (branding && typeof branding === "object" && !Array.isArray(branding)) {
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
      const validDefaults = defaults && typeof defaults === "object" && !Array.isArray(defaults) ? defaults : {};
      const merged = { ...brandDefaults, ...validDefaults };
      if (Object.keys(merged).length === 0) return;
      const filtered = mergeServerDefaults(
        this.userConfig,
        merged
      );
      if (Object.keys(filtered).length === 0) return;
      this.updateConfig(filtered);
      console.log("[Widget] Merged server defaults:", filtered);
    } catch {
    }
  }
  /**
   * Validate configuration
   */
  validateConfig() {
    if (!this.config.apiKey && !this._warnedNoKey) {
      this._warnedNoKey = true;
      const warning = new Error("No API key provided. Widget will attempt unauthenticated requests; ensure your API allows this.");
      this.config.onError(warning);
    }
  }
  /**
   * Initialize widget
   */
  init() {
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
      const err = error instanceof Error ? error : new Error("Widget initialization failed");
      this.config.onError(err);
    }
  }
  /**
   * Check privacy consent status
   */
  checkPrivacyStatus() {
    var _a, _b, _c, _d, _e;
    if (!this.config.privacyEnabled) {
      this.privacyAccepted = true;
      this.enableChat();
      if (this.privacyModal) {
        this.privacyModal.style.display = "none";
      }
      if (this.window && this.shouldOpenOnLoad()) {
        this.isOpen = true;
        this.window.classList.add("open");
        (_a = this.toggleButton) == null ? void 0 : _a.setAttribute("aria-expanded", "true");
        this.unreadCount = 0;
        this.updateBadge();
        (_b = this.inputField) == null ? void 0 : _b.focus();
        this.config.onOpen();
        this.sendInitialMessageIfNeeded();
      }
      return;
    }
    const accepted = localStorage.getItem("agentix-chat-privacy-accepted");
    if (accepted === "true") {
      this.privacyAccepted = true;
      this.enableChat();
      if (this.window && this.shouldOpenOnLoad()) {
        this.isOpen = true;
        this.window.classList.add("open");
        (_c = this.toggleButton) == null ? void 0 : _c.setAttribute("aria-expanded", "true");
        this.unreadCount = 0;
        this.updateBadge();
        (_d = this.inputField) == null ? void 0 : _d.focus();
        this.config.onOpen();
        this.sendInitialMessageIfNeeded();
      }
    } else {
      this.privacyAccepted = false;
      this.disableChat();
      if (this.window && this.shouldOpenOnLoad()) {
        this.isOpen = true;
        this.window.classList.add("open");
        (_e = this.toggleButton) == null ? void 0 : _e.setAttribute("aria-expanded", "true");
        this.unreadCount = 0;
        this.updateBadge();
        if (this.privacyModal) {
          this.privacyModal.style.display = "flex";
        }
      }
    }
  }
  /**
   * Determine if widget should open on load based on stored state
   */
  shouldOpenOnLoad() {
    const stored = localStorage.getItem("agentix-chat-open");
    if (stored === null) return this.config.showOnLoad;
    return stored === "true";
  }
  /**
   * Setup OpenAI API client
   */
  setupAPIClient() {
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
  async fetchAudioCapabilities() {
    var _a, _b, _c, _d, _e, _f;
    try {
      const baseUrl = this.config.apiUrl.replace(/\/v1$/, "");
      const url = `${baseUrl}/audio/config?model=${encodeURIComponent(this.config.model)}`;
      console.log("[Widget] Fetching audio config:", url);
      const res = await fetch(url, {
        credentials: "include"
      });
      if (!res.ok) {
        console.warn("[Widget] Audio config fetch failed:", res.status, await res.text());
        this.audioCapabilities = null;
        if (this.micButton) this.micButton.style.display = "none";
        if (this.phoneButton) this.phoneButton.style.display = "none";
        return;
      }
      this.audioCapabilities = await res.json();
      console.log("[Widget] Audio capabilities:", this.audioCapabilities);
      if (this.micButton) {
        this.micButton.style.display = ((_b = (_a = this.audioCapabilities) == null ? void 0 : _a.stt) == null ? void 0 : _b.button_visible) ? "" : "none";
      }
      if (this.phoneButton) {
        this.phoneButton.style.display = ((_d = (_c = this.audioCapabilities) == null ? void 0 : _c.phone) == null ? void 0 : _d.button_visible) ? "" : "none";
      }
      const speakWrappers = (_e = this.container) == null ? void 0 : _e.querySelectorAll(".agentix-chat-message-actions");
      if (speakWrappers) {
        speakWrappers.forEach((wrapper) => {
          var _a2, _b2;
          wrapper.style.display = ((_b2 = (_a2 = this.audioCapabilities) == null ? void 0 : _a2.tts) == null ? void 0 : _b2.button_visible) ? "" : "none";
        });
      }
    } catch (err) {
      console.warn("[Widget] Audio config error:", err);
      this.audioCapabilities = null;
      if (this.micButton) this.micButton.style.display = "none";
      if (this.phoneButton) this.phoneButton.style.display = "none";
      const speakWrappers = (_f = this.container) == null ? void 0 : _f.querySelectorAll(".agentix-chat-message-actions");
      if (speakWrappers) {
        speakWrappers.forEach((wrapper) => {
          wrapper.style.display = "none";
        });
      }
    }
  }
  /**
   * Fetch emotion config from backend
   */
  async fetchEmotionConfig() {
    if (this.lastEmotionModelRef === this.config.model) return;
    try {
      const baseUrl = this.config.apiUrl.replace(/\/v1$/, "");
      const res = await fetch(`${baseUrl}/emotion/config?model=${encodeURIComponent(this.config.model)}`, {
        credentials: "include"
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
  updateEmotionButtonVisibility() {
    var _a, _b, _c;
    if (!this.emotionButton) return;
    if (this.phoneMode) {
      this.emotionButton.style.display = ((_a = this.emotionConfig) == null ? void 0 : _a.enabled) ? "" : "none";
    } else {
      this.emotionButton.style.display = ((_c = (_b = this.emotionConfig) == null ? void 0 : _b.webcam) == null ? void 0 : _c.enabled) ? "" : "none";
    }
  }
  /**
   * Smart scroll to bottom (respects user scroll up)
   */
  smartScrollToBottom() {
    if (!this.userScrolledUpRef && this.messagesContainer) {
      scrollToBottom(this.messagesContainer);
    }
  }
  /**
   * Start webcam emotion detection
   */
  async startWebcamEmotion() {
    var _a, _b, _c, _d;
    if (!((_b = (_a = this.emotionConfig) == null ? void 0 : _a.webcam) == null ? void 0 : _b.enabled) || this.webcamActive) return;
    if (this.isStartingWebcamRef) return;
    if (!this.webcamConsent) {
      const agreed = window.confirm(
        "Webcam-Emotion Detection\n\nDie Kamera wird lokal im Browser ausgewertet. Es werden keine Bilder oder Videos aufgezeichnet oder übertragen. Nur aggregierte Emotions-Metadaten werden gesendet.\n\nMöchten Sie fortfahren?"
      );
      if (!agreed) return;
      this.webcamConsent = true;
    }
    this.isStartingWebcamRef = true;
    this.emotionRuntimeStatus = "loading";
    try {
      const cfg = this.emotionConfig.webcam;
      const snippetConfig = {
        windowMs: cfg.snippet_window_ms,
        sampleFps: cfg.sample_fps,
        maxFrames: cfg.max_frames_per_snippet,
        minValidFaceFrames: cfg.min_valid_face_frames,
        faceModelUrl: cfg.face_model_url,
        expressionModelUrl: cfg.expression_model_url,
        executionProviders: cfg.execution_providers
      };
      this.snippetBufferRef = createVideoSnippetBuffer(snippetConfig);
      await this.snippetBufferRef.start();
      if (this.isStartingWebcamRef) {
        if ((_d = (_c = this.emotionConfig) == null ? void 0 : _c.audio) == null ? void 0 : _d.enabled) {
          await this.startContinuousAudioEmotion().catch(() => {
          });
        }
        this.webcamActive = true;
        this.emotionRuntimeStatus = "ready";
      }
    } catch {
      this.emotionRuntimeStatus = "error";
      this.webcamActive = false;
    } finally {
      this.isStartingWebcamRef = false;
    }
  }
  /**
   * Stop webcam emotion detection
   */
  stopWebcamEmotion() {
    this.isStartingWebcamRef = false;
    if (this.snippetBufferRef) {
      this.snippetBufferRef.stop();
      this.snippetBufferRef = null;
    }
    this.stopContinuousAudioEmotion();
    this.webcamActive = false;
    this.emotionRuntimeStatus = "uninitialized";
  }
  /**
   * Start continuous audio emotion
   */
  async startContinuousAudioEmotion() {
    var _a, _b, _c;
    if ((_a = this.audioSnippetBufferRef) == null ? void 0 : _a.isActive()) return;
    if (this.isStartingAudioRef) return;
    if (!((_c = (_b = this.emotionConfig) == null ? void 0 : _b.audio) == null ? void 0 : _c.enabled)) {
      console.info("[Widget] Audio emotion disabled in config");
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
      const audioCfg = {
        windowMs: (cfg == null ? void 0 : cfg.snippet_window_ms) || 3e3,
        sampleFps: (cfg == null ? void 0 : cfg.sample_fps) || 3,
        maxFrames: (cfg == null ? void 0 : cfg.max_frames_per_snippet) || 12,
        minValidFaceFrames: (cfg == null ? void 0 : cfg.min_valid_face_frames) || 3,
        faceModelUrl: (cfg == null ? void 0 : cfg.face_model_url) || "/models/emotion/face-detector.onnx",
        expressionModelUrl: this.emotionConfig.audio.model_url || "/models/emotion/audio-emotion.onnx",
        executionProviders: (cfg == null ? void 0 : cfg.execution_providers) || ["wasm"],
        minSnrDb: this.emotionConfig.audio.min_snr_db ?? 10,
        noiseGateMin: this.emotionConfig.audio.noise_gate_min ?? 1e-3,
        noiseGateMax: this.emotionConfig.audio.noise_gate_max ?? 0.5
      };
      console.info("[Widget] Starting continuous audio emotion with modelUrl:", audioCfg.expressionModelUrl);
      this.audioSnippetBufferRef = createAudioSnippetBuffer(audioCfg);
      this.audioSnippetBufferRef.start(stream);
      console.info("[Widget] Continuous audio emotion started");
    } catch (err) {
      console.warn("[Widget] Audio emotion start failed:", err);
    } finally {
      this.isStartingAudioRef = false;
    }
  }
  /**
   * Stop continuous audio emotion
   */
  stopContinuousAudioEmotion() {
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
  async capturePreSendEmotion() {
    if (!this.snippetBufferRef || !this.snippetBufferRef.isActive()) return null;
    return this.snippetBufferRef.captureSnapshot();
  }
  /**
   * Build emotion payload for API request
   */
  buildEmotionPayload() {
    var _a, _b;
    if (!((_a = this.emotionConfig) == null ? void 0 : _a.enabled)) return void 0;
    const payload = {};
    payload.consent = {
      webcam_opt_in: this.webcamConsent,
      frames_sent_to_backend: false,
      audio_opt_in: ((_b = this.audioSnippetBufferRef) == null ? void 0 : _b.isActive()) ?? false
    };
    return payload;
  }
  /**
   * Start continuous emotion polling
   */
  startEmotionPolling(initialPhase = "waiting_for_response") {
    if (this.emotionPollIntervalRef) return;
    this.hasTransitionedToTypingRef = false;
    this.audioSnapshotsRef = [];
    const phases = this.createEmptyPhases();
    phases[initialPhase].startTime = Date.now();
    this.emotionTimelineRef = {
      phases,
      activePhase: initialPhase,
      startTime: Date.now()
    };
    this.emotionPollIntervalRef = setInterval(() => {
      this.processEmotionBatch();
    }, 2e3);
  }
  /**
   * Stop continuous emotion polling
   */
  stopEmotionPolling() {
    if (this.emotionPollIntervalRef) {
      clearInterval(this.emotionPollIntervalRef);
      this.emotionPollIntervalRef = null;
    }
    this.processEmotionBatch();
  }
  /**
   * Switch emotion phase
   */
  switchEmotionPhase(nextPhase) {
    const tl = this.emotionTimelineRef;
    if (!tl || tl.activePhase === nextPhase) return;
    const current = tl.phases[tl.activePhase];
    if (current.snapshots.length > 0) {
      const fused = fuseEmotions(current.snapshots);
      if (fused) {
        tl.phases[tl.activePhase] = {
          snapshots: [fused],
          startTime: current.startTime,
          endTime: Date.now()
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
  createEmptyPhases() {
    return {
      waiting_for_response: { snapshots: [], startTime: Date.now() },
      reading: { snapshots: [], startTime: 0 },
      typing: { snapshots: [], startTime: 0 },
      speaking: { snapshots: [], startTime: 0 }
    };
  }
  /**
   * Process emotion batch
   */
  processEmotionBatch() {
    var _a, _b;
    const tl = this.emotionTimelineRef;
    if (!tl) return;
    if ((_a = this.snippetBufferRef) == null ? void 0 : _a.isActive()) {
      this.snippetBufferRef.captureSnapshot().then((res) => {
        if (res) tl.phases[tl.activePhase].snapshots.push(res);
      }).catch(() => {
      });
    }
    if ((_b = this.audioSnippetBufferRef) == null ? void 0 : _b.isActive()) {
      this.audioSnippetBufferRef.captureSnapshot().then((res) => {
        if (res) {
          tl.phases[tl.activePhase].snapshots.push(res);
          this.audioSnapshotsRef.push(res);
        }
      }).catch(() => {
      });
    }
  }
  /**
   * Build emotion phases
   */
  buildEmotionPhases() {
    const tl = this.emotionTimelineRef;
    if (!tl) return [];
    const result = [];
    const phaseOrder = ["waiting_for_response", "reading", "typing", "speaking"];
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
            window_ms: fused.window_ms
          });
          continue;
        }
      }
      if (data.startTime > 0 || phase === tl.activePhase) {
        result.push({
          phase,
          primary_emotion: "unknown",
          confidence: 0,
          intensity: "low",
          source: `phase__${phase}`,
          timestamp_ms: data.startTime || Date.now()
        });
      }
    }
    return result;
  }
  /**
   * Compute overall fused emotion
   */
  computeOverallFused(phases) {
    if (phases.length === 0) return null;
    const asResults = phases.map((p) => ({
      primary_emotion: p.primary_emotion,
      confidence: p.confidence,
      intensity: p.intensity,
      source: p.source,
      timestamp_ms: p.timestamp_ms,
      sampled_frames: p.sampled_frames,
      valid_face_frames: p.valid_face_frames,
      face_detection_rate: p.face_detection_rate,
      window_ms: p.window_ms
    }));
    return fuseEmotions(asResults);
  }
  /**
   * Clean text for TTS: remove URLs, markdown, HTML tags, and emojis
   */
  cleanTextForTTS(text) {
    let clean = text;
    clean = clean.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
    clean = clean.replace(/https?:\/\/[^\s]+/g, "Link");
    clean = clean.replace(/^#{1,6}\s+/gm, "");
    clean = clean.replace(/[*_`#]/g, "");
    clean = clean.replace(/^[\-*]\s+/gm, "");
    clean = clean.replace(/^\d+\.\s+/gm, "");
    clean = clean.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{FE00}-\u{FE0F}\u{1F900}-\u{1F9FF}\u{1F018}-\u{1F270}\u{238C}\u{2B06}\u{2B07}\u{2B05}\u{27A1}\u{2194}-\u{21AA}]/gu, "");
    clean = clean.replace(/<[^>]+>/g, "");
    clean = clean.replace(/[ \t]+/g, " ");
    clean = clean.split("\n").map((l) => l.trim()).join("\n");
    clean = clean.replace(/\n{3,}/g, "\n\n");
    return clean.trim();
  }
  /**
   * Stop all TTS playback (single audio and streaming player)
   */
  stopAllTTS() {
    var _a;
    if (this.ttsAudio) {
      this.ttsAudio.pause();
      this.ttsAudio = null;
    }
    (_a = this.ttsSegmentPlayer) == null ? void 0 : _a.abort();
    this.ttsSegmentPlayer = null;
    this.ttsSentenceBuffer = "";
    this.switchEmotionPhase("typing");
    if (this.messagesContainer) {
      this.messagesContainer.querySelectorAll(".agentix-chat-speaking").forEach((el) => el.classList.remove("agentix-chat-speaking"));
      this.messagesContainer.querySelectorAll(".agentix-chat-speak-btn.speaking").forEach((el) => {
        el.classList.remove("speaking");
        const playIcon = el.querySelector(".speak-icon");
        const stopIcon = el.querySelector(".speak-stop-icon");
        if (playIcon) playIcon.style.display = "";
        if (stopIcon) stopIcon.style.display = "none";
      });
    }
  }
  /**
   * Speak text via TTS (toggle: click again to stop)
   */
  async speakText(text, messageElement, speakBtn) {
    var _a, _b, _c, _d;
    if (!((_b = (_a = this.audioCapabilities) == null ? void 0 : _a.tts) == null ? void 0 : _b.button_visible) || !((_d = (_c = this.audioCapabilities) == null ? void 0 : _c.tts) == null ? void 0 : _d.voice_id)) return;
    if (this.ttsAudio || this.ttsSegmentPlayer && this.ttsSegmentPlayer.queueLength > 0) {
      this.stopAllTTS();
      return;
    }
    const cleanText = this.cleanTextForTTS(text);
    if (!cleanText) return;
    const voiceId = this.audioCapabilities.tts.voice_id;
    const stopSpeaking = () => {
      messageElement.classList.remove("agentix-chat-speaking");
      speakBtn == null ? void 0 : speakBtn.classList.remove("speaking");
      const playIcon = speakBtn == null ? void 0 : speakBtn.querySelector(".speak-icon");
      const stopIcon = speakBtn == null ? void 0 : speakBtn.querySelector(".speak-stop-icon");
      if (playIcon) playIcon.style.display = "";
      if (stopIcon) stopIcon.style.display = "none";
      this.switchEmotionPhase("typing");
      if (this.phoneMode) {
        this.awaitingUserSpeech = true;
        this.updateMicButton();
        setTimeout(() => this.startRecording(), 300);
      }
    };
    try {
      messageElement.classList.add("agentix-chat-speaking");
      speakBtn == null ? void 0 : speakBtn.classList.add("speaking");
      this.switchEmotionPhase("speaking");
      const playIcon = speakBtn == null ? void 0 : speakBtn.querySelector(".speak-icon");
      const stopIcon = speakBtn == null ? void 0 : speakBtn.querySelector(".speak-stop-icon");
      if (playIcon) playIcon.style.display = "none";
      if (stopIcon) stopIcon.style.display = "";
      this.ttsSegmentPlayer = new TTSSegmentPlayer();
      const { chunks } = extractCompleteSentences(cleanText);
      for (const chunk of chunks) {
        this.ttsSegmentPlayer.add(chunk.text, voiceId, this.fetchTTSSegment.bind(this), chunk.pauseAfterMs);
      }
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
      messageElement.classList.remove("agentix-chat-speaking");
      speakBtn == null ? void 0 : speakBtn.classList.remove("speaking");
      const playIcon = speakBtn == null ? void 0 : speakBtn.querySelector(".speak-icon");
      const stopIcon = speakBtn == null ? void 0 : speakBtn.querySelector(".speak-stop-icon");
      if (playIcon) playIcon.style.display = "";
      if (stopIcon) stopIcon.style.display = "none";
      this.ttsSegmentPlayer = null;
    }
  }
  /**
   * STT Recording
   */
  async startRecording() {
    var _a, _b, _c, _d, _e;
    console.info("[Widget] startRecording called, isRecording=", this.isRecording);
    if (!((_b = (_a = this.audioCapabilities) == null ? void 0 : _a.stt) == null ? void 0 : _b.button_visible) || this.isRecording) {
      console.info("[Widget] startRecording early return");
      return;
    }
    if (!this.webcamActive && ((_d = (_c = this.emotionConfig) == null ? void 0 : _c.audio) == null ? void 0 : _d.enabled) && !((_e = this.audioSnippetBufferRef) == null ? void 0 : _e.isActive())) {
      console.info("[Widget] startRecording: auto-starting temporary audio emotion buffer");
      await this.startContinuousAudioEmotion().catch(() => {
      });
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : MediaRecorder.isTypeSupported("audio/ogg") ? "audio/ogg" : "";
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : void 0);
      this.mediaRecorder = recorder;
      this.audioChunks = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) this.audioChunks.push(e.data);
      };
      recorder.onstop = async () => {
        var _a2, _b2, _c2, _d2;
        console.info("[Widget] recorder.onstop fired");
        stream.getTracks().forEach((t) => t.stop());
        this.isRecording = false;
        this.updateMicButton();
        let audioEmotion = null;
        console.info("[Widget] STT stop: audioSnippetBuffer active=", (_a2 = this.audioSnippetBufferRef) == null ? void 0 : _a2.isActive());
        if ((_b2 = this.audioSnippetBufferRef) == null ? void 0 : _b2.isActive()) {
          audioEmotion = await this.audioSnippetBufferRef.captureSnapshot();
          console.info("[Widget] STT captureSnapshot result:", audioEmotion);
        }
        if (!this.webcamActive) {
          this.stopContinuousAudioEmotion();
        }
        if (this.audioChunks.length === 0) return;
        const audioBlob = new Blob(this.audioChunks, { type: mimeType || "audio/webm" });
        try {
          const wavBlob = await blobToWav(audioBlob);
          const formData = new FormData();
          formData.append("audio", wavBlob, "recording.wav");
          if ((_d2 = (_c2 = this.audioCapabilities) == null ? void 0 : _c2.stt) == null ? void 0 : _d2.language) {
            formData.append("language", this.audioCapabilities.stt.language);
          }
          const baseUrl = this.config.apiUrl.replace(/\/v1$/, "");
          const res = await fetch(`${baseUrl}/audio/stt`, {
            method: "POST",
            credentials: "include",
            body: formData
          });
          if (!res.ok) {
            if (res.status === 400) {
              const i18n = WIDGET_I18N[this.config.language];
              this._showToast(i18n.sttEmpty || "Ich konnte dich leider nicht verstehen.");
            }
            throw new Error(await res.text());
          }
          const data = await res.json();
          const transcript = (data.text || "").trim();
          const hasAudioEmotions = audioEmotion;
          if (transcript.length >= 2 && this.inputField) {
            this.inputField.value = transcript;
            this.autoResizeTextarea();
          }
          if (transcript.length >= 2 || hasAudioEmotions) {
            this.phoneEmptyRetries = 0;
            setTimeout(() => {
              if (this.inputField) this.inputField.value = "";
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
        } catch (err) {
          console.error("STT error:", err);
        }
      };
      recorder.start();
      this.isRecording = true;
      this.awaitingUserSpeech = false;
      this.updateMicButton();
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
          if (this.vadSpeechDetected && now - this.recordingStartTime > 2e3 && rms < 0.02) {
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
        if (timeoutRecorder.state === "recording") {
          timeoutRecorder.stop();
        }
      }, 3e4);
    } catch (err) {
      const helpMsg = "Mikrofon-Zugriff verweigert.\n\nBitte pruefe:\n1. Browser-Adressleiste -> Schloss -> Berechtigungen -> Mikrofon -> Erlauben\n2. macOS -> Systemeinstellungen -> Datenschutz & Sicherheit -> Mikrofon -> Browser aktivieren\n3. Seite danach neu laden (F5).";
      alert(helpMsg);
    }
  }
  stopRecording() {
    var _a;
    console.info("[Widget] stopRecording called, isRecording=", this.isRecording);
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
    if (this.mediaRecorder && this.mediaRecorder.state === "recording") {
      this.mediaRecorder.stop();
      (_a = this.mediaRecorder.stream) == null ? void 0 : _a.getTracks().forEach((t) => t.stop());
    }
    this.isRecording = false;
  }
  toggleRecording() {
    if (this.isRecording) {
      this.stopRecording();
    } else {
      this.startRecording();
    }
  }
  togglePhoneMode() {
    var _a, _b, _c, _d, _e, _f, _g, _h, _i;
    this.phoneMode = !this.phoneMode;
    const inputInner = (_a = this.inputField) == null ? void 0 : _a.parentElement;
    if (this.phoneMode) {
      this.phoneEmptyRetries = 0;
      if (!this.isRecording) {
        setTimeout(() => this.startRecording(), 100);
      }
      (_b = this.inputField) == null ? void 0 : _b.classList.add("phone-mode-hidden");
      (_c = this.sendButton) == null ? void 0 : _c.classList.add("phone-mode-hidden");
      (_d = this.stopButton) == null ? void 0 : _d.classList.add("phone-mode-hidden");
      (_e = this.micButton) == null ? void 0 : _e.classList.add("phone-mode-hidden");
      inputInner == null ? void 0 : inputInner.classList.add("phone-mode");
    } else {
      this.stopRecording();
      this.stopAllTTS();
      if (!this.webcamActive) {
        this.stopEmotionPolling();
        this.stopContinuousAudioEmotion();
      }
      this.awaitingUserSpeech = false;
      this.phoneEmptyRetries = 0;
      (_f = this.inputField) == null ? void 0 : _f.classList.remove("phone-mode-hidden");
      (_g = this.sendButton) == null ? void 0 : _g.classList.remove("phone-mode-hidden");
      if (this.sendButton) this.sendButton.style.display = "";
      (_h = this.stopButton) == null ? void 0 : _h.classList.remove("phone-mode-hidden");
      (_i = this.micButton) == null ? void 0 : _i.classList.remove("phone-mode-hidden");
      inputInner == null ? void 0 : inputInner.classList.remove("phone-mode");
    }
    this.updateEmotionButtonVisibility();
    this.updateMicButton();
    this.updatePhoneButton();
  }
  updatePhoneButton() {
    if (!this.phoneButton) return;
    const svg = this.phoneButton.querySelector("svg");
    if (this.phoneMode) {
      this.phoneButton.classList.add("active");
      if (svg) {
        svg.innerHTML = '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/><line x1="2" y1="2" x2="22" y2="22"/>';
      }
    } else {
      this.phoneButton.classList.remove("active");
      if (svg) {
        svg.innerHTML = '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/>';
      }
    }
  }
  updateEmotionButton(btn) {
    const target = btn || this.emotionButton;
    if (!target) return;
    target.classList.toggle("active", this.webcamActive);
    target.classList.toggle("error", this.emotionRuntimeStatus === "error");
    target.title = this.webcamActive ? "Emotion Detection aktiv" : "Emotion Detection starten";
  }
  updateMicButton() {
    var _a, _b;
    if (!this.micButton) return;
    this.micButton.style.display = ((_b = (_a = this.audioCapabilities) == null ? void 0 : _a.stt) == null ? void 0 : _b.button_visible) ? "" : "none";
    if (this.isRecording) {
      this.micButton.classList.add("recording");
      this.micButton.setAttribute("title", "Aufnahme stoppen");
    } else {
      this.micButton.classList.remove("recording");
      this.micButton.setAttribute("title", "Spracheingabe");
    }
    if (this.awaitingUserSpeech) {
      this.micButton.classList.add("awaiting");
    } else {
      this.micButton.classList.remove("awaiting");
    }
  }
  /**
   * Return existing session conversation id or create a new one.
   * Widget IDs are prefixed with `widget-` per spec.
   */
  getOrCreateSessionConversationId() {
    try {
      const key = "agentix-conversation-id";
      const existing = sessionStorage.getItem(key);
      if (existing && existing.trim()) return existing;
      const id = `widget-${crypto.randomUUID ? crypto.randomUUID() : String(Date.now())}`;
      sessionStorage.setItem(key, id);
      return id;
    } catch (err) {
      return `widget-${String(Date.now())}`;
    }
  }
  /**
   * Inject CSS styles into the document
   */
  injectStyles() {
    if (this.styleElement) return;
    this.styleElement = document.createElement("style");
    this.styleElement.textContent = WIDGET_STYLES;
    document.head.appendChild(this.styleElement);
  }
  /**
   * Apply custom color CSS variables to the widget container
   */
  applyColorVariables() {
    if (!this.container) return;
    const c = this.config.colors;
    const map = {
      "--chat-primary": c.primary,
      "--chat-header-bg": c.headerBackground,
      "--chat-header-text": c.headerText,
      "--chat-bg": c.background,
      "--chat-text": c.text,
      "--chat-user-msg-bg": c.userMessageBackground,
      "--chat-user-msg-text": c.primary ? "#ffffff" : void 0,
      "--chat-assistant-msg-bg": c.assistantMessageBackground,
      "--chat-assistant-msg-text": c.text,
      "--chat-input-bg": c.inputBackground,
      "--chat-input-text": c.inputText,
      "--chat-input-border": c.primary,
      "--chat-send-btn-bg": c.buttonBackground,
      "--chat-send-btn-text": c.buttonText
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
  hexToRgba(hex, opacity) {
    const clean = hex.replace("#", "");
    let r, g, b;
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
  createLogoElement(className, url) {
    const logoImg = createElement("img", {
      class: className,
      src: url,
      alt: "Logo",
      onerror: 'this.style.display="none"'
    });
    const border = this.config.logoBorder;
    const glow = this.config.logoGlow;
    const bg = this.config.logoBackground;
    const needsWrapper = (border == null ? void 0 : border.enabled) || (glow == null ? void 0 : glow.enabled) || (bg == null ? void 0 : bg.enabled);
    if (needsWrapper) {
      const wrapper = createElement("span", {
        class: "agentix-chat-logo-wrapper"
      });
      wrapper.style.display = "inline-flex";
      wrapper.style.padding = "2px";
      wrapper.style.borderRadius = "6px";
      if (bg == null ? void 0 : bg.enabled) {
        const bgColor = this.hexToRgba(bg.color || "#ffffff", bg.opacity ?? 0.35);
        wrapper.style.background = bgColor;
      }
      if (border == null ? void 0 : border.enabled) {
        const borderColor = this.hexToRgba(border.color || "#ffffff", border.opacity ?? 0.35);
        wrapper.style.border = `1px solid ${borderColor}`;
      }
      if (glow == null ? void 0 : glow.enabled) {
        const glowColor = this.hexToRgba(glow.color || "#ffffff", glow.opacity ?? 0.4);
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
  refreshHeaderLogo() {
    var _a;
    const header = (_a = this.container) == null ? void 0 : _a.querySelector(".agentix-chat-header");
    if (!header) return;
    header.querySelectorAll(".agentix-chat-header-logo, .agentix-chat-logo-wrapper").forEach((el) => el.remove());
    if (this.config.logoUrlHeader && this.config.showLogoInHeader) {
      const logoEl = this.createLogoElement("agentix-chat-header-logo", this.config.logoUrlHeader);
      header.insertBefore(logoEl, header.firstChild);
    }
  }
  /**
   * Rebuild the toggle-button content (logo element or 💬 fallback).
   */
  refreshToggleButtonLogo() {
    if (!this.toggleButton) return;
    this.toggleButton.innerHTML = "";
    if (this.config.logoUrlButton && this.config.showLogoInButton) {
      const logoEl = this.createLogoElement("", this.config.logoUrlButton);
      const img = logoEl.querySelector("img");
      if (img) {
        img.onerror = function() {
          this.style.display = "none";
          const parent = this.closest(".agentix-chat-logo-wrapper");
          if (parent) {
            parent.textContent = "💬";
          }
        };
      }
      this.toggleButton.appendChild(logoEl);
    } else {
      this.toggleButton.appendChild(document.createTextNode("💬"));
    }
  }
  /**
   * Create DOM structure
   */
  createDOM() {
    this.container = createElement("div", {
      class: `agentix-chat-widget-container ${this.config.position} ${this.config.theme}`,
      id: "agentix-chat-widget"
    });
    this.toggleButton = createElement("button", {
      class: "agentix-chat-toggle-btn",
      id: "agentix-chat-toggle",
      "aria-label": "Open chat",
      "aria-expanded": "false"
    });
    this.refreshToggleButtonLogo();
    this.container.appendChild(this.toggleButton);
    this.window = createElement("div", {
      class: `agentix-chat-window ${this.config.position}`,
      id: "agentix-chat-window",
      role: "dialog",
      "aria-labelledby": "agentix-chat-title"
    });
    this.window.style.width = `${this.config.width}px`;
    this.window.style.height = `${this.config.height}px`;
    const header = createElement("div", {
      class: "agentix-chat-header"
    });
    if (this.config.logoUrlHeader && this.config.showLogoInHeader) {
      const logoEl = this.createLogoElement("agentix-chat-header-logo", this.config.logoUrlHeader);
      header.appendChild(logoEl);
    }
    const headerInfo = createElement("div", {
      class: "agentix-chat-header-info"
    });
    const title = createElement("h2", {
      class: "agentix-chat-header-title",
      id: "agentix-chat-title"
    }, this.config.title);
    const subtitle = createElement("p", {
      class: "agentix-chat-header-subtitle"
    }, this.config.subtitle);
    headerInfo.appendChild(title);
    headerInfo.appendChild(subtitle);
    const closeBtn = createElement("button", {
      class: "agentix-chat-close-btn",
      "aria-label": "Close chat"
    }, "✕");
    header.appendChild(headerInfo);
    header.appendChild(closeBtn);
    this.messagesContainer = createElement("div", {
      class: "agentix-chat-messages",
      role: "log",
      "aria-live": "polite"
    });
    this.messagesContainer.addEventListener("scroll", () => {
      const nearBottom = this.messagesContainer.scrollHeight - this.messagesContainer.scrollTop - this.messagesContainer.clientHeight < 100;
      this.userScrolledUpRef = !nearBottom;
    });
    this.showEmptyState();
    const inputBar = createElement("div", {
      class: "agentix-chat-input-bar"
    });
    const inputInner = createElement("div", {
      class: "agentix-chat-input-inner"
    });
    this.inputField = createElement("textarea", {
      class: "agentix-chat-input-field",
      placeholder: this.config.inputPlaceholder,
      "aria-label": "Message input",
      rows: "1"
    });
    const phoneButton = createElement("button", {
      class: "agentix-chat-phone-btn",
      "aria-label": "Phone mode",
      style: "display:none"
    });
    const phoneSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    phoneSvg.setAttribute("width", "20");
    phoneSvg.setAttribute("height", "20");
    phoneSvg.setAttribute("viewBox", "0 0 24 24");
    phoneSvg.setAttribute("fill", "none");
    phoneSvg.setAttribute("stroke", "currentColor");
    phoneSvg.setAttribute("stroke-width", "2");
    phoneSvg.setAttribute("stroke-linecap", "round");
    phoneSvg.setAttribute("stroke-linejoin", "round");
    phoneSvg.innerHTML = '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/>';
    phoneButton.appendChild(phoneSvg);
    this.phoneButton = phoneButton;
    phoneButton.addEventListener("click", () => {
      this.togglePhoneMode();
    });
    this.micButton = createElement("button", {
      class: "agentix-chat-mic-btn",
      "aria-label": "Voice input",
      style: "display:none"
    });
    const micSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    micSvg.setAttribute("width", "20");
    micSvg.setAttribute("height", "20");
    micSvg.setAttribute("viewBox", "0 0 24 24");
    micSvg.setAttribute("fill", "none");
    micSvg.setAttribute("stroke", "currentColor");
    micSvg.setAttribute("stroke-width", "2");
    micSvg.setAttribute("stroke-linecap", "round");
    micSvg.setAttribute("stroke-linejoin", "round");
    micSvg.innerHTML = '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="23"/><line x1="8" x2="16" y1="23" y2="23"/>';
    this.micButton.appendChild(micSvg);
    this.micButton.addEventListener("click", () => this.toggleRecording());
    const emotionButton = createElement("button", {
      class: "agentix-chat-emotion-btn",
      "aria-label": "Emotion Detection",
      style: "display:none"
    });
    const emotionSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    emotionSvg.setAttribute("width", "20");
    emotionSvg.setAttribute("height", "20");
    emotionSvg.setAttribute("viewBox", "0 0 24 24");
    emotionSvg.setAttribute("fill", "none");
    emotionSvg.setAttribute("stroke", "currentColor");
    emotionSvg.setAttribute("stroke-width", "2");
    emotionSvg.setAttribute("stroke-linecap", "round");
    emotionSvg.setAttribute("stroke-linejoin", "round");
    emotionSvg.innerHTML = '<circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>';
    emotionButton.appendChild(emotionSvg);
    emotionButton.addEventListener("click", () => {
      if (this.webcamActive) {
        this.stopWebcamEmotion();
      } else {
        this.startWebcamEmotion();
      }
      this.updateEmotionButton(emotionButton);
    });
    this.stopButton = createElement("button", {
      class: "agentix-chat-stop-btn",
      "aria-label": "Stop generation",
      style: "display:none"
    });
    const stopPulse = createElement("span", { class: "agentix-chat-stop-pulse" });
    const stopIcon = createElement("span", { class: "agentix-chat-stop-icon" });
    const stopSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    stopSvg.setAttribute("width", "14");
    stopSvg.setAttribute("height", "14");
    stopSvg.setAttribute("viewBox", "0 0 24 24");
    stopSvg.setAttribute("fill", "currentColor");
    stopSvg.setAttribute("stroke", "none");
    stopSvg.innerHTML = '<rect x="6" y="6" width="12" height="12" rx="2" />';
    stopIcon.appendChild(stopSvg);
    this.stopButton.appendChild(stopPulse);
    this.stopButton.appendChild(stopIcon);
    this.stopButton.addEventListener("click", () => {
      var _a;
      (_a = this.abortController) == null ? void 0 : _a.abort();
    });
    this.sendButton = createElement("button", {
      class: "agentix-chat-send-btn",
      "aria-label": "Send message"
    });
    const sendSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    sendSvg.setAttribute("width", "20");
    sendSvg.setAttribute("height", "20");
    sendSvg.setAttribute("viewBox", "0 0 24 24");
    sendSvg.setAttribute("fill", "none");
    sendSvg.setAttribute("stroke", "currentColor");
    sendSvg.setAttribute("stroke-width", "2");
    sendSvg.setAttribute("stroke-linecap", "round");
    sendSvg.setAttribute("stroke-linejoin", "round");
    sendSvg.innerHTML = '<line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>';
    this.sendButton.appendChild(sendSvg);
    inputInner.appendChild(this.inputField);
    inputInner.appendChild(phoneButton);
    inputInner.appendChild(this.micButton);
    inputInner.appendChild(emotionButton);
    inputInner.appendChild(this.stopButton);
    inputInner.appendChild(this.sendButton);
    this.emotionButton = emotionButton;
    inputBar.appendChild(inputInner);
    const resizeHandle = createElement("div", {
      class: "agentix-chat-resize-handle",
      "aria-label": "Resize chat window"
    });
    this.window.appendChild(header);
    this.window.appendChild(this.messagesContainer);
    this.window.appendChild(inputBar);
    this.window.appendChild(resizeHandle);
    this.container.appendChild(this.window);
    document.body.appendChild(this.container);
    this.createPrivacyModal();
  }
  /**
   * Create privacy consent modal
   */
  createPrivacyModal() {
    var _a;
    this.privacyModal = createElement("div", {
      class: "agentix-chat-privacy-modal",
      id: "agentix-chat-privacy-modal",
      role: "dialog",
      "aria-labelledby": "agentix-chat-privacy-title"
    });
    const modalContent = createElement("div", {
      class: "agentix-chat-privacy-content"
    });
    const title = createElement("h3", {
      class: "agentix-chat-privacy-title",
      id: "agentix-chat-privacy-title"
    }, WIDGET_I18N[this.config.language].privacyTitle);
    const text = createElement("p", {
      class: "agentix-chat-privacy-text"
    }, this.config.privacyText);
    const buttonContainer = createElement("div", {
      class: "agentix-chat-privacy-buttons"
    });
    const acceptBtn = createElement("button", {
      class: "agentix-chat-privacy-accept",
      id: "agentix-chat-privacy-accept"
    }, WIDGET_I18N[this.config.language].privacyAccept);
    const declineBtn = createElement("button", {
      class: "agentix-chat-privacy-decline",
      id: "agentix-chat-privacy-decline"
    }, WIDGET_I18N[this.config.language].privacyDecline);
    buttonContainer.appendChild(acceptBtn);
    buttonContainer.appendChild(declineBtn);
    modalContent.appendChild(title);
    modalContent.appendChild(text);
    modalContent.appendChild(buttonContainer);
    this.privacyModal.appendChild(modalContent);
    (_a = this.window) == null ? void 0 : _a.appendChild(this.privacyModal);
    addEvent(acceptBtn, "click", () => this.acceptPrivacy());
    addEvent(declineBtn, "click", () => this.declinePrivacy());
  }
  /**
   * Accept privacy policy
   */
  acceptPrivacy() {
    this.privacyAccepted = true;
    localStorage.setItem("agentix-chat-privacy-accepted", "true");
    this.hidePrivacyModal();
    this.enableChat();
    if (this.isOpen) {
      this.config.onOpen();
      this.sendInitialMessageIfNeeded();
    }
  }
  /**
   * Send initial greeting message if conditions are met
   */
  sendInitialMessageIfNeeded() {
    if (!this.config.initialMessage || this.messages.length > 0 || !this.privacyAccepted) return;
    this.sendUserMessage(this.config.initialMessageContent, { silent: this.config.initialMessageSilent });
  }
  /**
   * Decline privacy policy
   */
  declinePrivacy() {
    this.privacyAccepted = false;
    localStorage.setItem("agentix-chat-privacy-accepted", "false");
    this.hidePrivacyModal();
    this.disableChat();
  }
  /**
   * Hide privacy modal
   */
  hidePrivacyModal() {
    if (this.privacyModal) {
      this.privacyModal.style.display = "none";
    }
  }
  /**
   * Enable chat input
   */
  enableChat() {
    if (this.inputField) {
      this.inputField.disabled = false;
      this.inputField.placeholder = this.config.inputPlaceholder;
    }
    if (this.sendButton) {
      this.sendButton.removeAttribute("disabled");
    }
  }
  /**
   * Disable chat input
   */
  disableChat() {
    if (this.inputField) {
      this.inputField.disabled = true;
      this.inputField.placeholder = WIDGET_I18N[this.config.language].privacyDisabledPlaceholder;
    }
    if (this.sendButton) {
      this.sendButton.setAttribute("disabled", "");
    }
  }
  /**
   * Show empty state
   */
  showEmptyState() {
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
  attachEventListeners() {
    if (!this.toggleButton || !this.window || !this.inputField || !this.sendButton) return;
    addEvent(this.toggleButton, "click", () => this.toggle());
    const closeBtn = this.window.querySelector(".agentix-chat-close-btn");
    if (closeBtn) {
      addEvent(closeBtn, "click", () => this.close());
    }
    addEvent(this.sendButton, "click", () => this.sendMessage());
    addEvent(this.inputField, "keydown", (e) => {
      const keyEvent = e;
      if (keyEvent.key === "Enter" && !keyEvent.shiftKey) {
        keyEvent.preventDefault();
        this.sendMessage();
      }
    });
    addEvent(this.inputField, "input", () => this.autoResizeTextarea());
    this.setupResizeHandle();
    this.windowResizeHandler = () => {
      if (!this.window) return;
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const currentWidth = this.window.offsetWidth;
      const currentHeight = this.window.offsetHeight;
      if (currentWidth > vw) {
        this.window.style.width = vw + "px";
      }
      if (currentHeight > vh) {
        this.window.style.height = vh + "px";
      }
    };
    window.addEventListener("resize", this.windowResizeHandler);
  }
  /**
   * Setup resize handle for the chat window
   */
  setupResizeHandle() {
    if (!this.window) return;
    const resizeHandle = this.window.querySelector(".agentix-chat-resize-handle");
    if (!resizeHandle) return;
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
      let newWidth = startWidth - deltaX;
      let newHeight = startHeight - deltaY;
      newWidth = Math.min(Math.max(newWidth, minWidth), maxWidth);
      newHeight = Math.min(Math.max(newHeight, minHeight), maxHeight);
      this.window.style.width = newWidth + "px";
      this.window.style.height = newHeight + "px";
      frameRequested = false;
    };
    const onPointerMove = (e) => {
      if (!isPointerDown) return;
      lastX = e.clientX;
      lastY = e.clientY;
      if (!frameRequested) {
        frameRequested = true;
        requestAnimationFrame(applySize);
      }
    };
    const onPointerUp = (e) => {
      var _a, _b, _c;
      isPointerDown = false;
      (_a = this.window) == null ? void 0 : _a.classList.remove("resizing");
      document.removeEventListener("pointermove", onPointerMove);
      document.removeEventListener("pointerup", onPointerUp);
      (_c = (_b = e.target).releasePointerCapture) == null ? void 0 : _c.call(_b, e.pointerId);
    };
    const onPointerDown = (e) => {
      var _a, _b, _c, _d, _e;
      if (e.button && e.button !== 0) return;
      isPointerDown = true;
      startX = e.clientX;
      startY = e.clientY;
      lastX = startX;
      lastY = startY;
      startWidth = ((_a = this.window) == null ? void 0 : _a.offsetWidth) || this.config.width;
      startHeight = ((_b = this.window) == null ? void 0 : _b.offsetHeight) || this.config.height;
      maxWidth = window.innerWidth;
      maxHeight = window.innerHeight;
      (_c = this.window) == null ? void 0 : _c.classList.add("resizing");
      try {
        (_e = (_d = e.target).setPointerCapture) == null ? void 0 : _e.call(_d, e.pointerId);
      } catch (err) {
      }
      document.addEventListener("pointermove", onPointerMove);
      document.addEventListener("pointerup", onPointerUp);
      e.preventDefault();
    };
    resizeHandle.addEventListener("pointerdown", onPointerDown, { passive: false });
  }
  /**
   * Show a temporary toast message inside the chat window
   */
  _showToast(message, duration = 2500) {
    var _a;
    const parent = (_a = this.inputField) == null ? void 0 : _a.parentElement;
    if (!parent) return;
    parent.style.position = "relative";
    const toast = document.createElement("div");
    toast.className = "agentix-chat-toast";
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
      toast.style.transform = "translateX(0)";
      toast.style.opacity = "1";
    });
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }
  /**
   * Auto-resize textarea based on content
   */
  autoResizeTextarea() {
    if (!this.inputField) return;
    this.inputField.style.height = "auto";
    const scrollHeight = this.inputField.scrollHeight;
    const newHeight = Math.min(scrollHeight, 100);
    this.inputField.style.height = newHeight + "px";
    if (scrollHeight <= 100) {
      this.inputField.style.overflowY = "hidden";
    } else {
      this.inputField.style.overflowY = "auto";
    }
  }
  /**
   * Toggle chat window
   */
  toggle() {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }
  /**
   * Open chat window
   */
  open() {
    var _a;
    if (!this.window || !this.toggleButton) return;
    this.isOpen = true;
    this.window.classList.add("open");
    this.toggleButton.setAttribute("aria-expanded", "true");
    this.unreadCount = 0;
    this.updateBadge();
    localStorage.setItem("agentix-chat-open", "true");
    this.stopBounce();
    if (!this.privacyAccepted) {
      if (this.privacyModal) {
        this.privacyModal.style.display = "flex";
      }
      return;
    }
    (_a = this.inputField) == null ? void 0 : _a.focus();
    this.config.onOpen();
    this.sendInitialMessageIfNeeded();
  }
  /**
   * Close chat window
   */
  close() {
    if (!this.window || !this.toggleButton) return;
    this.isOpen = false;
    this.window.classList.remove("open");
    this.toggleButton.setAttribute("aria-expanded", "false");
    localStorage.setItem("agentix-chat-open", "false");
    this.startBounce();
  }
  /**
   * Start bounce animation and speech bubble on toggle button
   */
  startBounce() {
    if (!this.toggleButton || this.bounceTimer !== null) return;
    const interval = this.config.bounceInterval || 1e4;
    if (interval <= 0) return;
    this.bounceTimer = window.setInterval(() => {
      if (!this.toggleButton || this.isOpen) return;
      this.toggleButton.classList.add("bouncing");
      if (!this.bubbleElement) {
        this.bubbleElement = createElement("div", { class: "agentix-chat-bubble" });
        this.toggleButton.appendChild(this.bubbleElement);
      }
      this.bubbleElement.textContent = this.config.bounceText || WIDGET_I18N[this.config.language].bounceText;
      this.bubbleElement.classList.add("visible");
      window.setTimeout(() => {
        var _a, _b;
        (_a = this.toggleButton) == null ? void 0 : _a.classList.remove("bouncing");
        (_b = this.bubbleElement) == null ? void 0 : _b.classList.remove("visible");
      }, 3e3);
    }, interval);
  }
  /**
   * Stop bounce animation and speech bubble
   */
  stopBounce() {
    if (this.bounceTimer !== null) {
      clearInterval(this.bounceTimer);
      this.bounceTimer = null;
    }
    if (this.toggleButton) {
      this.toggleButton.classList.remove("bouncing");
    }
    if (this.bubbleElement) {
      this.bubbleElement.classList.remove("visible");
    }
  }
  /**
   * Public method to send a message programmatically
   */
  async sendUserMessage(content, options) {
    var _a;
    if (!this.messagesContainer || !this.apiClient || this.isLoading) return;
    const sanitizedContent = sanitizeInput(content);
    if (!sanitizedContent.trim()) return;
    const originalValue = ((_a = this.inputField) == null ? void 0 : _a.value) || "";
    if (this.inputField) {
      this.inputField.value = sanitizedContent;
    }
    await this.sendMessage(options == null ? void 0 : options.silent);
    if (this.inputField && originalValue) {
      this.inputField.value = originalValue;
    }
  }
  /**
   * Extract a user-friendly German error message from an API error
   */
  getUserFriendlyErrorMessage(error) {
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
  async fetchTTSSegment(text, voiceId) {
    const baseUrl = this.config.apiUrl.replace(/\/v1$/, "");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1e4);
    try {
      const res = await fetch(`${baseUrl}/audio/tts/segment`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice_id: voiceId }),
        signal: controller.signal
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
  shouldStreamTTS() {
    var _a, _b, _c, _d;
    if (!((_b = (_a = this.audioCapabilities) == null ? void 0 : _a.tts) == null ? void 0 : _b.button_visible) || !((_d = (_c = this.audioCapabilities) == null ? void 0 : _c.tts) == null ? void 0 : _d.voice_id)) return false;
    if (this.phoneMode) return true;
    if (this.audioCapabilities.tts.auto_read && this.audioCapabilities.tts.streaming_enabled) return true;
    return false;
  }
  async sendMessage(silent = false, text, audioEmotions) {
    var _a, _b, _c, _d;
    if (!this.messagesContainer || !this.apiClient) return;
    if (!text && (!this.inputField || this.isLoading)) return;
    const userInput = text ? sanitizeInput(text) : sanitizeInput(this.inputField.value);
    if (!userInput.trim() && !(audioEmotions == null ? void 0 : audioEmotions.audio_prosody)) return;
    this.stopAllTTS();
    const userMessage = {
      id: `msg-${Date.now()}`,
      role: "user",
      content: userInput,
      timestamp: Date.now()
    };
    if (!silent) {
      this.messages.push(userMessage);
      this.renderMessage(userMessage);
    }
    if (this.inputField) {
      this.inputField.value = "";
      this.autoResizeTextarea();
    }
    this.isLoading = true;
    if (this.inputField) {
      this.inputField.disabled = true;
    }
    (_a = this.sendButton) == null ? void 0 : _a.setAttribute("disabled", "");
    if (this.sendButton) this.sendButton.style.display = "none";
    if (this.stopButton) this.stopButton.style.display = this.phoneMode ? "none" : "";
    const emptyState = this.messagesContainer.querySelector(".agentix-chat-empty-state");
    if (emptyState) {
      emptyState.remove();
    }
    const assistantMessage = {
      id: `msg-${Date.now()}-streaming`,
      role: "assistant",
      content: "",
      timestamp: Date.now(),
      isStreaming: true
    };
    this.messages.push(assistantMessage);
    const messageElement = this.renderMessage(assistantMessage);
    try {
      this.abortController = new AbortController();
      let apiMessages = this.messages.filter((m) => m.role !== "system").filter((m) => !(m.role === "assistant" && m.isStreaming && m.content === "")).map((m) => ({
        role: m.role,
        content: m.content
      }));
      if (silent) {
        apiMessages.push({ role: "user", content: userInput });
      }
      this.stopEmotionPolling();
      const phases = this.buildEmotionPhases();
      const fused = this.computeOverallFused(phases);
      let fullContent = "";
      let streamingContentElement = null;
      const streamTTS = this.shouldStreamTTS();
      if (streamTTS) {
        this.ttsSegmentPlayer = new TTSSegmentPlayer();
      }
      const voiceId = ((_c = (_b = this.audioCapabilities) == null ? void 0 : _b.tts) == null ? void 0 : _c.voice_id) || null;
      const preSendEmotion = await this.capturePreSendEmotion();
      const emotionPayload = this.buildEmotionPayload();
      if (emotionPayload && preSendEmotion) {
        emotionPayload.visual_pre_send = preSendEmotion;
      }
      if (emotionPayload && (audioEmotions == null ? void 0 : audioEmotions.audio_prosody)) {
        emotionPayload.audio_prosody = audioEmotions.audio_prosody;
        if (emotionPayload.consent) {
          emotionPayload.consent.audio_opt_in = true;
        }
      }
      if (emotionPayload) {
        if (fused) emotionPayload.fused = fused;
        if (phases.length > 0) emotionPayload.phases = phases;
        if (this.audioSnapshotsRef.length > 0) {
          const bestAudio = this.audioSnapshotsRef.reduce(
            (best, curr) => curr.confidence > best.confidence ? curr : best
          );
          emotionPayload.audio_prosody = bestAudio;
          this.audioSnapshotsRef = [];
        }
      }
      let backendEmotion = null;
      for await (const chunk of this.apiClient.streamChat(apiMessages, emotionPayload)) {
        if (chunk.emotion) {
          backendEmotion = chunk.emotion;
        }
        if (chunk.content) {
          fullContent += chunk.content;
          assistantMessage.content = fullContent;
          if (!this.hasTransitionedToTypingRef) {
            this.switchEmotionPhase("typing");
            this.hasTransitionedToTypingRef = true;
          }
          if (streamTTS && voiceId) {
            this.ttsSentenceBuffer += chunk.content;
            const { chunks, remainder } = extractCompleteSentences(this.ttsSentenceBuffer);
            this.ttsSentenceBuffer = remainder;
            for (const chunk2 of chunks) {
              const clean = this.cleanTextForTTS(chunk2.text);
              if (clean) {
                this.ttsSegmentPlayer.add(clean, voiceId, this.fetchTTSSegment.bind(this), chunk2.pauseAfterMs);
              }
            }
          }
        }
        if (chunk.toolExecution) {
          assistantMessage.terminalEvents = [...assistantMessage.terminalEvents || [], chunk.toolExecution];
        }
        const contentElement = messageElement == null ? void 0 : messageElement.querySelector(".agentix-chat-message-content");
        if (contentElement) {
          if (!streamingContentElement) {
            streamingContentElement = createElement("div", { class: "agentix-chat-streaming-content" });
            contentElement.appendChild(streamingContentElement);
          }
          streamingContentElement.innerHTML = renderMarkdown(fullContent);
        }
        this.smartScrollToBottom();
      }
      if (streamTTS && voiceId && this.ttsSentenceBuffer.trim()) {
        const clean = this.cleanTextForTTS(this.ttsSentenceBuffer.trim());
        if (clean) {
          this.ttsSegmentPlayer.add(clean, voiceId, this.fetchTTSSegment.bind(this), 0);
        }
        this.ttsSentenceBuffer = "";
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
      if (messageElement) {
        const thinkingWrapper = messageElement.querySelector(".agentix-chat-thinking");
        if (thinkingWrapper) {
          thinkingWrapper.classList.add("fade-out");
          setTimeout(() => {
            if (thinkingWrapper.parentNode) {
              thinkingWrapper.parentNode.removeChild(thinkingWrapper);
            }
          }, 300);
        }
      }
      if (messageElement) {
        const contentEl = messageElement.querySelector(".agentix-chat-message-content");
        if (contentEl) {
          this.postProcessHtmlBlocks(contentEl, assistantMessage.content);
        }
      }
      if (this.phoneMode && fullContent.trim() && messageElement) {
        if (this.ttsSegmentPlayer) {
          const waitForPlayer = () => {
            var _a2;
            console.info("[Widget] waitForPlayer poll, queueLength=", (_a2 = this.ttsSegmentPlayer) == null ? void 0 : _a2.queueLength);
            if (this.ttsSegmentPlayer && this.ttsSegmentPlayer.queueLength > 0) {
              setTimeout(waitForPlayer, 200);
            } else {
              console.info("[Widget] waitForPlayer done, restarting recording");
              if (this.phoneMode) {
                this.awaitingUserSpeech = true;
                this.updateMicButton();
                setTimeout(() => this.startRecording(), 300);
              }
            }
          };
          setTimeout(waitForPlayer, 200);
        } else {
          setTimeout(() => this.speakText(fullContent.trim(), messageElement), 200);
        }
      }
    } catch (error) {
      this.stopAllTTS();
      const assistantIndex = this.messages.findIndex((m) => m.id === assistantMessage.id);
      if (assistantIndex !== -1) {
        this.messages.splice(assistantIndex, 1);
      }
      if (messageElement && messageElement.parentNode) {
        messageElement.parentNode.removeChild(messageElement);
      }
      const friendlyMessage = this.getUserFriendlyErrorMessage(
        error instanceof Error ? error : new Error(String(error))
      );
      const errorMsgElement = createElement("div", {
        class: "agentix-chat-error-message"
      });
      const errorIcon = createElement("span", { class: "agentix-chat-error-icon" });
      errorIcon.textContent = "⚠️";
      const errorText = createElement("span");
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
      (_d = this.sendButton) == null ? void 0 : _d.removeAttribute("disabled");
      if (this.sendButton) this.sendButton.style.display = this.phoneMode ? "none" : "";
      if (this.stopButton) this.stopButton.style.display = "none";
      this.smartScrollToBottom();
      if (!this.phoneMode && this.inputField) {
        this.inputField.focus();
      }
      if (this.ttsSegmentPlayer) {
        const player = this.ttsSegmentPlayer;
        setTimeout(() => {
          if (player.queueLength === 0) {
            player.abort();
          }
        }, 5e3);
      }
    }
  }
  /**
   * Add a speak button to a completed assistant message element if not already present.
   */
  addSpeakButton(messageElement, message) {
    var _a, _b;
    if (message.role !== "assistant" || message.isStreaming || !message.content) return;
    if (messageElement.querySelector(".agentix-chat-message-actions")) return;
    const actionsWrapper = createElement("div", {
      class: "agentix-chat-message-actions",
      style: ((_b = (_a = this.audioCapabilities) == null ? void 0 : _a.tts) == null ? void 0 : _b.button_visible) ? "" : "display:none"
    });
    const speakBtn = createElement("button", {
      class: "agentix-chat-speak-btn",
      "aria-label": "Vorlesen",
      title: "Vorlesen"
    });
    const ring1 = createElement("span", { class: "speak-ring" });
    const ring2 = createElement("span", { class: "speak-ring" });
    const ring3 = createElement("span", { class: "speak-ring" });
    speakBtn.appendChild(ring1);
    speakBtn.appendChild(ring2);
    speakBtn.appendChild(ring3);
    const speakSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    speakSvg.setAttribute("class", "speak-icon");
    speakSvg.setAttribute("width", "16");
    speakSvg.setAttribute("height", "16");
    speakSvg.setAttribute("viewBox", "0 0 24 24");
    speakSvg.setAttribute("fill", "none");
    speakSvg.setAttribute("stroke", "currentColor");
    speakSvg.setAttribute("stroke-width", "2");
    speakSvg.setAttribute("stroke-linecap", "round");
    speakSvg.setAttribute("stroke-linejoin", "round");
    speakSvg.innerHTML = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>';
    speakBtn.appendChild(speakSvg);
    const stopSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    stopSvg.setAttribute("class", "speak-stop-icon");
    stopSvg.setAttribute("width", "16");
    stopSvg.setAttribute("height", "16");
    stopSvg.setAttribute("viewBox", "0 0 24 24");
    stopSvg.setAttribute("fill", "currentColor");
    stopSvg.setAttribute("stroke", "none");
    stopSvg.innerHTML = '<rect x="6" y="6" width="12" height="12" rx="2"/>';
    stopSvg.style.display = "none";
    speakBtn.appendChild(stopSvg);
    speakBtn.addEventListener("click", () => {
      this.speakText(message.content, messageElement, speakBtn);
    });
    actionsWrapper.appendChild(speakBtn);
    messageElement.appendChild(actionsWrapper);
  }
  /**
   * Render message in DOM
   */
  renderMessage(message) {
    if (!this.messagesContainer) return null;
    const messageElement = createElement("div", {
      class: `agentix-chat-message ${message.role} ${message.isStreaming ? "streaming" : ""}`,
      "data-id": message.id
    });
    const contentElement = createElement("div", {
      class: "agentix-chat-message-content"
    });
    if (message.role === "user") {
      contentElement.textContent = message.content;
    } else if (message.content) {
      contentElement.innerHTML = renderMarkdown(message.content);
      if (!message.isStreaming) {
        this.postProcessHtmlBlocks(contentElement, message.content || "");
      }
    } else {
      const thinkingWrapper = createElement("div", { class: "agentix-chat-thinking" });
      const thinkingLabel = createElement("span", { class: "agentix-chat-thinking-label" });
      thinkingLabel.textContent = "Thinking";
      const thinkingDots = createElement("div", { class: "agentix-chat-thinking-dots" });
      thinkingDots.appendChild(createElement("span"));
      thinkingDots.appendChild(createElement("span"));
      thinkingDots.appendChild(createElement("span"));
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
  showHtmlPreviewModal(html2) {
    if (!this.window) return;
    const overlay = createElement("div", { class: "agentix-chat-modal-overlay" });
    overlay.addEventListener("click", () => {
      overlay.remove();
      this.previewHtml = null;
    });
    const box = createElement("div", { class: "agentix-chat-modal-box preview-modal" });
    box.addEventListener("click", (e) => e.stopPropagation());
    const header = createElement("div", { class: "agentix-chat-modal-header" });
    const title = createElement("h4", {}, "HTML Preview");
    const closeBtn = createElement("button", { class: "agentix-chat-modal-close" }, "✕");
    closeBtn.addEventListener("click", () => {
      overlay.remove();
      this.previewHtml = null;
    });
    header.appendChild(title);
    header.appendChild(closeBtn);
    box.appendChild(header);
    const content = createElement("div", { class: "agentix-chat-preview-content" });
    const iframe = createElement("iframe", {
      class: "agentix-chat-preview-frame",
      sandbox: "allow-scripts",
      srcdoc: html2,
      title: "HTML preview"
    });
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
  postProcessHtmlBlocks(contentElement, content) {
    const blocks = extractCodeBlocks(content);
    const htmlBlocks = blocks.filter((b) => b.language === "html");
    if (htmlBlocks.length === 0) return;
    const allPreCodes = contentElement.querySelectorAll("pre > code");
    let firstHtmlPre = null;
    const toRemove = [];
    allPreCodes.forEach((code) => {
      const className = code.getAttribute("class") || "";
      const lang = className.replace("language-", "").trim().toLowerCase();
      const pre = code.parentElement;
      if (!pre) return;
      if (lang === "html") {
        if (!firstHtmlPre) {
          firstHtmlPre = pre;
        } else {
          toRemove.push(pre);
        }
      } else if (lang === "css" || lang === "javascript" || lang === "js") {
        toRemove.push(pre);
      }
    });
    if (!firstHtmlPre) return;
    const i18n = WIDGET_I18N[this.config.language];
    const htmlDoc = buildHtmlDoc(blocks);
    const wrapper = createElement("div", { class: "agentix-chat-html-renderbox" });
    const toggle = createElement("div", { class: "agentix-chat-code-toggle" });
    const codeBtn = createElement("button", { class: "active" }, i18n.codeView);
    const webBtn = createElement("button", {}, i18n.webView);
    toggle.appendChild(codeBtn);
    toggle.appendChild(webBtn);
    wrapper.appendChild(toggle);
    const iframeWrapper = createElement("div", { class: "agentix-chat-html-iframe-wrapper" });
    const iframe = createElement("iframe", {
      class: "agentix-chat-html-iframe",
      sandbox: "allow-scripts",
      srcdoc: htmlDoc,
      title: "HTML preview"
    });
    iframe.style.display = "none";
    iframeWrapper.appendChild(iframe);
    const maxBtn = createElement("button", { class: "agentix-chat-html-maximize", title: "Fullscreen" });
    maxBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>';
    maxBtn.style.display = "none";
    maxBtn.addEventListener("click", () => {
      this.showHtmlPreviewModal(htmlDoc);
    });
    iframeWrapper.appendChild(maxBtn);
    wrapper.appendChild(iframeWrapper);
    if (firstHtmlPre) {
      const preEl = firstHtmlPre;
      const parent = preEl.parentElement;
      if (parent) {
        parent.insertBefore(wrapper, preEl);
        wrapper.appendChild(preEl);
      }
    }
    codeBtn.addEventListener("click", () => {
      codeBtn.classList.add("active");
      webBtn.classList.remove("active");
      firstHtmlPre.style.display = "";
      iframe.style.display = "none";
      maxBtn.style.display = "none";
    });
    webBtn.addEventListener("click", () => {
      webBtn.classList.add("active");
      codeBtn.classList.remove("active");
      firstHtmlPre.style.display = "none";
      iframe.style.display = "";
      maxBtn.style.display = "";
    });
    toRemove.forEach((el) => el.remove());
  }
  /**
   * Update badge with unread count
   */
  updateBadge() {
    if (!this.toggleButton) return;
    const badge = this.toggleButton.querySelector(".agentix-chat-badge");
    if (this.unreadCount > 0 && !this.isOpen) {
      if (badge) {
        badge.textContent = String(this.unreadCount);
      } else {
        const newBadge = createElement("div", {
          class: "agentix-chat-badge"
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
  saveMessages() {
    try {
      const messagesToSave = this.messages.filter((m) => m.role !== "system");
      localStorage.setItem("agentix-chat-messages", JSON.stringify(messagesToSave));
    } catch (error) {
      console.warn("Could not save messages to localStorage:", error);
    }
  }
  /**
   * Load messages from localStorage
   */
  loadMessages() {
    try {
      const saved = localStorage.getItem("agentix-chat-messages");
      if (saved) {
        this.messages = JSON.parse(saved);
        if (this.messagesContainer && this.messages.length > 0) {
          this.messagesContainer.innerHTML = "";
          this.messages.forEach((msg) => this.renderMessage(msg));
        }
      }
    } catch (error) {
      console.warn("Could not load messages from localStorage:", error);
    }
  }
  /**
   * Get message count
   */
  getMessageCount() {
    return this.messages.filter((m) => m.role !== "system").length;
  }
  /**
   * Clear messages
   */
  clearMessages() {
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
  destroy() {
    var _a;
    this.stopWebcamEmotion();
    this.stopEmotionPolling();
    if (this.windowResizeHandler) {
      window.removeEventListener("resize", this.windowResizeHandler);
      this.windowResizeHandler = null;
    }
    if (this.container) {
      this.container.remove();
    }
    if (this.styleElement) {
      this.styleElement.remove();
    }
    (_a = this.abortController) == null ? void 0 : _a.abort();
  }
  /**
   * Reset conversation id so the next request uses a fresh session.
   */
  resetConversationId() {
    sessionStorage.removeItem("agentix-conversation-id");
    this.setupAPIClient();
  }
  /**
   * Update configuration
   */
  updateConfig(newConfig) {
    var _a, _b, _c, _d, _e, _f, _g;
    if (newConfig.language === "auto") {
      newConfig = { ...newConfig, language: detectLanguage() };
    }
    this.config = { ...this.config, ...newConfig };
    if (!this.config.logoUrlButton) {
      this.config.logoUrlButton = this.config.logoUrl || WIDGET_FAVICON_BASE64;
    }
    if (!this.config.logoUrlHeader) {
      this.config.logoUrlHeader = this.config.logoUrl || WIDGET_LOGO_BASE64;
    }
    this.validateConfig();
    this.setupAPIClient();
    if (newConfig.title !== void 0) {
      const titleEl = (_a = this.container) == null ? void 0 : _a.querySelector(".agentix-chat-header-title");
      if (titleEl) {
        titleEl.textContent = this.config.title;
      }
    }
    if (newConfig.subtitle !== void 0) {
      const subtitleEl = (_b = this.container) == null ? void 0 : _b.querySelector(".agentix-chat-header-subtitle");
      if (subtitleEl) {
        subtitleEl.textContent = this.config.subtitle;
      }
    }
    const localizedKeys = [
      "privacyText",
      "privacyTextDe",
      "privacyTextEn",
      "initialMessageContent",
      "initialMessageContentDe",
      "initialMessageContentEn",
      "bounceText",
      "bounceTextDe",
      "bounceTextEn",
      "inputPlaceholder",
      "inputPlaceholderDe",
      "inputPlaceholderEn"
    ];
    if (newConfig.language !== void 0 || localizedKeys.some((k) => newConfig[k] !== void 0)) {
      const lang = this.config.language;
      const i18n = WIDGET_I18N[lang];
      const privacyOverride = lang === "de" ? this.config.privacyTextDe : this.config.privacyTextEn;
      const initialOverride = lang === "de" ? this.config.initialMessageContentDe : this.config.initialMessageContentEn;
      const bounceOverride = lang === "de" ? this.config.bounceTextDe : this.config.bounceTextEn;
      const placeholderOverride = lang === "de" ? this.config.inputPlaceholderDe : this.config.inputPlaceholderEn;
      this.config.privacyText = privacyOverride || this.config.privacyText || i18n.privacyText;
      this.config.initialMessageContent = initialOverride || this.config.initialMessageContent || i18n.initialMessageContent;
      this.config.bounceText = bounceOverride || this.config.bounceText || i18n.bounceText;
      this.config.inputPlaceholder = placeholderOverride || i18n.inputPlaceholder;
      if (this.inputField) {
        this.inputField.placeholder = this.privacyAccepted ? this.config.inputPlaceholder : i18n.privacyDisabledPlaceholder;
      }
      const privacyTitleEl = (_c = this.container) == null ? void 0 : _c.querySelector(".agentix-chat-privacy-title");
      if (privacyTitleEl) {
        privacyTitleEl.textContent = i18n.privacyTitle;
      }
      const privacyTextEl = (_d = this.container) == null ? void 0 : _d.querySelector(".agentix-chat-privacy-text");
      if (privacyTextEl) {
        privacyTextEl.textContent = this.config.privacyText;
      }
      const privacyAcceptEl = (_e = this.container) == null ? void 0 : _e.querySelector(".agentix-chat-privacy-accept");
      if (privacyAcceptEl) {
        privacyAcceptEl.textContent = i18n.privacyAccept;
      }
      const privacyDeclineEl = (_f = this.container) == null ? void 0 : _f.querySelector(".agentix-chat-privacy-decline");
      if (privacyDeclineEl) {
        privacyDeclineEl.textContent = i18n.privacyDecline;
      }
      const emptyStateEl = (_g = this.container) == null ? void 0 : _g.querySelector(".agentix-chat-empty-state-text");
      if (emptyStateEl) {
        emptyStateEl.innerHTML = i18n.emptyStateText;
      }
      if (this.bubbleElement) {
        this.bubbleElement.textContent = this.config.bounceText;
      }
    }
    if (newConfig.model !== void 0) {
      this.fetchAudioCapabilities();
    }
    if (newConfig.colors) {
      this.applyColorVariables();
    }
    if (newConfig.showLogoInButton !== void 0 || newConfig.logoBorder !== void 0 || newConfig.logoGlow !== void 0 || newConfig.logoBackground !== void 0) {
      this.refreshToggleButtonLogo();
    }
    if (newConfig.logoUrlButton !== void 0 || newConfig.logoUrlHeader !== void 0 || newConfig.logoUrl !== void 0 || newConfig.showLogoInHeader !== void 0 || newConfig.logoBorder !== void 0 || newConfig.logoGlow !== void 0 || newConfig.logoBackground !== void 0) {
      this.refreshHeaderLogo();
      if (newConfig.logoBorder !== void 0 || newConfig.logoGlow !== void 0 || newConfig.logoBackground !== void 0) {
        this.refreshToggleButtonLogo();
      }
    }
    if (newConfig.bounceInterval !== void 0 || newConfig.bounceText !== void 0) {
      if (!this.isOpen) {
        this.stopBounce();
        this.startBounce();
      }
    }
    if (newConfig.position !== void 0 || newConfig.theme !== void 0) {
      if (this.container) {
        this.container.className = `agentix-chat-widget-container ${this.config.position} ${this.config.theme}`;
      }
      if (this.window && newConfig.position !== void 0) {
        this.window.className = `agentix-chat-window ${this.config.position}`;
      }
    }
    if (newConfig.width !== void 0 || newConfig.height !== void 0) {
      if (this.window) {
        this.window.style.width = `${this.config.width}px`;
        this.window.style.height = `${this.config.height}px`;
      }
    }
    if (newConfig.privacyEnabled === false && !this.privacyAccepted) {
      this.privacyAccepted = true;
      this.enableChat();
      if (this.privacyModal) {
        this.privacyModal.style.display = "none";
      }
    }
  }
}
function deepMerge(target, source) {
  const result = { ...target };
  for (const key in source) {
    if (source[key] !== void 0) {
      if (typeof source[key] === "object" && source[key] !== null && !Array.isArray(source[key]) && typeof result[key] === "object" && result[key] !== null) {
        result[key] = deepMerge(result[key], source[key]);
      } else {
        result[key] = source[key];
      }
    }
  }
  return result;
}
function getWidgetBaseUrl() {
  const scripts = document.querySelectorAll("script");
  for (const script of scripts) {
    if (script.src.includes("widget.umd.js") || script.src.includes("widget.esm.js")) {
      const url = new URL(script.src);
      const prefix = url.pathname.replace(/\/widget\/widget\.(umd|esm)\.js$/, "");
      return url.origin + prefix;
    }
  }
  return window.location.origin;
}
async function fetchWidgetDefaults() {
  try {
    const baseUrl = getWidgetBaseUrl();
    const res = await fetch(`${baseUrl}/widget-defaults`, {
      method: "GET",
      credentials: "omit"
    });
    if (!res.ok) return {};
    const data = await res.json();
    if (typeof data !== "object" || data === null || Array.isArray(data)) return {};
    return sanitizeServerDefaults(data);
  } catch {
    return {};
  }
}
window.AgentixChatWidget = ChatWidget;
window.initAgentixChat = async (config) => {
  const defaults = await fetchWidgetDefaults();
  const merged = deepMerge(defaults, config || {});
  return new ChatWidget(merged);
};
document.addEventListener("DOMContentLoaded", async () => {
  const script = document.currentScript;
  if ((script == null ? void 0 : script.dataset.autoInit) === "true") {
    const configStr = script == null ? void 0 : script.dataset.config;
    const config = configStr ? JSON.parse(configStr) : void 0;
    const defaults = await fetchWidgetDefaults();
    const merged = deepMerge(defaults, config || {});
    new ChatWidget(merged);
  }
});
export {
  ChatWidget
};
//# sourceMappingURL=widget.esm.js.map
