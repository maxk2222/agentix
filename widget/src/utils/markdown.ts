import { marked } from 'marked';

interface CodeBlockInfo {
  language: string;
  content: string;
}

/**
 * Extract code blocks from markdown text
 */
export function extractCodeBlocks(markdown: string): CodeBlockInfo[] {
  const blocks: CodeBlockInfo[] = [];
  const regex = /```(\w*)\n([\s\S]*?)```/g;
  let match;
  while ((match = regex.exec(markdown)) !== null) {
    blocks.push({ language: match[1].trim().toLowerCase(), content: match[2] });
  }
  return blocks;
}

/**
 * Build a full HTML document from extracted code blocks
 */
export function buildHtmlDoc(blocks: CodeBlockInfo[]): string {
  const css = blocks
    .filter((b) => b.language === 'css')
    .map((b) => b.content)
    .join('\n');
  const js = blocks
    .filter((b) => b.language === 'javascript' || b.language === 'js')
    .map((b) => b.content)
    .join('\n');
  const html = blocks
    .filter((b) => b.language === 'html')
    .map((b) => b.content)
    .join('\n');

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
</script>
</head>
<body>
${html}
</body>
</html>`;
}

/**
 * Configure marked for safe markdown rendering
 */
function configureMarked() {
  marked.setOptions({
    breaks: false,
    gfm: true
  });
}

/**
 * Safely render markdown to HTML
 * Prevents XSS attacks by sanitizing code blocks
 */
export function renderMarkdown(text: string): string {
  configureMarked();

  const html = marked(text) as string;

  // Sanitize dangerous attributes from code blocks
  return html
    .replace(/on\w+="[^"]*"/g, '') // Remove event handlers
    .replace(/javascript:/g, '') // Remove javascript: protocol
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, ''); // Remove script tags
}

/**
 * Escape HTML special characters
 */
export function escapeHtml(text: string): string {
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, (char) => map[char]);
}

/**
 * Sanitize user input before sending to API
 */
export function sanitizeInput(input: string): string {
  return input
    .trim()
    .replace(/\x00/g, '') // Remove null bytes
    .replace(/[\x80-\xFF]/g, (char) => {
      // Keep valid unicode but remove control characters
      const code = char.charCodeAt(0);
      return code >= 0x0020 ? char : '';
    })
    .substring(0, 4000); // Max input length
}
