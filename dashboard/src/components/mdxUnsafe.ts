/* MDXEditor parses content as MDX, not CommonMark — `{…}` becomes a JSX
 * expression, `<x` a JSX element, and `import`/`export` at line start an
 * ESM node. Expressions and ESM nodes are silently dropped when the editor
 * serializes back to markdown, and malformed variants (`x <5`, `</tag>`,
 * `<!-- -->`, `import the data first` — a plain sentence!) throw parse
 * errors. Content matching any of these must stay out of the rich editor;
 * it edits in the code view and renders losslessly in the preview. */
export function mdxUnsafe(md: string): boolean {
  const lines: string[] = []
  let fenceChar = ''
  let fenceLen = 0
  for (const line of md.split('\n')) {
    const m = line.match(/^ {0,3}(`{3,}|~{3,})/)
    if (m) {
      if (!fenceChar) {
        fenceChar = m[1][0]
        fenceLen = m[1].length
      } else if (m[1][0] === fenceChar && m[1].length >= fenceLen) {
        fenceChar = ''
        fenceLen = 0
      }
      continue
    }
    if (!fenceChar) lines.push(line)
  }
  const text = lines
    .join('\n')
    .replace(/`[^`\n]*`/g, '')
    .replace(/\\./g, '')
  return /\{/.test(text) || /<(?![ \t])/.test(text) || /^[ \t>]*(?:import|export)\s/m.test(text)
}
