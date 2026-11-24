import { describe, expect, it } from 'vitest'
import { mdxUnsafe } from './mdxUnsafe'

/* Regression coverage for the MDX safety gate. Every `true` case below was
 * verified against the real MDX parser (micromark-extension-mdxjs): the
 * content either throws a parse error or silently loses nodes when the
 * editor serializes back to markdown — so it must never reach rich mode. */
describe('mdxUnsafe', () => {
  it.each([
    ['unfenced JSON object', 'Handle {"a": 1} here'],
    ['moustache placeholder', 'Use {{context}} value'],
    ['inline JSX tag', 'Handle <context> here'],
    ['JSX block', 'Before\n\n<Rules>\nsome rules\n</Rules>\n\nAfter.'],
    ['closing tag only', 'text </context>'],
    ['html comment', 'a <!-- c --> b'],
    ['less-than digit', 'x <5'],
    ['less-than-equals', 'x <= 5'],
    ['less-than at end of line', 'end <'],
    ['less-than at end of file', 'end <\n\nnext'],
    ['import as plain sentence', 'import the data first'],
    ['export as plain sentence', 'export your results to csv'],
    ['esm import', 'import x from "y"'],
    ['esm export', 'export const z = 1'],
    ['import inside blockquote', '> import the file'],
    ['double-backslash brace', 'literal \\\\{ brace'],
    ['unfenced brace after closed fence', '```\ncode\n```\n{"a":1}'],
    ['indented json (conservative)', '   paragraph\n    {"a":1}'],
  ])('marks %s as unsafe', (_name, md) => {
    expect(mdxUnsafe(md)).toBe(true)
  })

  it.each([
    ['fenced json', '```json\n{"a":1}\n```'],
    ['tilde-fenced json', '~~~json\n{"a":1}\n~~~'],
    ['unclosed fence (rest is code)', 'Text:\n```json\n{"a":1}'],
    ['longer closing fence', '```json\n{"a":1}\n````'],
    ['inline code with braces', 'Use `{"a":1}` inline'],
    ['inline code with tag', 'Use `<context>` inline'],
    ['less-than with space', 'a < b'],
    ['mid-sentence import', 'Please import the data'],
    ['escaped brace', 'literal \\{ brace'],
    ['plain markdown prompt', 'You are a helpful assistant.\n\n- be nice\n- use **bold** and `code`'],
    ['empty', ''],
  ])('keeps %s in rich mode', (_name, md) => {
    expect(mdxUnsafe(md)).toBe(false)
  })
})
