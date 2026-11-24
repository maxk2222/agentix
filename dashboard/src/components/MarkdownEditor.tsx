import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Code, Eye, Pencil } from 'lucide-react'
import {
  MDXEditor,
  headingsPlugin,
  listsPlugin,
  quotePlugin,
  thematicBreakPlugin,
  markdownShortcutPlugin,
  linkPlugin,
  tablePlugin,
  codeBlockPlugin,
  toolbarPlugin,
  UndoRedo,
  BoldItalicUnderlineToggles,
  BlockTypeSelect,
  ListsToggle,
  CreateLink,
  InsertCodeBlock,
  InsertThematicBreak,
} from '@mdxeditor/editor'
import '@mdxeditor/editor/style.css'
import { mdxUnsafe } from './mdxUnsafe'

type EditorMode = 'rich' | 'preview' | 'code'

export default function MarkdownEditor({
  value,
  onChange,
  placeholder,
  disabled = false,
}: {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  disabled?: boolean
}) {
  const [mode, setMode] = useState<EditorMode>(() => (mdxUnsafe(value) ? 'preview' : 'rich'))
  const [markdown, setMarkdown] = useState(value)
  const editorRef = useRef<any>(null)
  const lastValueRef = useRef(value)
  const unsafe = mdxUnsafe(markdown)

  useEffect(() => {
    if (value !== lastValueRef.current) {
      lastValueRef.current = value
      setMarkdown(value)
      if (mdxUnsafe(value)) {
        setMode('preview')
      } else {
        editorRef.current?.setMarkdown(value)
      }
    }
  }, [value])

  const edit = (md: string) => {
    lastValueRef.current = md
    setMarkdown(md)
    onChange(md)
  }

  return (
    <div className="inline-markdown-editor" style={{ opacity: disabled ? 0.6 : 1 }}>
      {!disabled && (
        <div className="editor-mode-toggle">
          <button
            type="button"
            className={mode === 'rich' ? 'active' : ''}
            onClick={() => setMode('rich')}
            disabled={unsafe}
            title={
              unsafe
                ? 'Rich text is unavailable: the content contains {…} expressions, <tags> or import/export lines'
                : 'Rich text'
            }
          >
            <Pencil size={14} />
          </button>
          <button
            type="button"
            className={mode === 'preview' ? 'active' : ''}
            onClick={() => setMode('preview')}
            title="Rendered preview"
          >
            <Eye size={14} />
          </button>
          <button
            type="button"
            className={mode === 'code' ? 'active' : ''}
            onClick={() => setMode('code')}
            title="Markdown source"
          >
            <Code size={14} />
          </button>
        </div>
      )}
      {mode === 'rich' && !disabled ? (
        <MDXEditor
          ref={editorRef}
          markdown={markdown}
          onChange={edit}
          plugins={[
            headingsPlugin(),
            listsPlugin(),
            quotePlugin(),
            thematicBreakPlugin(),
            markdownShortcutPlugin(),
            linkPlugin(),
            tablePlugin(),
            codeBlockPlugin(),
            toolbarPlugin({
              toolbarContents: () => (
                <>
                  <UndoRedo />
                  <BlockTypeSelect />
                  <BoldItalicUnderlineToggles />
                  <ListsToggle />
                  <CreateLink />
                  <InsertCodeBlock />
                  <InsertThematicBreak />
                </>
              ),
            }),
          ]}
        />
      ) : mode === 'code' && !disabled ? (
        <textarea
          className="editor-textarea"
          value={markdown}
          onChange={(e) => edit(e.target.value)}
          placeholder={placeholder}
          spellCheck={false}
        />
      ) : (
        <div className="markdown-body md-preview">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
        </div>
      )}
    </div>
  )
}
