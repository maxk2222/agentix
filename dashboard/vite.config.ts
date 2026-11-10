import { defineConfig, Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import * as fs from 'fs'

/**
 * After the multi-page build, Vite emits the public-chat entry as
 * `dist/public-chat/index.html` but keeps the shared chunks in
 * `dist/assets`.  Because the entry sits one directory deeper, Vite
 * writes `../assets/…` for the script/link tags.  When FastAPI
 * mounts `/chat` -> `dist/public-chat/`, those `../assets/…`
 * resolve to `/assets/…` at the root, which is *outside* the mount
 * and therefore 404.
 *
 * This plugin patches the emitted HTML so the browser resolves the
 * asset URLs relative to `/chat/` (i.e. `/chat/assets/…`), which we
 * then serve via an additional `StaticFiles` mount in `main.py`.
 */
function patchPublicChatHtml(): Plugin {
  return {
    name: 'patch-public-chat-html',
    closeBundle() {
      const htmlPath = resolve(__dirname, 'dist', 'public-chat', 'index.html')
      if (!fs.existsSync(htmlPath)) return
      let html = fs.readFileSync(htmlPath, 'utf-8')
      // Replace ../assets/ with ./assets/ so the browser resolves them
      // relative to /chat/ → /chat/assets/...
      html = html.replace(/\.\.\/assets\//g, './assets/')
      // Embed favicon as inline base64 data URI so no external request is needed.
      const faviconPath = resolve(__dirname, 'public', 'favicon.png')
      if (fs.existsSync(faviconPath)) {
        const faviconB64 = Buffer.from(fs.readFileSync(faviconPath)).toString('base64')
        html = html.replace(
          /href="\.\.\/favicon\.png"/g,
          `href="data:image/png;base64,${faviconB64}"`
        )
      }
      fs.writeFileSync(htmlPath, html)
    },
  }
}

export default defineConfig({
  plugins: [react(), patchPublicChatHtml()],
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        chat: resolve(__dirname, 'public-chat/index.html'),
      },
    },
  },
})
