import { defineConfig } from 'vite'
import basicSsl from '@vitejs/plugin-basic-ssl'

export default defineConfig({
  plugins: [basicSsl()],
  build: {
    lib: {
      entry: 'src/widget.ts',
      name: 'AgentixChatWidget',
      formats: ['umd', 'es'],
      fileName: (format) => `widget.${format === 'es' ? 'esm' : 'umd'}.js`
    },
    minify: 'terser',
    sourcemap: true,
    rollupOptions: {
      external: ['onnxruntime-web'],
      output: {
        globals: {
          'onnxruntime-web': 'ort'
        }
      }
    }
  },
  server: {
    port: 5173,
    https: true
  }
})
