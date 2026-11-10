/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_OPENAI_API_KEY: string;
  readonly VITE_OPENAI_API_URL: string;
  readonly VITE_OPENAI_MODEL: string;
  readonly VITE_WIDGET_POSITION: string;
  readonly VITE_WIDGET_THEME: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}