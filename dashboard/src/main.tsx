import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { AppPreferencesProvider } from './ui/AppPreferences'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppPreferencesProvider>
      <App />
    </AppPreferencesProvider>
  </React.StrictMode>
)
