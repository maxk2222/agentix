// Service worker for the public chat page. Its only job is to make the
// page installable (browsers require an active worker with a fetch handler)
// and to cache the hashed static bundles. API calls, SSE streams and
// everything outside /chat/assets/ pass straight through to the network.
const CACHE = 'agentix-chat-static-v1'

self.addEventListener('install', () => self.skipWaiting())

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  )
})

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url)
  if (event.request.method !== 'GET' || url.origin !== self.location.origin) return
  if (!url.pathname.startsWith('/chat/assets/')) return
  event.respondWith(
    caches.match(event.request).then(
      (hit) =>
        hit ||
        fetch(event.request).then((res) => {
          if (res.ok) {
            const clone = res.clone()
            caches.open(CACHE).then((cache) => cache.put(event.request, clone))
          }
          return res
        })
    )
  )
})
