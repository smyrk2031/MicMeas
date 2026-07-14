/* ネットワーク優先＋キャッシュフォールバック（オフラインでも起動できるように） */
const CACHE = 'otoscope-v5';
const FILES = ['./', 'index.html', 'style.css', 'app.js', 'manifest.webmanifest', 'icon.svg',
  'vendor/tf.min.js'];   // モデル本体(models/)は初回読込時にランタイムキャッシュされる

self.addEventListener('install', e =>
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(FILES))));

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET' || !e.request.url.startsWith(self.location.origin)) return;
  e.respondWith(
    fetch(e.request)
      .then(r => { const cl = r.clone(); caches.open(CACHE).then(c => c.put(e.request, cl)); return r; })
      .catch(() => caches.match(e.request))
  );
});
