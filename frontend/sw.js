/* VirtuCoach Service Worker - PWA App Shell
 * 策略：
 *  - 静态资源（CSS/JS/图标/manifest）：缓存优先，后台更新
 *  - 页面导航：网络优先，离线回退到缓存的首页
 *  - API / 上传 / 截图 / 视频：一律不缓存（隐私与新鲜度）
 */
const CACHE = "virtucoach-shell-v3";
const SHELL = [
  "/",
  "/index.html",
  "/analysis.html",
  "/dashboard.html",
  "/login.html",
  "/register.html",
  "/admin.html",
  "/style.css",
  "/app.js",
  "/js/auth.js",
  "/js/hand-check.js",
  "/js/reference-manager.js",
  "/manifest.webmanifest",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

function cacheKey(url) {
  const u = new URL(url);
  u.search = "";
  return u.href;
}

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);

  // 只处理同源请求
  if (url.origin !== self.location.origin) return;

  // API 与用户数据：永远走网络
  if (
    url.pathname.startsWith("/api/") ||
    url.pathname.startsWith("/uploads/") ||
    url.pathname.startsWith("/snapshots/") ||
    url.pathname.startsWith("/eval-videos/")
  ) {
    return;
  }

  // 页面导航：网络优先
  if (req.mode === "navigate") {
    event.respondWith(
      fetch(req)
        .then((resp) => {
          const copy = resp.clone();
          caches.open(CACHE).then((cache) => cache.put("/index.html", copy));
          return resp;
        })
        .catch(() =>
          caches.match("/index.html").then((hit) => hit || caches.match("/"))
        )
    );
    return;
  }

  // 静态资源：缓存优先 + 后台刷新
  if (req.method === "GET") {
    event.respondWith(
      caches.match(cacheKey(req.url)).then((hit) => {
        const network = fetch(req)
          .then((resp) => {
            if (resp && resp.ok) {
              const copy = resp.clone();
              caches.open(CACHE).then((cache) => cache.put(cacheKey(req.url), copy));
            }
            return resp;
          })
          .catch(() => hit);
        return hit || network;
      })
    );
  }
});
