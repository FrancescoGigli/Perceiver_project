/* Only the player shell is cached. Access codes, API responses and media never are. */
"use strict";
const CACHE = "perceiver-player-shell-v4";
const SHELL = ["./", "index.html", "player.css", "commands.js", "player.js", "manifest.webmanifest", "icon.svg", "icon-192.png", "icon-512.png", "icon-180.png"].map(path => new URL(path, self.location.href).href);
self.addEventListener("install", event => event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(SHELL)).then(() => self.skipWaiting())));
self.addEventListener("activate", event => event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key.startsWith("perceiver-player-shell-") && key !== CACHE).map(key => caches.delete(key)))).then(() => self.clients.claim())));
self.addEventListener("fetch", event => {
  if (event.request.method !== "GET") return;
  const url = new URL(event.request.url);
  url.search = "";
  if (url.origin !== self.location.origin || !SHELL.includes(url.href)) return;
  event.respondWith((async () => {
    try {
      const response = await fetch(event.request);
      if (!response.ok) return await caches.match(url.href) || response;
      if (response.type === "basic") {
        const copy = response.clone();
        try { const cache = await caches.open(CACHE); await cache.put(url.href, copy); } catch (_) {}
      }
      return response;
    } catch (error) {
      const cached = await caches.match(url.href);
      if (cached) return cached;
      throw error;
    }
  })());
});
