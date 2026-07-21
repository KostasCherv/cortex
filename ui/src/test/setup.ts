import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = vi.fn()
}

// ponytail: Node 26's experimental global `localStorage` shadows jsdom's own
// implementation in this environment, leaving `window.localStorage` undefined.
// Minimal in-memory polyfill so components that read/write localStorage (e.g.
// ThemeProvider) don't crash under test. Upgrade path: drop this once jsdom/vitest
// resolve the global ordering (track jsdom-testing-library or vitest-environment-jsdom).
if (!window.localStorage) {
  const store = new Map<string, string>()
  Object.defineProperty(window, 'localStorage', {
    value: {
      getItem: (key: string) => store.get(key) ?? null,
      setItem: (key: string, value: string) => void store.set(key, String(value)),
      removeItem: (key: string) => void store.delete(key),
      clear: () => store.clear(),
      key: (index: number) => Array.from(store.keys())[index] ?? null,
      get length() {
        return store.size
      },
    },
    writable: true,
  })
}

// jsdom doesn't implement matchMedia; components like ThemeProvider read it for
// the prefers-color-scheme default.
if (!window.matchMedia) {
  window.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }) as unknown as MediaQueryList
}
