# UI Design System

Frontend-specific reference: component library, theming, and conventions. For product/backend architecture, stack, and API flows, see the root [README.md](../README.md) — this doc doesn't repeat any of that.

## Stack

- **React 19** + **Vite** + **TypeScript**
- **Tailwind CSS v4** (`@tailwindcss/vite`, `@tailwindcss/typography`) — CSS-first config, no `tailwind.config.js`; theme lives in `src/index.css`
- **shadcn/ui** (style `default`, base color `zinc` — see `components.json`) on top of **Radix UI** primitives
- `class-variance-authority` for variant definitions, `tailwind-merge` + `clsx` for class composition

## Theming

CSS custom properties in [src/index.css](src/index.css), values in OKLCH: `background`, `foreground`, `card`, `popover`, `primary`, `secondary`, `muted`, `accent`, `destructive`, `success`, `border`, `input`, `ring`, `radius`. Defined once in `:root`, overridden in `.dark`.

Use `success`/`destructive` for semantic status (ready/online vs. failed/offline) — never raw Tailwind palette colors (`green-500`, `red-500`, etc.), which don't adapt to dark mode or a future theme change. A Tailwind v4 `@theme inline` block maps each token to a `--color-*` utility (e.g. `--color-primary: var(--primary)`), so components just use `bg-primary`, `text-muted-foreground`, etc.

Dark mode is **class-based, not media-query-based**: `@custom-variant dark (&:where(.dark, .dark *))` gates `.dark` variants on a literal `.dark` class on `<html>`, not `prefers-color-scheme` alone.

- [`ThemeProvider`](src/components/layout/ThemeProvider.tsx) mounts at the app root ([src/main.tsx](src/main.tsx)), toggles `.dark` on `document.documentElement`, and persists the choice to `localStorage['theme']`. `prefers-color-scheme` is only the initial default before a user has chosen.
- [`theme-context.ts`](src/components/layout/theme-context.ts) + [`useTheme`](src/hooks/useTheme.ts) expose the current theme and a toggle to components (see the sun/moon toggle in [`Navbar.tsx`](src/components/layout/Navbar.tsx) or [`AgentRail.tsx`](src/components/shell/AgentRail.tsx)).

## Component conventions

- **Variants**: define with `cva()`, type props with `VariantProps<typeof xVariants>`. Reference: [`button.tsx`](src/components/ui/button.tsx) — `variant` (default/destructive/outline/secondary/ghost/link) × `size` (default/sm/lg/icon).
- **Class composition**: always merge classes through `cn()` ([src/lib/utils.ts](src/lib/utils.ts), `twMerge(clsx(inputs))`), so caller-supplied `className` can safely override variant classes.
- **Polymorphism**: use Radix `Slot` + an `asChild` prop when a component needs to render as a different element (see `button.tsx`).
- Path aliases from `components.json`: `@/components`, `@/components/ui`, `@/lib`, `@/lib/utils`, `@/hooks`.

## Adding a new shadcn primitive

```bash
cd ui && npx shadcn add <name>
```

Reads `components.json` and writes into `src/components/ui/`, matching the existing `default`/`zinc` style so new primitives look consistent with what's already there.

## Current primitives (`src/components/ui/`)

avatar, badge, button, card, checkbox, dialog, dropdown-menu, input, label, popover, scroll-area, separator, sheet, switch, textarea

## Folder map (`src/components/*`)

| Folder | What lives here |
|---|---|
| `ui/` | shadcn primitives (above) — generic, no app-specific logic |
| `research/` | Research-session UI: sidebar, query composer/form, streaming progress, report viewer with feedback |
| `chat/` | Chat thread machinery: `GenericChat` composes `ChatThreadContainer` + a transport, plus markdown rendering and the tool menu |
| `shell/` | Top-level app shell: `AppShell` wires auth/health/pages together, `AgentRail` is the left nav rail (agents, theme toggle, billing) |
| `layout/` | Cross-page chrome: `Navbar` (top bar, theme toggle, user menu), the theme system (`ThemeProvider`, `theme-context.ts`), and shared `HealthDot` status indicator |
| `resources/` | RAG resource management: table + upload dialog |
| `agents/` | RAG agent management: card, chat, create-agent sheet |
