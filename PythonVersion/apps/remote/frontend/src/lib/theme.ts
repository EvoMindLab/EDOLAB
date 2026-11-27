import { tr } from "./i18n";
export type ThemeId = "plain" | "oxford" | "ivory" | "sage";

export interface ThemeOption {
  id: ThemeId;
  index: number;
  label: string;
  description: string;
  surfaceSwatch: string;
  accentSwatch: string;
  themeColor: string;
}

export const THEME_OPTIONS: readonly ThemeOption[] = [
  {
    id: "plain",
    index: 1,
    label: tr("Plain"),
    description: tr("Neutral white and grey, charcoal for main actions"),
    surfaceSwatch: "#ededef",
    accentSwatch: "#3a3a3e",
    themeColor: "#f5f5f7",
  },
  {
    id: "oxford",
    index: 2,
    label: tr("Indigo"),
    description: tr("Cool mist-blue canvas, Oxford indigo accent"),
    surfaceSwatch: "#dce6f2",
    accentSwatch: "#1b4978",
    themeColor: "#f1f5fa",
  },
  {
    id: "ivory",
    index: 3,
    label: tr("Paper"),
    description: tr("Warm rice-paper white, chestnut accent"),
    surfaceSwatch: "#ebe3d2",
    accentSwatch: "#7a5230",
    themeColor: "#f6f1e6",
  },
  {
    id: "sage",
    index: 4,
    label: tr("Ink green"),
    description: tr("Soft grey-green, easy on the eyes, pine-green accent"),
    surfaceSwatch: "#dde8e1",
    accentSwatch: "#1f5a44",
    themeColor: "#f1f6f3",
  },
] as const;

const THEME_STORAGE_KEY = "edolab.relay.theme";
const VALID_THEMES = new Set<ThemeId>(THEME_OPTIONS.map((item) => item.id));
const LEGACY_THEME_MAP: Record<string, ThemeId> = {
  graphite: "plain",
  dark: "sage",
};

export const DEFAULT_THEME: ThemeId = "plain";

export const readSavedTheme = (): ThemeId => {
  try {
    const raw = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (!raw) return DEFAULT_THEME;
    if (VALID_THEMES.has(raw as ThemeId)) return raw as ThemeId;
    if (raw in LEGACY_THEME_MAP) return LEGACY_THEME_MAP[raw];
  } catch {
    // ignore storage access errors
  }
  return DEFAULT_THEME;
};

export const applyTheme = (theme: ThemeId): void => {
  const safeTheme = VALID_THEMES.has(theme) ? theme : DEFAULT_THEME;
  const meta = THEME_OPTIONS.find((item) => item.id === safeTheme) ?? THEME_OPTIONS[0];
  document.documentElement.dataset.theme = safeTheme;
  const metaTag = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  if (metaTag) metaTag.content = meta.themeColor;
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, safeTheme);
  } catch {
    // ignore storage quota errors
  }
};

export const initTheme = (): ThemeId => {
  const initial = readSavedTheme();
  applyTheme(initial);
  return initial;
};

export const readThemeCanvasHex = (): string => {
  const computed = getComputedStyle(document.documentElement).getPropertyValue("--bg-canvas").trim();
  return computed || "#f5f5f7";
};
