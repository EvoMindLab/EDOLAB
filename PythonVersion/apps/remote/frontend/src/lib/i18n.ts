// Interface language of Relay: English or Chinese.
//
// The text of the source is English; tr() returns the Chinese text of i18n.zh.ts when the interface is in Chinese.
// The language is the one chosen with the switch in the top bar, otherwise the browser's language; when neither can
// be read, English. Changing the language reloads the page, so that every text is created again in that language.

import { ZH } from "./i18n.zh";

export type Language = "en" | "zh";

const STORAGE_KEY = "edolab.relay.language";

/** The saved choice, otherwise the browser's preferred language; English when nothing can be read. */
export const detectLanguage = (saved: string | null, browser: readonly unknown[]): Language => {
  if (saved === "en" || saved === "zh") return saved;
  const preferred = browser.find((item): item is string => typeof item === "string" && item.length > 0);
  return preferred?.toLowerCase().startsWith("zh") ? "zh" : "en";
};

const readSaved = (): string | null => {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
};

const readBrowser = (): unknown[] => {
  try {
    const { languages, language } = window.navigator;
    return [...(languages ?? []), language];
  } catch {
    return [];
  }
};

export const LANGUAGE: Language = typeof window === "undefined" ? "en" : detectLanguage(readSaved(), readBrowser());

/** Locale for number and date formatting. */
export const LOCALE = LANGUAGE === "zh" ? "zh-CN" : "en-US";

const shown = (value: unknown): string =>
  value === null || value === undefined || typeof value === "boolean" ? "" : String(value);

/** The text in the interface language; {0}, {1}, ... are replaced by the arguments. */
export const tr = (text: string, ...args: unknown[]): string => {
  const template = LANGUAGE === "zh" ? ZH[text] ?? text : text;
  if (!args.length) return template;
  return template.replace(/\{(\d+)\}/g, (match, index: string) =>
    Number(index) < args.length ? shown(args[Number(index)]) : match);
};

export const setLanguage = (language: Language): void => {
  if (language === LANGUAGE) return;
  try {
    window.localStorage.setItem(STORAGE_KEY, language);
  } catch {
    return; // the choice cannot be kept, so the page would come back in the same language
  }
  window.location.reload();
};

if (typeof document !== "undefined") {
  document.documentElement.lang = LANGUAGE === "zh" ? "zh-CN" : "en";
  document.title = tr("EDOLAB Relay · Remote experiment console");
}
