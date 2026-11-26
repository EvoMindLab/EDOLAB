import { LANGUAGE, setLanguage, tr, type Language } from "../lib/i18n";

const OPTIONS: ReadonlyArray<{ id: Language; label: string; lang: string }> = [
  { id: "en", label: "EN", lang: "en" },
  { id: "zh", label: "中文", lang: "zh-CN" },
];

/** English / Chinese switch of the top bar; the choice is kept in this browser. */
export function LanguageSwitch() {
  return (
    <div className="theme-switcher language-switcher" role="radiogroup" aria-label={tr("Interface language")}>
      {OPTIONS.map((option) => {
        const active = option.id === LANGUAGE;
        return (
          <button
            key={option.id}
            type="button"
            role="radio"
            aria-checked={active}
            lang={option.lang}
            title={option.id === "en" ? "English" : "中文"}
            className={"theme-option" + (active ? " is-active" : "")}
            onClick={() => setLanguage(option.id)}
          >
            <span>{option.label}</span>
          </button>
        );
      })}
    </div>
  );
}
