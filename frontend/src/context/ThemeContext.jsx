import { createContext, useContext, useState } from "react";

// ── Dark theme — exact current colours ───────────────────────────────────────
const DARK = {
  isDark: true,
  // Layout
  pageBg:        "#161614",
  sidebarBg:     "#0f0f0e",
  panelBg:       "#0f0f0e",
  border:        "#252422",
  inputAreaBg:   "#161614",
  // Text
  text:          "#e8e6e3",
  textMuted:     "#c4c0bb",
  textDim:       "#57534e",
  textFaint:     "#3a3632",
  // Logo
  logoColor:     "#00d4ff",
  logoShadow:    "0 0 12px rgba(0,212,255,0.5)",
  logoSub:       "#0e7490",
  // Session
  sessionBg:     "#1e1c1a",
  sessionBorder: "#0e3a47",
  sessionDot:    "#00d4ff",
  sessionDotGlow:"0 0 6px #00d4ff",
  sessionText:   "#c4c0bb",
  // Live badge
  liveColor:     "#00d4ff",
  liveDotGlow:   "0 0 8px #00d4ff",
  // Accents
  cyan:          "#00d4ff",
  pink:          "#e879f9",
  // Agent footer
  agentFooter:   "#2a3a3f",
  // User bubble
  userBubbleBg:    "#0d1a2e",
  userBubbleBorder:"#1e3a5f",
  userBubbleText:  "#e8e6e3",
  // Bot
  botAvatarBg:     "#0d1f26",
  botAvatarBorder: "#00d4ff44",
  botAvatarShadow: "0 0 8px rgba(0,212,255,0.12)",
  botAvatarIcon:   "#00d4ff",
  botText:         "#c4c0bb",
  // Input
  inputBg:          "#0f1923",
  inputBorderRest:  "#1e2a35",
  inputBorderFocus: "#00d4ff55",
  inputShadowFocus: "0 0 0 3px rgba(0,212,255,0.08), 0 0 12px rgba(0,212,255,0.06)",
  inputText:        "#e8e6e3",
  inputPlaceholder: "#2a4a5e",
  // Send button
  sendActiveBg:     "rgba(0,212,255,0.12)",
  sendActiveBorder: "#00d4ff55",
  sendActiveColor:  "#00d4ff",
  sendActiveShadow: "0 0 8px rgba(0,212,255,0.2)",
  sendInactiveBg:   "#111920",
  sendInactiveBorder:"#1e2a35",
  sendInactiveColor:"#1e3040",
  // Camera button
  camBg:          "rgba(0,212,255,0.05)",
  camBorder:      "#2a4a5e",
  camColor:       "#4a8fa8",
  camActiveBg:    "rgba(0,212,255,0.15)",
  camActiveBorder:"#00d4ff88",
  camActiveColor: "#00d4ff",
  // Image preview clear button
  clearBg:    "#1a1917",
  clearBorder:"#44403c",
  clearColor: "#78716c",
  // Typing dots
  dotColor:   "#00d4ff",
  dotShadow:  "0 0 4px #00d4ff",
  // Empty state
  emptyIconBg:     "#0d1f26",
  emptyIconBorder: "#00d4ff33",
  emptyIconShadow: "0 0 16px rgba(0,212,255,0.1)",
  emptyIconColor:  "#00d4ff",
  emptyTitle:      "#e8e6e3",
  emptyBody:       "#57534e",
  hintBg:          "#0a1a1f",
  hintBorder:      "#0e3a47",
  hintText:        "#57534e",
  // Profile panel
  panelHeaderColor: "#78716c",
  panelDivider:     "#292524",
  sectionLabelColor:"#e879f9",
  fieldBg:          "rgba(0,212,255,0.04)",
  fieldBorderLeft:  "rgba(0,212,255,0.25)",
  fieldKey:         "#57534e",
  fieldVal:         "#00d4ff",
  // Symptom tags
  tagBg:     "rgba(232,121,249,0.08)",
  tagColor:  "#e879f9",
  tagBorder: "rgba(232,121,249,0.25)",
  tagShadow: "0 0 6px rgba(232,121,249,0.1)",
  // Image tags
  imgTagBg:     "rgba(0,212,255,0.08)",
  imgTagColor:  "#00d4ff",
  imgTagBorder: "rgba(0,212,255,0.25)",
  imgTagShadow: "0 0 6px rgba(0,212,255,0.08)",
  // Confidence bar
  imgBarFg: "#00d4ff",
  imgBarBg: "#0e2a35",
  // Empty profile box
  emptyBoxBorder: "#292524",
  emptyBoxText:   "#44403c",
  // Disclaimer
  disclaimer: "#3a3632",
  // Toggle button
  toggleBg:     "#1e1c1a",
  toggleBorder: "#2e2b28",
  toggleColor:  "#78716c",
};

// ── Light theme — warm neutral + muted cyan/pink accents ─────────────────────
const LIGHT = {
  isDark: false,
  pageBg:        "#f7f5f2",
  sidebarBg:     "#ede9e3",
  panelBg:       "#ede9e3",
  border:        "#d8d0c8",
  inputAreaBg:   "#f7f5f2",
  text:          "#1c1917",
  textMuted:     "#44403c",
  textDim:       "#78716c",
  textFaint:     "#b8b0a8",
  logoColor:     "#0284c7",
  logoShadow:    "none",
  logoSub:       "#0369a1",
  sessionBg:     "#e2dbd2",
  sessionBorder: "#b8d8e8",
  sessionDot:    "#0284c7",
  sessionDotGlow:"none",
  sessionText:   "#44403c",
  liveColor:     "#0284c7",
  liveDotGlow:   "none",
  cyan:          "#0284c7",
  pink:          "#a21caf",
  agentFooter:   "#a8a29e",
  userBubbleBg:    "#dbeafe",
  userBubbleBorder:"#93c5fd",
  userBubbleText:  "#1e3a5f",
  botAvatarBg:     "#e0f2fe",
  botAvatarBorder: "#7dd3fc",
  botAvatarShadow: "none",
  botAvatarIcon:   "#0284c7",
  botText:         "#1c1917",
  inputBg:          "#ffffff",
  inputBorderRest:  "#d4cfc9",
  inputBorderFocus: "#0284c788",
  inputShadowFocus: "0 0 0 3px rgba(2,132,199,0.1)",
  inputText:        "#1c1917",
  inputPlaceholder: "#a8a29e",
  sendActiveBg:     "rgba(2,132,199,0.1)",
  sendActiveBorder: "#0284c766",
  sendActiveColor:  "#0284c7",
  sendActiveShadow: "0 0 6px rgba(2,132,199,0.15)",
  sendInactiveBg:   "#f0ede8",
  sendInactiveBorder:"#d4cfc9",
  sendInactiveColor:"#c4bdb6",
  camBg:          "rgba(2,132,199,0.06)",
  camBorder:      "#b8d8e8",
  camColor:       "#0284c7",
  camActiveBg:    "rgba(2,132,199,0.18)",
  camActiveBorder:"#0284c7aa",
  camActiveColor: "#0284c7",
  clearBg:    "#f0ede8",
  clearBorder:"#d4cfc9",
  clearColor: "#78716c",
  dotColor:   "#0284c7",
  dotShadow:  "none",
  emptyIconBg:     "#e0f2fe",
  emptyIconBorder: "#7dd3fc",
  emptyIconShadow: "none",
  emptyIconColor:  "#0284c7",
  emptyTitle:      "#1c1917",
  emptyBody:       "#78716c",
  hintBg:          "#f0ede8",
  hintBorder:      "#d4cfc9",
  hintText:        "#78716c",
  panelHeaderColor: "#78716c",
  panelDivider:     "#d8d0c8",
  sectionLabelColor:"#a21caf",
  fieldBg:          "rgba(2,132,199,0.05)",
  fieldBorderLeft:  "rgba(2,132,199,0.3)",
  fieldKey:         "#78716c",
  fieldVal:         "#0284c7",
  tagBg:     "rgba(162,28,175,0.07)",
  tagColor:  "#a21caf",
  tagBorder: "rgba(162,28,175,0.25)",
  tagShadow: "none",
  imgTagBg:     "rgba(2,132,199,0.07)",
  imgTagColor:  "#0284c7",
  imgTagBorder: "rgba(2,132,199,0.3)",
  imgTagShadow: "none",
  imgBarFg: "#0284c7",
  imgBarBg: "#dbeafe",
  emptyBoxBorder: "#d8d0c8",
  emptyBoxText:   "#a8a29e",
  disclaimer: "#c4bdb6",
  toggleBg:     "#e2dbd2",
  toggleBorder: "#d4cfc9",
  toggleColor:  "#78716c",
};

const ThemeContext = createContext({ c: DARK, toggleTheme: () => {} });

export function ThemeProvider({ children }) {
  const [isDark, setIsDark] = useState(true);
  const toggleTheme = () => setIsDark(d => !d);
  return (
    <ThemeContext.Provider value={{ c: isDark ? DARK : LIGHT, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
