import { useState } from "react";

export default function ChatInput({ onSend, disabled }) {
  const [input, setInput] = useState("");
  const [focused, setFocused] = useState(false);

  const handleSend = () => {
    if (!input.trim() || disabled) return;
    onSend(input);
    setInput("");
  };

  const canSend = input.trim().length > 0 && !disabled;

  const wrapStyle = {
    display: "flex",
    alignItems: "flex-end",
    gap: "10px",
    background: "#0f1923",
    border: `1px solid ${focused ? "#00d4ff55" : "#1e2a35"}`,
    borderRadius: "12px",
    padding: "10px 10px 10px 14px",
    transition: "border-color 0.15s, box-shadow 0.15s",
    boxShadow: focused ? "0 0 0 3px rgba(0,212,255,0.08), 0 0 12px rgba(0,212,255,0.06)" : "none",
  };

  const textareaStyle = {
    flex: 1,
    background: "transparent",
    border: "none",
    outline: "none",
    color: "#e8e6e3",
    fontSize: "14px",
    lineHeight: "1.6",
    resize: "vertical",
    minHeight: "24px",
    maxHeight: "200px",
    fontFamily: "inherit",
    padding: 0,
  };

  const btnStyle = {
    flexShrink: 0,
    width: "32px",
    height: "32px",
    borderRadius: "8px",
    border: canSend ? "1px solid #00d4ff55" : "1px solid #1e2a35",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: canSend ? "pointer" : "not-allowed",
    background: canSend ? "rgba(0,212,255,0.12)" : "#111920",
    color: canSend ? "#00d4ff" : "#1e3040",
    transition: "background 0.15s, color 0.15s, box-shadow 0.15s",
    boxShadow: canSend ? "0 0 8px rgba(0,212,255,0.2)" : "none",
    marginBottom: "1px",
  };

  return (
    <div style={wrapStyle}>
      <textarea
        rows={1}
        style={textareaStyle}
        placeholder="Describe your pet's symptoms…"
        value={input}
        disabled={disabled}
        onChange={(e) => setInput(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
          }
        }}
      />
      <button style={btnStyle} onClick={handleSend} disabled={!canSend} title="Send (Enter)">
        {disabled ? (
          <svg style={{ width: "14px", height: "14px", animation: "spin 1s linear infinite" }} fill="none" viewBox="0 0 24 24">
            <circle style={{ opacity: 0.25 }} cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
            <path style={{ opacity: 0.75 }} fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
          </svg>
        ) : (
          <svg style={{ width: "14px", height: "14px" }} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14M12 5l7 7-7 7" />
          </svg>
        )}
      </button>
    </div>
  );
}
