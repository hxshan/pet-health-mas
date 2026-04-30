import { useState, useRef } from "react";

export default function ChatInput({ onSend, disabled }) {
  const [input, setInput]       = useState("");
  const [focused, setFocused]   = useState(false);
  const [imageFile, setImageFile] = useState(null);   // { name, dataUrl }
  const fileRef = useRef(null);

  const canSend = (input.trim().length > 0 || imageFile) && !disabled;

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setImageFile({ name: file.name, dataUrl: reader.result });
    reader.readAsDataURL(file);
    // reset input so same file can be re-selected
    e.target.value = "";
  };

  const clearImage = () => setImageFile(null);

  const handleSend = () => {
    if (!canSend) return;
    onSend(input, imageFile?.dataUrl ?? null);
    setInput("");
    setImageFile(null);
  };

  const wrapStyle = {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    background: "#0f1923",
    border: `1px solid ${focused ? "#00d4ff55" : "#1e2a35"}`,
    borderRadius: "12px",
    padding: "10px 10px 10px 14px",
    transition: "border-color 0.15s, box-shadow 0.15s",
    boxShadow: focused ? "0 0 0 3px rgba(0,212,255,0.08), 0 0 12px rgba(0,212,255,0.06)" : "none",
  };

  const rowStyle = { display: "flex", alignItems: "flex-end", gap: "10px" };

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

  const iconBtnStyle = (active) => ({
    flexShrink: 0,
    width: "32px",
    height: "32px",
    borderRadius: "8px",
    border: active ? "1px solid #00d4ff88" : "1px solid #2a4a5e",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    background: active ? "rgba(0,212,255,0.15)" : "rgba(0,212,255,0.05)",
    color: active ? "#00d4ff" : "#4a8fa8",
    transition: "background 0.15s, color 0.15s, border-color 0.15s",
    marginBottom: "1px",
  });

  const sendBtnStyle = {
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
      {/* Image preview strip */}
      {imageFile && (
        <div style={{ display: "flex", alignItems: "center", gap: "8px", paddingBottom: "2px" }}>
          <div style={{ position: "relative", display: "inline-flex" }}>
            <img
              src={imageFile.dataUrl}
              alt="attached"
              style={{ height: "48px", width: "48px", objectFit: "cover", borderRadius: "6px", border: "1px solid #00d4ff33" }}
            />
            <button
              onClick={clearImage}
              style={{ position: "absolute", top: "-6px", right: "-6px", width: "16px", height: "16px", borderRadius: "50%", background: "#1a1917", border: "1px solid #44403c", color: "#78716c", fontSize: "10px", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", lineHeight: 1 }}
            >
              ✕
            </button>
          </div>
          <span style={{ fontSize: "11px", color: "#44403c", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "160px" }}>
            {imageFile.name}
          </span>
        </div>
      )}

      <div style={rowStyle}>
        {/* Attach button */}
        <button
          style={iconBtnStyle(!!imageFile)}
          title="Attach image"
          onClick={() => fileRef.current?.click()}
          disabled={disabled}
        >
          <svg style={{ width: "15px", height: "15px" }} fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0zM18.75 10.5h.008v.008h-.008V10.5z" />
          </svg>
        </button>
        <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={handleFileChange} />

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
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
          }}
        />

        {/* Send button */}
        <button style={sendBtnStyle} onClick={handleSend} disabled={!canSend} title="Send (Enter)">
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
    </div>
  );
}
