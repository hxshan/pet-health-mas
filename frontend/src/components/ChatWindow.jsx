import { useEffect, useRef } from "react";

const S = {
  outer:       { flex: 1, overflowY: "auto", minHeight: 0 },
  inner:       { maxWidth: "640px", margin: "0 auto", padding: "40px 20px 16px", display: "flex", flexDirection: "column", gap: "28px" },

  emptyWrap:   { display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "80px 0", textAlign: "center", userSelect: "none" },
  emptyIcon:   { width: "44px", height: "44px", borderRadius: "10px", background: "#0d1f26", border: "1px solid #00d4ff33", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: "20px", boxShadow: "0 0 16px rgba(0,212,255,0.1)" },
  emptyIconSvg:{ width: "20px", height: "20px", color: "#00d4ff" },
  emptyTitle:  { fontSize: "16px", fontWeight: 600, color: "#e8e6e3", margin: "0 0 8px", letterSpacing: "-0.2px" },
  emptyBody:   { fontSize: "13px", color: "#57534e", maxWidth: "280px", lineHeight: "1.6", margin: 0 },
  hintsWrap:   { marginTop: "24px", display: "flex", flexDirection: "column", gap: "6px", width: "100%", maxWidth: "400px" },
  hintItem:    { textAlign: "left", padding: "8px 12px", borderRadius: "6px", border: "1px solid #0e3a47", fontSize: "12px", color: "#57534e", background: "#0a1a1f", cursor: "default", lineHeight: 1.5 },

  userRow:     { display: "flex", justifyContent: "flex-end" },
  userBubble:  { maxWidth: "72%", background: "#0d1a2e", border: "1px solid #1e3a5f", color: "#e8e6e3", borderRadius: "14px", borderTopRightRadius: "4px", padding: "10px 14px", fontSize: "14px", lineHeight: "1.6" },

  botRow:      { display: "flex", gap: "12px", alignItems: "flex-start" },
  botAvatar:   { width: "26px", height: "26px", borderRadius: "6px", background: "#0d1f26", border: "1px solid #00d4ff44", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: "1px", boxShadow: "0 0 8px rgba(0,212,255,0.12)" },
  botAvatarSvg:{ width: "13px", height: "13px", color: "#00d4ff" },
  botText:     { fontSize: "14px", color: "#c4c0bb", lineHeight: "1.7", paddingTop: "2px", flex: 1 },

  dotsRow:     { display: "flex", alignItems: "center", gap: "4px", paddingTop: "6px" },
  dot:         { width: "5px", height: "5px", borderRadius: "50%", background: "#00d4ff", boxShadow: "0 0 4px #00d4ff" },
};

export default function ChatWindow({ messages }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div style={S.outer}>
      <div style={S.inner}>

        {/* Empty state */}
        {messages.length === 0 && (
          <div style={S.emptyWrap}>
            <div style={S.emptyIcon}>
              <svg style={S.emptyIconSvg} fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
              </svg>
            </div>
            <h2 style={S.emptyTitle}>How can I help your pet today?</h2>
            <p style={S.emptyBody}>Describe your pet's symptoms and our agents will guide you through a diagnostic assessment.</p>
            <div style={S.hintsWrap}>
              {[
                "My dog has been vomiting for 2 days and won't eat",
                "My cat is lethargic and drinking a lot of water",
                "My dog is scratching constantly and has red patches",
              ].map((hint, i) => (
                <div key={i} style={S.hintItem}>"{hint}"</div>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, index) => {
          const isUser   = msg.role === "user";
          const isTyping = msg.text === "...";

          if (isUser) {
            return (
              <div key={index} style={S.userRow}>
                <div style={S.userBubble}>
                  {msg.imageBase64 && (
                    <img
                      src={msg.imageBase64}
                      alt="attached"
                      style={{ display: "block", maxWidth: "200px", maxHeight: "160px", objectFit: "cover", borderRadius: "8px", marginBottom: msg.text ? "8px" : 0, border: "1px solid #1e3a5f" }}
                    />
                  )}
                  {msg.text && <span>{msg.text}</span>}
                </div>
              </div>
            );
          }

          return (
            <div key={index} style={S.botRow}>
              <div style={S.botAvatar}>
                <svg style={S.botAvatarSvg} fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25zm.75-12h9v9h-9v-9z" />
                </svg>
              </div>
              <div style={{ flex: 1 }}>
                {isTyping ? (
                  <div style={S.dotsRow}>
                    <span style={{ ...S.dot, animationName: "bounce", animationDuration: "1s", animationDelay: "0ms",   animationIterationCount: "infinite" }} />
                    <span style={{ ...S.dot, animationName: "bounce", animationDuration: "1s", animationDelay: "160ms", animationIterationCount: "infinite" }} />
                    <span style={{ ...S.dot, animationName: "bounce", animationDuration: "1s", animationDelay: "320ms", animationIterationCount: "infinite" }} />
                  </div>
                ) : (
                  <p style={S.botText}>{msg.text}</p>
                )}
              </div>
            </div>
          );
        })}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}

