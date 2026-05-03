import { useEffect, useRef } from "react";
import { useTheme } from "../context/ThemeContext";

const URGENCY_COLOR = {
  urgent:       "#f87171",
  vet_soon:     "#fb923c",
  monitor:      "#facc15",
  "non-urgent": "#4ade80",
};

const URGENCY_LABEL = {
  urgent:       "URGENT",
  vet_soon:     "VET SOON",
  monitor:      "MONITOR",
  "non-urgent": "NON-URGENT",
};

function ConfBar({ value, color, bg }) {
  const pct = Math.round((value ?? 0) * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <div style={{ flex: 1, height: "5px", borderRadius: "3px", background: bg, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", borderRadius: "3px", background: color, transition: "width 0.4s ease" }} />
      </div>
      <span style={{ fontSize: "11px", fontWeight: 600, color, minWidth: "32px", textAlign: "right" }}>{pct}%</span>
    </div>
  );
}

function FinalReportCard({ data, c }) {
  const { finalReport: fr, triageResult: tr, symptomAssessment: sym, imageAssessment: img } = data;
  const urgency      = tr?.urgency_level || "monitor";
  const urgencyColor = URGENCY_COLOR[urgency] || c.cyan;
  const urgencyLabel = URGENCY_LABEL[urgency] || urgency.toUpperCase();

  const summary      = fr?.triage_summary?.summary || "";
  const agreementExp = fr?.triage_summary?.agreement_explanation || "";
  const confExp      = fr?.triage_summary?.confidence_explanation || "";
  const recommendation = fr?.recommendations?.primary || tr?.recommendation || "";
  const nextSteps    = fr?.recommendations?.next_steps || "";
  const reasoning    = fr?.reasoning || tr?.reasoning || "";
  const disclaimer   = fr?.disclaimer || "";

  const symPred  = sym?.top_prediction;
  const symConf  = sym?.confidence;
  const symAlts  = sym?.alternatives || [];
  const imgPred  = img?.image_prediction;
  const imgConf  = img?.confidence;
  const hasImage = img && img.image_validity && !["none","unusable","error",""].includes(img.image_validity);

  const card = {
    borderRadius: "16px",
    border: `1px solid ${urgencyColor}33`,
    background: c.panelBg,
    overflow: "hidden",
    width: "100%",
  };
  const headerBg = {
    background: `linear-gradient(135deg, ${urgencyColor}18 0%, ${c.panelBg} 100%)`,
    borderBottom: `1px solid ${urgencyColor}22`,
    padding: "16px 20px",
    display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "12px",
  };
  const badge = {
    fontSize: "10px", fontWeight: 700, padding: "4px 12px",
    borderRadius: "999px", letterSpacing: "0.08em",
    background: urgencyColor + "22",
    border: `1px solid ${urgencyColor}66`,
    color: urgencyColor, flexShrink: 0,
  };
  const body   = { padding: "16px 20px", display: "flex", flexDirection: "column", gap: "16px" };
  const label  = { fontSize: "10px", fontWeight: 600, color: c.sectionLabelColor, textTransform: "uppercase", letterSpacing: "0.1em", opacity: 0.7, marginBottom: "6px" };
  const divider = { borderTop: `1px solid ${c.panelDivider}` };
  const prose  = { fontSize: "13px", color: c.botText, lineHeight: "1.7", margin: 0 };
  const muted  = { fontSize: "12px", color: c.textMuted, lineHeight: "1.6", margin: 0 };
  const dim    = { fontSize: "11px", color: c.textDim, lineHeight: "1.6", margin: 0 };

  return (
    <div style={card}>

      {/* ── Header: urgency + title ── */}
      <div style={headerBg}>
        <div>
          <p style={{ fontSize: "13px", fontWeight: 600, color: c.text, margin: "0 0 4px", letterSpacing: "-0.1px" }}>
            Triage Report
          </p>
          <p style={{ fontSize: "12px", color: c.textMuted, margin: 0 }}>
            {symPred ? `${symPred.replace(/_/g, " ")} · Agent 4 synthesis` : "Agent 4 synthesis"}
          </p>
        </div>
        <span style={badge}>{urgencyLabel}</span>
      </div>

      <div style={body}>

        {/* ── Summary (LLM-generated) ── */}
        {summary && (
          <div>
            <p style={label}>Summary</p>
            <p style={prose}>{summary}</p>
          </div>
        )}

        {/* ── Recommendation + next steps ── */}
        {(recommendation || nextSteps) && (
          <>
            <div style={divider} />
            <div>
              <p style={label}>Recommendation</p>
              {recommendation && <p style={prose}>{recommendation}</p>}
              {nextSteps && (
                <div style={{ display: "flex", gap: "8px", marginTop: "8px", alignItems: "flex-start" }}>
                  <span style={{ fontSize: "14px", flexShrink: 0, marginTop: "1px" }}>→</span>
                  <p style={muted}>{nextSteps}</p>
                </div>
              )}
            </div>
          </>
        )}

        {/* ── Agent 2 + Agent 3 evidence ── */}
        <div style={divider} />
        <div>
          <p style={label}>Supporting Evidence</p>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>

            {/* Symptom model */}
            {symPred && (
              <div style={{ background: c.fieldBg, borderRadius: "10px", padding: "10px 12px", borderLeft: `3px solid ${c.cyan}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: symConf != null ? "6px" : 0 }}>
                  <span style={{ fontSize: "11px", fontWeight: 600, color: c.cyan, textTransform: "uppercase", letterSpacing: "0.06em" }}>Agent 2 — Symptoms</span>
                  <span style={{ fontSize: "12px", fontWeight: 500, color: c.text }}>{symPred.replace(/_/g, " ")}</span>
                </div>
                {symConf != null && <ConfBar value={symConf} color={c.cyan} bg={c.imgBarBg} />}
                {symAlts.length > 0 && (
                  <p style={{ ...dim, marginTop: "6px" }}>
                    Also considered: {symAlts.map(a => `${(a.condition||"").replace(/_/g," ")} (${Math.round((a.confidence ?? a.probability ?? 0) * 100)}%)`).join(" · ")}
                  </p>
                )}
              </div>
            )}

            {/* Image model */}
            {hasImage && imgPred && (
              <div style={{ background: c.fieldBg, borderRadius: "10px", padding: "10px 12px", borderLeft: `3px solid ${c.pink}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: imgConf != null ? "6px" : 0 }}>
                  <span style={{ fontSize: "11px", fontWeight: 600, color: c.pink, textTransform: "uppercase", letterSpacing: "0.06em" }}>Agent 3 — Image</span>
                  <span style={{ fontSize: "12px", fontWeight: 500, color: c.text }}>{imgPred.replace(/_/g, " ")}</span>
                </div>
                {imgConf != null && <ConfBar value={imgConf} color={c.pink} bg={c.imgBarBg} />}
              </div>
            )}

            {!hasImage && (
              <p style={{ ...dim, fontStyle: "italic" }}>No image was submitted for this case.</p>
            )}
          </div>
        </div>

        {/* ── Agreement / confidence explanations ── */}
        {(agreementExp || confExp) && (
          <>
            <div style={divider} />
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {agreementExp && <p style={muted}>{agreementExp}</p>}
              {confExp       && <p style={dim}>{confExp}</p>}
            </div>
          </>
        )}

        {/* ── Reasoning trace ── */}
        {reasoning && (
          <>
            <div style={divider} />
            <div>
              <p style={label}>Reasoning</p>
              <p style={dim}>{reasoning}</p>
            </div>
          </>
        )}

        {/* ── Disclaimer ── */}
        {disclaimer && (
          <div style={{ background: `${urgencyColor}0d`, border: `1px solid ${urgencyColor}22`, borderRadius: "8px", padding: "10px 12px" }}>
            <p style={{ ...dim, fontStyle: "italic" }}>⚠ {disclaimer}</p>
          </div>
        )}

      </div>
    </div>
  );
}

export default function ChatWindow({ messages }) {
  const { c } = useTheme();
  const bottomRef = useRef(null);

  const S = {
    outer:       { flex: 1, overflowY: "auto", minHeight: 0, background: c.pageBg },
    inner:       { maxWidth: "640px", margin: "0 auto", padding: "40px 20px 16px", display: "flex", flexDirection: "column", gap: "28px" },

    emptyWrap:   { display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "80px 0", textAlign: "center", userSelect: "none" },
    emptyIcon:   { width: "44px", height: "44px", borderRadius: "10px", background: c.emptyIconBg, border: `1px solid ${c.emptyIconBorder}`, display: "flex", alignItems: "center", justifyContent: "center", marginBottom: "20px", boxShadow: c.emptyIconShadow },
    emptyIconSvg:{ width: "20px", height: "20px", color: c.emptyIconColor },
    emptyTitle:  { fontSize: "16px", fontWeight: 600, color: c.emptyTitle, margin: "0 0 8px", letterSpacing: "-0.2px" },
    emptyBody:   { fontSize: "13px", color: c.emptyBody, maxWidth: "280px", lineHeight: "1.6", margin: 0 },
    hintsWrap:   { marginTop: "24px", display: "flex", flexDirection: "column", gap: "6px", width: "100%", maxWidth: "400px" },
    hintItem:    { textAlign: "left", padding: "8px 12px", borderRadius: "6px", border: `1px solid ${c.hintBorder}`, fontSize: "12px", color: c.hintText, background: c.hintBg, cursor: "default", lineHeight: 1.5 },

    userRow:     { display: "flex", justifyContent: "flex-end" },
    userBubble:  { maxWidth: "72%", background: c.userBubbleBg, border: `1px solid ${c.userBubbleBorder}`, color: c.userBubbleText, borderRadius: "14px", borderTopRightRadius: "4px", padding: "10px 14px", fontSize: "14px", lineHeight: "1.6" },

    botRow:      { display: "flex", gap: "12px", alignItems: "flex-start" },
    botAvatar:   { width: "26px", height: "26px", borderRadius: "6px", background: c.botAvatarBg, border: `1px solid ${c.botAvatarBorder}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: "1px", boxShadow: c.botAvatarShadow },
    botAvatarSvg:{ width: "13px", height: "13px", color: c.botAvatarIcon },
    botText:     { fontSize: "14px", color: c.botText, lineHeight: "1.7", paddingTop: "2px", flex: 1 },

    dotsRow:     { display: "flex", alignItems: "center", gap: "4px", paddingTop: "6px" },
    dot:         { width: "5px", height: "5px", borderRadius: "50%", background: c.dotColor, boxShadow: c.dotShadow },
  };

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
                      style={{ display: "block", maxWidth: "200px", maxHeight: "160px", objectFit: "cover", borderRadius: "8px", marginBottom: msg.text ? "8px" : 0, border: `1px solid ${c.userBubbleBorder}` }}
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
                ) : msg.type === "final_report" ? (
                  <FinalReportCard data={msg.data} c={c} />
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

