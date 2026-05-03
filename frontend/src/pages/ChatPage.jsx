import { useState, useRef } from "react";
import ChatWindow from "../components/ChatWindow";
import ChatInput from "../components/ChatInput";
import CaseProfilePanel from "../components/CaseProfilePanel";
import { useTheme } from "../context/ThemeContext";

export default function ChatPage() {
  const { c, toggleTheme } = useTheme();
  const [messages, setMessages] = useState([]);
  const [answers, setAnswers] = useState({});
  const [lastQuestion, setLastQuestion] = useState(null);
  const [loading, setLoading] = useState(false);
  const [petProfile, setPetProfile] = useState({});
  const [extractedSymptoms, setExtractedSymptoms] = useState([]);
  const [imageAssessment, setImageAssessment] = useState(null);
  const [triageResult, setTriageResult] = useState(null);
  const askedQuestions = useRef(new Set());
  const initialInput = useRef("");
  const persistedImage = useRef(null);   // keep the image for every follow-up round

  const sendToBackend = async (answersObj, imageBase64 = null) => {
    // Use the freshly supplied image, or fall back to the one stored from a
    // previous turn so the image agent always gets the file.
    const imageToSend = imageBase64 ?? persistedImage.current ?? null;
    const res = await fetch("http://127.0.0.1:8000/cases/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        raw_text_input: initialInput.current,
        follow_up_answers: answersObj,
        ...(imageToSend ? { image_base64: imageToSend } : {}),
      }),
    });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    return res.json();
  };

  const isSimilarToAsked = (newQ) => {
    const newWords = new Set(newQ.toLowerCase().split(/\W+/).filter(w => w.length > 3));
    for (const asked of askedQuestions.current) {
      const askedWords = new Set(asked.toLowerCase().split(/\W+/).filter(w => w.length > 3));
      const overlap = [...newWords].filter(w => askedWords.has(w)).length;
      const similarity = overlap / Math.max(newWords.size, askedWords.size);
      if (similarity > 0.5) return true;
    }
    return false;
  };

  const handleSend = async (text, imageBase64 = null) => {
    const userMessage = { role: "user", text, imageBase64 };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);

    if (messages.length === 0) {
      initialInput.current = text;
    }

    // Persist image so every subsequent backend call includes it
    if (imageBase64) {
      persistedImage.current = imageBase64;
    }

    let updatedAnswers = { ...answers };
    if (lastQuestion) {
      updatedAnswers[lastQuestion] = text;
      setAnswers(updatedAnswers);
    }

    setMessages([...updatedMessages, { role: "bot", text: "..." }]);
    setLoading(true);

    try {
      const res = await sendToBackend(updatedAnswers, imageBase64);

      // Update the live case profile after every round
      if (res.pet_profile) setPetProfile(res.pet_profile);
      if (res.extracted_symptoms?.length) setExtractedSymptoms(res.extracted_symptoms);
      if (res.image_assessment && Object.keys(res.image_assessment).length > 0) {
        setImageAssessment(res.image_assessment);
      }
      if (res.triage_result && Object.keys(res.triage_result).length > 0) {
        setTriageResult(res.triage_result);
      }

      const freshQuestions = (res.follow_up_questions || []).filter(
        (q) => !isSimilarToAsked(q)
      );

      if (!freshQuestions.length) {
        const triage = res.triage_result || {};
        const fr     = res.final_report  || {};

        const botMessages = [
          {
            role: "bot",
            type: "final_report",
            data: {
              finalReport:       fr,
              triageResult:      triage,
              symptomAssessment: res.symptom_assessment || {},
              imageAssessment:   res.image_assessment   || {},
            },
          },
        ];

        setMessages([...updatedMessages, ...botMessages]);
        setLastQuestion(null);
        return;
      }

      const nextQ = freshQuestions[0];
      askedQuestions.current.add(nextQ);
      setLastQuestion(nextQ);
      setMessages([...updatedMessages, { role: "bot", text: nextQ }]);
    } catch (err) {
      setMessages([...updatedMessages, { role: "bot", text: `❌ Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden", background: c.pageBg, color: c.text, fontFamily: "'Inter', system-ui, sans-serif" }}>

      {/* ── Left nav sidebar ── */}
      <div style={{ width: "216px", flexShrink: 0, background: c.sidebarBg, borderRight: `1px solid ${c.border}`, display: "flex", flexDirection: "column", padding: "20px 12px" }}>
        {/* Logo */}
        <div style={{ padding: "0 8px", marginBottom: "28px" }}>
          <p style={{ fontSize: "14px", fontWeight: 700, color: c.logoColor, lineHeight: 1, margin: 0, letterSpacing: "-0.2px", textShadow: c.logoShadow }}>PetHealth AI</p>
          <p style={{ fontSize: "11px", color: c.logoSub, margin: "4px 0 0", letterSpacing: "0.06em", textTransform: "uppercase" }}>Diagnostic System</p>
        </div>

        {/* Session item */}
        <div style={{ padding: "0 4px" }}>
          <p style={{ fontSize: "10px", fontWeight: 600, color: c.textDim, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "6px", paddingLeft: "4px" }}>Session</p>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "8px 10px", borderRadius: "6px", background: c.sessionBg, border: `1px solid ${c.sessionBorder}` }}>
            <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: c.sessionDot, flexShrink: 0, boxShadow: c.sessionDotGlow }} />
            <span style={{ fontSize: "12px", color: c.sessionText, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>New consultation</span>
          </div>
        </div>

        {/* Footer */}
        <div style={{ marginTop: "auto", padding: "0 8px" }}>
          <p style={{ fontSize: "10px", color: c.agentFooter, margin: 0, lineHeight: "2" }}>
            <span style={{ color: c.cyan, opacity: 0.7 }}>Agent 1</span> — Intake<br />
            <span style={{ color: c.pink, opacity: 0.7 }}>Agent 2</span> — Symptoms<br />
            <span style={{ color: c.cyan, opacity: 0.7 }}>Agent 3</span> — Images<br />
            <span style={{ color: c.pink, opacity: 0.7 }}>Agent 4</span> — Triage
          </p>
        </div>
      </div>

      {/* ── Centre: Chat column ── */}
      <div style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0 }}>

        {/* Top bar */}
        <div style={{ flexShrink: 0, height: "48px", borderBottom: `1px solid ${c.border}`, display: "flex", alignItems: "center", padding: "0 24px" }}>
          <span style={{ fontSize: "13px", fontWeight: 500, color: c.textMuted, letterSpacing: "-0.1px" }}>Pet Health Consultation</span>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", marginLeft: "auto" }}>
            {/* Live indicator */}
            <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: c.liveColor, boxShadow: c.liveDotGlow }} />
            <span style={{ fontSize: "11px", color: c.liveColor, fontWeight: 500, marginRight: "12px" }}>Live</span>
            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              title={c.isDark ? "Switch to light mode" : "Switch to dark mode"}
              style={{
                width: "28px", height: "28px", borderRadius: "7px",
                border: `1px solid ${c.toggleBorder}`,
                background: c.toggleBg, color: c.toggleColor,
                display: "flex", alignItems: "center", justifyContent: "center",
                cursor: "pointer", transition: "background 0.2s, border-color 0.2s",
              }}
            >
              {c.isDark ? (
                /* Sun icon */
                <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="4" />
                  <path strokeLinecap="round" d="M12 2v2M12 20v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M2 12h2M20 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
                </svg>
              ) : (
                /* Moon icon */
                <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 12.79A9 9 0 1111.21 3a7 7 0 009.79 9.79z" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Chat + input */}
        <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
          <div style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0 }}>
            <ChatWindow messages={messages} />
            {/* Input bar */}
            <div style={{ flexShrink: 0, padding: "12px 20px 18px", borderTop: `1px solid ${c.border}`, background: c.inputAreaBg }}>
              <div style={{ maxWidth: "640px", margin: "0 auto" }}>
                <ChatInput onSend={handleSend} disabled={loading} />
                <p style={{ textAlign: "center", fontSize: "10px", color: c.disclaimer, marginTop: "8px", lineHeight: 1.5 }}>
                  Always consult a qualified veterinarian. This tool does not replace professional advice.
                </p>
              </div>
            </div>
          </div>

          {/* ── Case Profile panel ── */}
          <div style={{ width: "260px", flexShrink: 0, borderLeft: `1px solid ${c.border}`, background: c.panelBg, overflowY: "auto" }}>
            <CaseProfilePanel profile={petProfile} symptoms={extractedSymptoms} imageAssessment={imageAssessment} triageResult={triageResult} />
          </div>
        </div>
      </div>

    </div>
  );
}