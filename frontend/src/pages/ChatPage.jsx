import { useState, useRef } from "react";
import ChatWindow from "../components/ChatWindow";
import ChatInput from "../components/ChatInput";
import CaseProfilePanel from "../components/CaseProfilePanel";

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [answers, setAnswers] = useState({});
  const [lastQuestion, setLastQuestion] = useState(null);
  const [loading, setLoading] = useState(false);
  const [petProfile, setPetProfile] = useState({});
  const [extractedSymptoms, setExtractedSymptoms] = useState([]);
  const askedQuestions = useRef(new Set());
  const initialInput = useRef("");

  const sendToBackend = async (answersObj) => {
    const res = await fetch("http://127.0.0.1:8000/cases/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        raw_text_input: initialInput.current,
        follow_up_answers: answersObj,
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

  const handleSend = async (text) => {
    const userMessage = { role: "user", text };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);

    if (messages.length === 0) {
      initialInput.current = text;
    }

    let updatedAnswers = { ...answers };
    if (lastQuestion) {
      updatedAnswers[lastQuestion] = text;
      setAnswers(updatedAnswers);
    }

    setMessages([...updatedMessages, { role: "bot", text: "..." }]);
    setLoading(true);

    try {
      const res = await sendToBackend(updatedAnswers);

      // Update the live case profile after every round
      if (res.pet_profile) setPetProfile(res.pet_profile);
      if (res.extracted_symptoms?.length) setExtractedSymptoms(res.extracted_symptoms);

      const freshQuestions = (res.follow_up_questions || []).filter(
        (q) => !isSimilarToAsked(q)
      );

      if (!freshQuestions.length) {
        const summary = res.final_report?.summary || res.recommendation || "Analysis complete. Please consult a vet.";
        const urgency = res.urgency_level || "unknown";
        const symptom = res.symptom_assessment;

        const botMessages = [
          { role: "bot", text: `✅ ${summary}` },
          { role: "bot", text: `🚨 Urgency: ${urgency}` },
        ];

        if (symptom?.assessment_status === "completed" && symptom.top_prediction) {
          const conf = symptom.confidence != null
            ? ` — ${Math.round(symptom.confidence * 100)}% confidence`
            : "";
          botMessages.push({
            role: "bot",
            text: `🔬 Agent 2 diagnosis: ${symptom.top_prediction}${conf}`,
          });

          if (symptom.local_interpretation) {
            botMessages.push({
              role: "bot",
              text: `📝 ${symptom.local_interpretation}`,
            });
          }

          if (symptom.alternatives?.length) {
            const alts = symptom.alternatives
              .map(a => `${a.condition} (${Math.round((a.confidence ?? a.probability ?? 0) * 100)}%)`)
              .join(", ");
            botMessages.push({
              role: "bot",
              text: `🔁 Other possibilities: ${alts}`,
            });
          }

          if (symptom.uncertainty_flag) {
            botMessages.push({
              role: "bot",
              text: `⚠️ Uncertain: ${symptom.uncertainty_reason}`,
            });
          }
        } else if (symptom?.assessment_status === "needs_more_info") {
          botMessages.push({
            role: "bot",
            text: `⚠️ Agent 2 could not assess — missing: ${(symptom.missing_fields || []).join(", ")}`,
          });
        } else if (symptom?.assessment_status === "error") {
          botMessages.push({
            role: "bot",
            text: `❌ Agent 2 error: ${symptom.error || "unknown error"}`,
          });
        }

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
    <div style={{ display: "flex", height: "100vh", overflow: "hidden", background: "#161614", color: "#e8e6e3", fontFamily: "'Inter', system-ui, sans-serif" }}>

      {/* ── Left nav sidebar ── */}
      <div style={{ width: "216px", flexShrink: 0, background: "#0f0f0e", borderRight: "1px solid #252422", display: "flex", flexDirection: "column", padding: "20px 12px" }}>
        {/* Logo */}
        <div style={{ padding: "0 8px", marginBottom: "28px" }}>
          <p style={{ fontSize: "14px", fontWeight: 700, color: "#00d4ff", lineHeight: 1, margin: 0, letterSpacing: "-0.2px", textShadow: "0 0 12px rgba(0,212,255,0.5)" }}>PetHealth AI</p>
          <p style={{ fontSize: "11px", color: "#0e7490", margin: "4px 0 0", letterSpacing: "0.06em", textTransform: "uppercase" }}>Diagnostic System</p>
        </div>

        {/* Session item */}
        <div style={{ padding: "0 4px" }}>
          <p style={{ fontSize: "10px", fontWeight: 600, color: "#44403c", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "6px", paddingLeft: "4px" }}>Session</p>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "8px 10px", borderRadius: "6px", background: "#1e1c1a", border: "1px solid #0e3a47" }}>
            <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#00d4ff", flexShrink: 0, boxShadow: "0 0 6px #00d4ff" }} />
            <span style={{ fontSize: "12px", color: "#c4c0bb", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>New consultation</span>
          </div>
        </div>

        {/* Footer */}
        <div style={{ marginTop: "auto", padding: "0 8px" }}>
          <p style={{ fontSize: "10px", color: "#2a3a3f", margin: 0, lineHeight: "2" }}>
            <span style={{ color: "#00d4ff", opacity: 0.7 }}>Agent 1</span> — Intake<br />
            <span style={{ color: "#e879f9", opacity: 0.7 }}>Agent 2</span> — Symptoms<br />
            <span style={{ color: "#00d4ff", opacity: 0.7 }}>Agent 3</span> — Triage
          </p>
        </div>
      </div>

      {/* ── Centre: Chat column ── */}
      <div style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0 }}>

        {/* Top bar */}
        <div style={{ flexShrink: 0, height: "48px", borderBottom: "1px solid #252422", display: "flex", alignItems: "center", padding: "0 24px" }}>
          <span style={{ fontSize: "13px", fontWeight: 500, color: "#c4c0bb", letterSpacing: "-0.1px" }}>Pet Health Consultation</span>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", marginLeft: "auto" }}>
            <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#00d4ff", boxShadow: "0 0 8px #00d4ff" }} />
            <span style={{ fontSize: "11px", color: "#00d4ff", fontWeight: 500 }}>Live</span>
          </div>
        </div>

        {/* Chat + input */}
        <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
          <div style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0 }}>
            <ChatWindow messages={messages} />
            {/* Input bar */}
            <div style={{ flexShrink: 0, padding: "12px 20px 18px", borderTop: "1px solid #252422", background: "#161614" }}>
              <div style={{ maxWidth: "640px", margin: "0 auto" }}>
                <ChatInput onSend={handleSend} disabled={loading} />
                <p style={{ textAlign: "center", fontSize: "10px", color: "#3a3632", marginTop: "8px", lineHeight: 1.5 }}>
                  Always consult a qualified veterinarian. This tool does not replace professional advice.
                </p>
              </div>
            </div>
          </div>

          {/* ── Case Profile panel ── */}
          <div style={{ width: "260px", flexShrink: 0, borderLeft: "1px solid #252422", background: "#0f0f0e", overflowY: "auto" }}>
            <CaseProfilePanel profile={petProfile} symptoms={extractedSymptoms} />
          </div>
        </div>
      </div>

    </div>
  );
}