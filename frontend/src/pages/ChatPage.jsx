import { useState, useRef } from "react";
import ChatWindow from "../components/ChatWindow";
import ChatInput from "../components/ChatInput";

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [answers, setAnswers] = useState({});
  const [lastQuestion, setLastQuestion] = useState(null);
  const askedQuestions = useRef(new Set()); // ✅ track ALL asked questions
  const initialInput = useRef("");          // ✅ track original complaint

  const sendToBackend = async (text, answersObj) => {
    const res = await fetch("http://127.0.0.1:8000/cases/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        raw_text_input: initialInput.current,  // ✅ always send original
        follow_up_answers: answersObj,
      }),
    });
    return res.json();
  };

  // Simple similarity check — avoids rephrased duplicates
  const isSimilarToAsked = (newQ) => {
    const newWords = new Set(newQ.toLowerCase().split(/\W+/).filter(w => w.length > 3));
    for (const asked of askedQuestions.current) {
      const askedWords = new Set(asked.toLowerCase().split(/\W+/).filter(w => w.length > 3));
      const overlap = [...newWords].filter(w => askedWords.has(w)).length;
      const similarity = overlap / Math.max(newWords.size, askedWords.size);
      if (similarity > 0.5) return true; // 50% word overlap = duplicate
    }
    return false;
  };

  const handleSend = async (text) => {
    const userMessage = { role: "user", text };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);

    // Save first message as the original complaint
    if (messages.length === 0) {
      initialInput.current = text;
    }

    let updatedAnswers = { ...answers };
    if (lastQuestion) {
      updatedAnswers[lastQuestion] = text;
      setAnswers(updatedAnswers);
    }

    // Show typing indicator
    setMessages([...updatedMessages, { role: "bot", text: "..." }]);

    try {
      const res = await sendToBackend(text, updatedAnswers);

      // Filter out already-asked questions on frontend too
      const freshQuestions = (res.follow_up_questions || []).filter(
        (q) => !isSimilarToAsked(q)
      );

      if (!freshQuestions.length) {
        // ✅ Show final report when no more questions
        const report = res.final_report?.summary || res.recommendation || "✅ Analysis complete. Please consult a vet.";
        setMessages([
          ...updatedMessages,
          { role: "bot", text: `✅ ${report}` },
          { role: "bot", text: `🚨 Urgency: ${res.urgency_level || "unknown"}` },
        ]);
        return;
      }

      // Ask next question and track it
      const nextQ = freshQuestions[0];
      askedQuestions.current.add(nextQ);
      setLastQuestion(nextQ);

      setMessages([...updatedMessages, { role: "bot", text: nextQ }]);
    } catch (err) {
      setMessages([...updatedMessages, { role: "bot", text: "❌ Backend error" }]);
    }
  };

  return (
    <div className="max-w-xl mx-auto mt-10">
      <h1 className="text-2xl font-bold text-center mb-4">
        🐾 Pet Health Chatbot
      </h1>
      <ChatWindow messages={messages} />
      <ChatInput onSend={handleSend} />
    </div>
  );
}