import { useState } from "react";
import ChatWindow from "../components/ChatWindow";
import ChatInput from "../components/ChatInput";

const BASE_URL = "http://127.0.0.1:8000";

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const sendToBackend = async (history) => {
    const res = await fetch(`${BASE_URL}/cases/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: history, 
      }),
    });

    return res.json();
  };

  const handleSend = async (text) => {
    const updated = [...messages, { role: "user", text }];
    setMessages(updated);

    setLoading(true);

    try {
      const res = await sendToBackend(updated);

      // show follow-up questions
      const botMessages = res.follow_up_questions.map((q) => ({
        role: "bot",
        text: q,
      }));

      setMessages([...updated, ...botMessages]);
    } catch {
      setMessages([
        ...updated,
        { role: "bot", text: "Backend error" },
      ]);
    }

    setLoading(false);
  };

  return (
    <div className="max-w-xl mx-auto mt-10">
      <h1 className="text-2xl font-bold text-center mb-4">
        🐾 Pet Health Chatbot
      </h1>

      <ChatWindow messages={messages} />

      {loading && <p className="text-gray-400">Thinking...</p>}

      <ChatInput onSend={handleSend} />
    </div>
  );
}