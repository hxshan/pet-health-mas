import { useState } from "react";

export default function ChatInput({ onSend }) {
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (!input.trim()) return;
    onSend(input);
    setInput("");
  };

  return (
    <div className="flex gap-2 mt-3">
      <input
        className="flex-1 border rounded-lg px-3 py-2"
        placeholder="Type your message..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />
      <button
        onClick={handleSend}
        className="bg-green-500 text-white px-4 rounded-lg"
      >
        Send
      </button>
    </div>
  );
}