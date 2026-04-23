import ChatMessage from "./ChatMessage";

export default function ChatWindow({ messages }) {
  return (
    <div className="flex flex-col gap-2 p-4 h-[400px] overflow-y-auto bg-white rounded-lg shadow">
      {messages.map((msg, i) => (
        <ChatMessage key={i} message={msg} />
      ))}
    </div>
  );
}