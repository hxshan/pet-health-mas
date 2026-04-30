import ChatMessage from "./ChatMessage";

export default function ChatWindow({ messages }) {
  return (
    <div className="bg-gray-900 p-4 rounded h-96 overflow-y-auto mb-4">
      {messages.map((msg, index) => (
        <div
          key={index}
          className={`mb-2 ${
            msg.role === "user" ? "text-right" : "text-left"
          }`}
        >
          <span
            className={`inline-block px-3 py-2 rounded-lg ${
              msg.role === "user"
                ? "bg-blue-500 text-white"
                : "bg-gray-700 text-white"
            }`}
          >
            {msg.text === "..." ? (
              <span className="animate-pulse">Typing...</span>
            ) : (
              msg.text
            )}
          </span>
        </div>
      ))}
    </div>
  );
}