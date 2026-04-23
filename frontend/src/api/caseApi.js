const BASE_URL = "http://127.0.0.1:8000";

export const analyzeCase = async (text) => {
  const res = await fetch(`${BASE_URL}/cases/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ raw_text_input: text }),
  });

  return res.json();
};