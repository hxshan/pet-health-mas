const BASE_URL = "http://127.0.0.1:8000";

/**
 * @param {string} rawTextInput - Original complaint text
 * @param {Record<string,string>} followUpAnswers - Answers keyed by question text
 */
export const analyzeCase = async (rawTextInput, followUpAnswers = {}) => {
  const res = await fetch(`${BASE_URL}/cases/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      raw_text_input: rawTextInput,
      follow_up_answers: followUpAnswers,
    }),
  });

  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  return res.json();
};
