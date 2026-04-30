const BASE_URL = "http://127.0.0.1:8000";

/**
 * @param {string} rawTextInput       - Original complaint text
 * @param {Record<string,string>} followUpAnswers - Answers keyed by question text
 * @param {string|null} imageBase64   - Base64-encoded image string (data URI ok)
 */
export const analyzeCase = async (rawTextInput, followUpAnswers = {}, imageBase64 = null) => {
  const body = {
    raw_text_input: rawTextInput,
    follow_up_answers: followUpAnswers,
  };

  if (imageBase64) {
    body.image_base64 = imageBase64;
  }

  const res = await fetch(`${BASE_URL}/cases/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  return res.json();
};
