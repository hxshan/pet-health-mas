import { useTheme } from "../context/ThemeContext";

export default function CaseProfilePanel({ profile, symptoms, imageAssessment, triageResult }) {
  const { c } = useTheme();

  const S = {
    root: { display: "flex", flexDirection: "column", height: "100%" },
    header: { flexShrink: 0, height: "48px", display: "flex", alignItems: "center", padding: "0 16px", borderBottom: `1px solid ${c.panelDivider}` },
    headerLabel: { fontSize: "10px", fontWeight: 600, color: c.panelHeaderColor, textTransform: "uppercase", letterSpacing: "0.1em" },
    badge: { marginLeft: "auto", display: "flex", alignItems: "center", gap: "4px", fontSize: "10px", color: c.cyan, fontWeight: 500 },
    dot: { width: "6px", height: "6px", borderRadius: "50%", background: c.cyan, boxShadow: c.isDark ? `0 0 6px ${c.cyan}` : "none" },
    body: { flex: 1, overflowY: "auto", padding: "16px", display: "flex", flexDirection: "column", gap: "16px", background: c.panelBg },
    sectionLabel: { fontSize: "10px", fontWeight: 600, color: c.sectionLabelColor, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "8px", opacity: 0.7 },
    fieldRow: (known) => ({
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "6px 10px", borderRadius: "8px",
      background: known ? c.fieldBg : "transparent",
      borderLeft: known ? `2px solid ${c.fieldBorderLeft}` : "2px solid transparent",
    }),
    fieldKey: { fontSize: "12px", color: c.fieldKey },
    fieldVal: (known) => ({ fontSize: "12px", fontWeight: 500, color: known ? c.fieldVal : c.textMuted }),
    divider: { borderTop: `1px solid ${c.panelDivider}` },
    tag: {
      display: "inline-block", background: c.tagBg, color: c.tagColor,
      border: `1px solid ${c.tagBorder}`, fontSize: "11px", fontWeight: 500,
      padding: "3px 10px", borderRadius: "999px", boxShadow: c.tagShadow,
    },
    tagsWrap: { display: "flex", flexWrap: "wrap", gap: "6px" },
    imgTag: {
      display: "inline-block", background: c.imgTagBg, color: c.imgTagColor,
      border: `1px solid ${c.imgTagBorder}`, fontSize: "11px", fontWeight: 500,
      padding: "3px 10px", borderRadius: "999px", boxShadow: c.imgTagShadow,
    },
    imgConfBar: (pct) => ({
      height: "4px", borderRadius: "2px",
      background: `linear-gradient(to right, ${c.imgBarFg} ${pct}%, ${c.imgBarBg} ${pct}%)`,
      marginTop: "4px",
    }),
    emptyBox: {
      border: `1px dashed ${c.emptyBoxBorder}`, borderRadius: "12px",
      padding: "24px 16px", textAlign: "center",
    },
    emptyText: { fontSize: "12px", color: c.emptyBoxText, lineHeight: "1.6" },
  };

  const FIELDS = [
    { label: "Species", key: "species" },
    { label: "Breed",   key: "breed"   },
    { label: "Age",     key: "age"     },
    { label: "Sex",     key: "sex"     },
    { label: "Weight",  key: "weight"  },
  ];

  const hasAnyProfile = profile && Object.values(profile).some(v => v && v !== "unknown");
  const hasSymptoms   = symptoms && symptoms.length > 0;
  const hasImage      = imageAssessment && imageAssessment.image_prediction && imageAssessment.image_prediction !== "unknown";
  const hasTriage     = triageResult && triageResult.urgency_level;
  const isActive      = hasAnyProfile || hasSymptoms || hasImage || hasTriage;

  const URGENCY_COLOR = {
    urgent:      "#f87171",
    vet_soon:    "#fb923c",
    monitor:     "#facc15",
    "non-urgent":"#4ade80",
  };
  const urgencyColor = triageResult ? (URGENCY_COLOR[triageResult.urgency_level] || c.cyan) : c.cyan;

  return (
    <div style={S.root}>
      <div style={S.header}>
        <span style={S.headerLabel}>Case Profile</span>
        {isActive && (
          <span style={S.badge}>
            <span style={S.dot} />
            Building
          </span>
        )}
      </div>

      <div style={S.body}>
        {/* Pet details */}
        <div>
          <p style={S.sectionLabel}>Pet Details</p>
          {FIELDS.map(({ label, key }) => {
            const value = profile?.[key];
            const known = value && value !== "unknown";
            return (
              <div key={key} style={S.fieldRow(known)}>
                <span style={S.fieldKey}>{label}</span>
                <span style={S.fieldVal(known)}>{known ? value : "—"}</span>
              </div>
            );
          })}
        </div>

        <div style={S.divider} />

        {/* Symptoms */}
        <div>
          <p style={S.sectionLabel}>Detected Symptoms</p>
          {hasSymptoms ? (
            <div style={S.tagsWrap}>
              {symptoms.map((s, i) => (
                <span key={i} style={S.tag}>{s.replace(/_/g, " ")}</span>
              ))}
            </div>
          ) : (
            <p style={{ fontSize: "12px", color: c.textMuted, fontStyle: "italic" }}>None detected yet</p>
          )}
        </div>

        <div style={S.divider} />

        {/* Image Assessment (Agent 3) */}
        <div>
          <p style={S.sectionLabel}>Image Assessment</p>
          {hasImage ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              <div style={S.fieldRow(true)}>
                <span style={S.fieldKey}>Prediction</span>
                <span style={{ ...S.fieldVal(true), fontSize: "11px" }}>
                  {imageAssessment.image_prediction}
                </span>
              </div>
              {imageAssessment.confidence != null && (
                <div style={{ padding: "0 10px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "2px" }}>
                    <span style={{ fontSize: "10px", color: c.fieldKey }}>Confidence</span>
                    <span style={{ fontSize: "10px", color: c.imgTagColor }}>
                      {Math.round(imageAssessment.confidence * 100)}%
                    </span>
                  </div>
                  <div style={S.imgConfBar(Math.round(imageAssessment.confidence * 100))} />
                </div>
              )}
              {imageAssessment.uncertainty_flag && (
                <span style={{ fontSize: "10px", color: "#f59e0b", padding: "0 10px" }}>
                  Low confidence result
                </span>
              )}
              {imageAssessment.image_validity === "unusable" && (
                <span style={{ fontSize: "10px", color: "#f87171", padding: "0 10px" }}>
                  Image unusable
                </span>
              )}
              {imageAssessment.alternatives?.length > 0 && (
                <div style={{ padding: "0 10px" }}>
                  <p style={{ fontSize: "10px", color: c.fieldKey, marginBottom: "4px" }}>Alternatives</p>
                  <div style={S.tagsWrap}>
                    {imageAssessment.alternatives.map((a, i) => (
                      <span key={i} style={S.imgTag}>
                        {a.condition} {Math.round((a.confidence ?? 0) * 100)}%
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p style={{ fontSize: "12px", color: c.textMuted, fontStyle: "italic" }}>
              {imageAssessment?.image_validity === "unusable"
                ? "Image could not be assessed"
                : "No image submitted"}
            </p>
          )}
        </div>

        {!isActive && (
          <div style={S.emptyBox}>
            <p style={S.emptyText}>Start the conversation to begin<br />building the case profile.</p>
          </div>
        )}

        {/* Triage Report (Agent 4) */}
        {hasTriage && (
          <>
            <div style={S.divider} />
            <div>
              <p style={S.sectionLabel}>Triage Report</p>

              {/* Urgency badge */}
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
                <span style={{
                  fontSize: "11px", fontWeight: 700, padding: "3px 10px",
                  borderRadius: "999px", letterSpacing: "0.05em",
                  background: urgencyColor + "1a",
                  border: `1px solid ${urgencyColor}55`,
                  color: urgencyColor,
                }}>
                  {triageResult.urgency_level?.replace("_", " ").toUpperCase()}
                </span>
                {triageResult.uncertainty_status && triageResult.uncertainty_status !== "confident" && (
                  <span style={{ fontSize: "10px", color: c.textDim }}>
                    {triageResult.uncertainty_status}
                  </span>
                )}
              </div>

              {/* Recommendation */}
              {triageResult.recommendation && (
                <div style={{ ...S.fieldRow(true), marginBottom: "6px" }}>
                  <span style={{ fontSize: "12px", color: c.textMuted, lineHeight: "1.5" }}>
                    {triageResult.recommendation}
                  </span>
                </div>
              )}

              {/* Reasoning */}
              {triageResult.reasoning && (
                <p style={{ fontSize: "11px", color: c.textDim, padding: "0 4px", lineHeight: "1.5", margin: 0 }}>
                  {triageResult.reasoning}
                </p>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
