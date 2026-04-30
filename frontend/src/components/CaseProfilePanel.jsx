const S = {
  root: { display: "flex", flexDirection: "column", height: "100%" },
  header: { flexShrink: 0, height: "48px", display: "flex", alignItems: "center", padding: "0 16px", borderBottom: "1px solid #292524" },
  headerLabel: { fontSize: "10px", fontWeight: 600, color: "#78716c", textTransform: "uppercase", letterSpacing: "0.1em" },
  badge: { marginLeft: "auto", display: "flex", alignItems: "center", gap: "4px", fontSize: "10px", color: "#00d4ff", fontWeight: 500 },
  dot: { width: "6px", height: "6px", borderRadius: "50%", background: "#00d4ff", boxShadow: "0 0 6px #00d4ff" },
  body: { flex: 1, overflowY: "auto", padding: "16px", display: "flex", flexDirection: "column", gap: "16px" },
  sectionLabel: { fontSize: "10px", fontWeight: 600, color: "#e879f9", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "8px", opacity: 0.7 },
  fieldRow: (known) => ({
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "6px 10px", borderRadius: "8px",
    background: known ? "rgba(0,212,255,0.04)" : "transparent",
    borderLeft: known ? "2px solid rgba(0,212,255,0.25)" : "2px solid transparent",
  }),
  fieldKey: { fontSize: "12px", color: "#57534e" },
  fieldVal: (known) => ({ fontSize: "12px", fontWeight: 500, color: known ? "#00d4ff" : "#44403c" }),
  divider: { borderTop: "1px solid #292524" },
  tag: {
    display: "inline-block", background: "rgba(232,121,249,0.08)", color: "#e879f9",
    border: "1px solid rgba(232,121,249,0.25)", fontSize: "11px", fontWeight: 500,
    padding: "3px 10px", borderRadius: "999px", boxShadow: "0 0 6px rgba(232,121,249,0.1)",
  },
  tagsWrap: { display: "flex", flexWrap: "wrap", gap: "6px" },
  emptyBox: {
    border: "1px dashed #292524", borderRadius: "12px",
    padding: "24px 16px", textAlign: "center",
  },
  emptyText: { fontSize: "12px", color: "#44403c", lineHeight: "1.6" },
};

const FIELDS = [
  { label: "Species", key: "species" },
  { label: "Breed",   key: "breed"   },
  { label: "Age",     key: "age"     },
  { label: "Sex",     key: "sex"     },
  { label: "Weight",  key: "weight"  },
];

export default function CaseProfilePanel({ profile, symptoms }) {
  const hasAnyProfile = profile && Object.values(profile).some(v => v && v !== "unknown");
  const hasSymptoms = symptoms && symptoms.length > 0;
  const isActive = hasAnyProfile || hasSymptoms;

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
            <p style={{ fontSize: "12px", color: "#44403c", fontStyle: "italic" }}>None detected yet</p>
          )}
        </div>

        {!isActive && (
          <div style={S.emptyBox}>
            <p style={S.emptyText}>Start the conversation to begin<br />building the case profile.</p>
          </div>
        )}
      </div>
    </div>
  );
}
