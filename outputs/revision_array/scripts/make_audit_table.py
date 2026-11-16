"""Write the LaTeX rows of the 20NG audit excerpt table (the 15 audited U_all documents) to tab_audit_excerpts.tex."""
import os
import re
import pandas as pd
import array_utils as u

sh = pd.read_csv(os.path.join(u.OUT, "a3_audit_sheet_annotated.csv"))
key = pd.read_csv(os.path.join(u.OUT, "a3_audit_key.csv"))
m = sh.merge(key[["audit_id", "set", "majority_pred"]], on="audit_id")
m = m[m.set == "U_all"].sort_values(["judgement", "audit_id"])


def esc(t):
    t = re.sub(r"\s+", " ", str(t)).strip()
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
                 ("<", r"\textless{}"), (">", r"\textgreater{}"), ('"', "''")]:
        t = t.replace(a, b)
    return t


def short(t, n=70):
    t = re.sub(r"\s+", " ", str(t)).strip().replace(" [...]", "")
    return (t[:n].rsplit(" ", 1)[0] + " …") if len(t) > n else t


BS = chr(92)  # backslash
LD = BS + "ldots"
EOL = BS + BS
TT = BS + "texttt"
rows = []
for _, r in m.iterrows():
    rows.append(f"{r.audit_id} & {TT}{{{esc(r.observed_newsgroup)}}} & {TT}{{{esc(r.majority_pred)}}} & "
                f"{r.n_words} & {esc(short(r.excerpt)).replace(chr(0x2026), LD)} & {r.judgement} " + EOL)
open(os.path.join(u.OUT, "tab_audit_excerpts.tex"), "w", encoding="utf-8").write("\n".join(rows) + "\n")
print("\n".join(rows))
