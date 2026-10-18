"""The H11 regex edit (second -> third person), extracted verbatim from f80_you_edit.py for reuse (Addendum 74, A4)."""
import re
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
OBJ_PREV = set("to with for at about of from on in by like than against toward towards upon without into onto over under behind beside "
               "tell told ask asked thank thanks give gave show showed let see saw hear heard call called love hate help helped remind "
               "reminded warn warned assure assured bet owe believe trust pay paid send sent bring brought teach taught invite invited "
               "join forgive blame congratulate congrats mock mocked call meet met follow followed".split())
def edit(t):
    def sub(m):
        w = m.group(0); lw = w.lower().replace("’", "'")
        rep = {"your": "their", "yours": "theirs", "yourself": "themselves", "yourselves": "themselves", "you're": "they're", "youre": "theyre",
               "you've": "they've", "you'd": "they'd", "you'll": "they'll"}.get(lw)
        if rep is None:                                  # bare 'you': object after a preposition/verb or before punctuation, else subject
            pre = t[: m.start()].split()[-1:] if t[: m.start()].split() else [""]
            prev = re.sub(r"[^a-z]", "", pre[0].lower()); nxt = t[m.end():m.end() + 2]
            rep = "them" if (prev in OBJ_PREV or not nxt.strip() or nxt.strip()[:1] in ".,;:!?)\"") else "they"
        return rep.capitalize() if w[0].isupper() else rep
    t2 = re.sub(r"\byou people\b", lambda m: "Those people" if m.group(0)[0].isupper() else "those people", t, flags=re.I)
    return YOU.sub(sub, t2)
