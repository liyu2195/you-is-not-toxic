"""Verbatim annotation prompts (pilot). Construct definitions follow the Jigsaw/Civil Comments rater questions."""
TOX_OPTS = ["A", "B", "C", "D"]          # A very toxic, B toxic, C hard to say, D not toxic
IDA_OPTS = ["A", "B"]                    # A yes, B no
SYSTEM = "You are a careful content moderator rating comments posted under online news articles."
TOX_SCALE = ("A) Very toxic: a very hateful, aggressive, or disrespectful comment that is very likely to make you leave a discussion.\n"
             "B) Toxic: a rude, disrespectful, or unreasonable comment that is somewhat likely to make you leave a discussion.\n"
             "C) Hard to say.\n"
             "D) Not toxic.")
IDA_DEF = ("Does the comment contain an identity-based attack, that is, a negative, discriminatory, or hateful statement about a person or "
           "group based on criteria including (but not limited to) race or ethnicity, religion, gender, nationality or citizenship, "
           "disability, age, or sexual orientation?\nA) Yes\nB) No")
CAP = 2000

def _q(t):
    t = (t or "").strip()
    return t[:CAP] + (" [...]" if len(t) > CAP else "")

def tox_prompt(text, parent=None):
    if parent is None:
        return (f"Rate the toxicity of the comment below.\n\n{TOX_SCALE}\n\nComment:\n\"\"\"{_q(text)}\"\"\"\n\n"
                "Answer with a single letter (A, B, C or D).")
    return (f"The comment below is a reply to an earlier comment, which is shown first for context. Rate the toxicity of the reply only.\n\n"
            f"{TOX_SCALE}\n\nEarlier comment (context):\n\"\"\"{_q(parent)}\"\"\"\n\nReply to rate:\n\"\"\"{_q(text)}\"\"\"\n\n"
            "Answer with a single letter (A, B, C or D).")

def ida_prompt(text, parent=None):
    if parent is None:
        return f"{IDA_DEF}\n\nComment:\n\"\"\"{_q(text)}\"\"\"\n\nAnswer with a single letter (A or B)."
    return (f"The comment below is a reply to an earlier comment, which is shown first for context. Judge the reply only.\n\n{IDA_DEF}\n\n"
            f"Earlier comment (context):\n\"\"\"{_q(parent)}\"\"\"\n\nReply to judge:\n\"\"\"{_q(text)}\"\"\"\n\nAnswer with a single letter (A or B).")

TASKS = {"tox": (tox_prompt, TOX_OPTS), "ida": (ida_prompt, IDA_OPTS)}

# CAD (Contextual Abuse Dataset, Vidgen et al. 2021): abuse = identity-, affiliation- or person-directed abuse (codebook summary)
CAD_DEF = ("Is the Reddit entry below abusive? Abusive means it attacks, demeans, threatens or expresses hate toward "
           "(a) a person (another user, the person being replied to, or a named individual), (b) a group defined by identity "
           "(for example race, religion, gender, sexuality, disability or nationality), or (c) a group defined by affiliation "
           "(for example a political party, a profession or an ideology). Disagreement, criticism of ideas, profanity not aimed "
           "at anyone, and pushing back against abuse are not abusive.\nA) Yes\nB) No")
def cad_prompt(text, parent=None):
    if parent is None:
        return f"{CAD_DEF}\n\nEntry:\n\"\"\"{_q(text)}\"\"\"\n\nAnswer with a single letter (A or B)."
    return (f"The Reddit entry below replies to an earlier entry, which is shown first for context. Judge the reply only.\n\n{CAD_DEF}\n\n"
            f"Earlier entry (context):\n\"\"\"{_q(parent)}\"\"\"\n\nReply to judge:\n\"\"\"{_q(text)}\"\"\"\n\nAnswer with a single letter (A or B).")
TASKS["cad"] = (cad_prompt, IDA_OPTS)

# CGA-Wiki (Conversations Gone Awry, Zhang et al. 2018): personal attack, following the Wikipedia Detox / CGA rater question
PA_DEF = ("Does the comment contain a personal attack or harassment, that is, an insulting, belittling, threatening or harassing "
          "remark aimed at a person (for example the editor being replied to, or a third party)? Disagreement, criticism of "
          "edits or arguments, and firm but civil language are not personal attacks.\nA) Yes\nB) No")
def pa_prompt(text, parent=None):
    if parent is None:
        return f"{PA_DEF}\n\nComment (from a Wikipedia talk page):\n\"\"\"{_q(text)}\"\"\"\n\nAnswer with a single letter (A or B)."
    return (f"The comment below replies to an earlier comment on a Wikipedia talk page, which is shown first for context. Judge the reply only.\n\n{PA_DEF}\n\n"
            f"Earlier comment (context):\n\"\"\"{_q(parent)}\"\"\"\n\nReply to judge:\n\"\"\"{_q(text)}\"\"\"\n\nAnswer with a single letter (A or B).")
TASKS["pa"] = (pa_prompt, IDA_OPTS)

# ---- Addendum 24 (second batch) ----
SYSTEMS = {}                      # task-specific system prompts; tasks not listed use SYSTEM
# Qian et al. (2019) Reddit and Gab conversations: workers were shown Facebook's definition of hate speech
HATE_DEF = ("Is the post below hate speech? Hate speech is a direct attack on people based on protected characteristics: race, ethnicity, "
            "national origin, religious affiliation, sexual orientation, caste, sex, gender, gender identity, and serious disease or disability."
            "\nA) Yes\nB) No")
def hate_prompt(text, parent=None):
    if parent is None:
        return f"{HATE_DEF}\n\nPost:\n\"\"\"{_q(text)}\"\"\"\n\nAnswer with a single letter (A or B)."
    return (f"The post below replies to an earlier post in the same online conversation, which is shown first for context. Judge the reply only.\n\n{HATE_DEF}\n\n"
            f"Earlier post (context):\n\"\"\"{_q(parent)}\"\"\"\n\nReply to judge:\n\"\"\"{_q(text)}\"\"\"\n\nAnswer with a single letter (A or B).")
TASKS["hate"] = (hate_prompt, IDA_OPTS)
SYSTEMS["hate"] = "You are a careful content moderator rating posts from online discussion platforms."
# prompt sensitivity, alternative 1: yes/no with the Perspective API definition of toxicity (default system prompt)
TOXBIN_DEF = ("Is the comment below toxic? A toxic comment is a rude, disrespectful, or unreasonable comment that is likely to make people "
              "leave a discussion.\nA) Yes\nB) No")
def toxbin_prompt(text, parent=None):
    if parent is None:
        return f"{TOXBIN_DEF}\n\nComment:\n\"\"\"{_q(text)}\"\"\"\n\nAnswer with a single letter (A or B)."
    return (f"The comment below is a reply to an earlier comment, which is shown first for context. Judge the reply only.\n\n{TOXBIN_DEF}\n\n"
            f"Earlier comment (context):\n\"\"\"{_q(parent)}\"\"\"\n\nReply to judge:\n\"\"\"{_q(text)}\"\"\"\n\nAnswer with a single letter (A or B).")
TASKS["tox_bin"] = (toxbin_prompt, IDA_OPTS)
# prompt sensitivity, alternative 2: the 4-point scale in reverse order (A = not toxic), new wording and system prompt
TOXREV_SCALE = ("A) Not toxic.\nB) Hard to say.\nC) Toxic: rude, disrespectful, or unreasonable; somewhat likely to make someone leave a discussion.\n"
                "D) Very toxic: very hateful, aggressive, or disrespectful; very likely to make someone leave a discussion.")
def toxrev_prompt(text, parent=None):
    if parent is None:
        return (f"How toxic is this comment from a news website's comment section?\n\n{TOXREV_SCALE}\n\nComment:\n\"\"\"{_q(text)}\"\"\"\n\n"
                "Reply with one letter: A, B, C or D.")
    return (f"Here is a comment from a news website's comment section, together with the comment it replies to. How toxic is the reply? "
            f"Rate the reply only.\n\n{TOXREV_SCALE}\n\nComment being replied to:\n\"\"\"{_q(parent)}\"\"\"\n\nReply:\n\"\"\"{_q(text)}\"\"\"\n\n"
            "Reply with one letter: A, B, C or D.")
TASKS["tox_rev"] = (toxrev_prompt, TOX_OPTS)
SYSTEMS["tox_rev"] = "You annotate online comments for a social science research project."
# letter -> score, and flag = score >= 2/3 (toxic or very toxic; "Yes" for the yes/no tasks)
SCORE_MAPS = {"tox": {"A": 1.0, "B": 2 / 3, "C": 1 / 3, "D": 0.0}, "tox_rev": {"D": 1.0, "C": 2 / 3, "B": 1 / 3, "A": 0.0},
              "tox_bin": {"A": 1.0, "B": 0.0}, "ida": {"A": 1.0, "B": 0.0}, "hate": {"A": 1.0, "B": 0.0}, "cad": {"A": 1.0, "B": 0.0}, "pa": {"A": 1.0, "B": 0.0}}

# ---- Addendum 42 (held-out): Bot-Adversarial Dialogue (Xu et al. 2021), crowd question "OK to send in a friendly conversation" ----
UNSAFE_SCALE = ("A) Definitely not OK\nB) Probably not OK\nC) Probably OK\nD) Definitely OK")
UNSAFE_Q = ("Would the message below be OK for a person you just met online to send in a reasonable conversation? Messages that are "
            "offensive, hateful, harassing, sexually explicit, threatening, or that encourage harm or unsafe behavior are not OK.")
def unsafe_prompt(text, parent=None):
    if parent is None:
        return f"{UNSAFE_Q}\n\n{UNSAFE_SCALE}\n\nMessage:\n\"\"\"{_q(text)}\"\"\"\n\nAnswer with a single letter (A, B, C or D)."
    return (f"The message below replies to an earlier message in the same online chat, which is shown first for context. Judge the reply only.\n\n"
            f"{UNSAFE_Q}\n\n{UNSAFE_SCALE}\n\nEarlier message (context):\n\"\"\"{_q(parent)}\"\"\"\n\nReply to judge:\n\"\"\"{_q(text)}\"\"\"\n\n"
            "Answer with a single letter (A, B, C or D).")
TASKS["unsafe"] = (unsafe_prompt, TOX_OPTS)
SYSTEMS["unsafe"] = "You are a careful content moderator rating messages from online chats."
SCORE_MAPS["unsafe"] = {"A": 1.0, "B": 2 / 3, "C": 1 / 3, "D": 0.0}

# ---- Addendum 80, B80e: the main toxicity prompt with an instruction against the second-person mechanism ----
YOU_NOTE = "Addressing someone directly as \"you\" is not toxic in itself."
def toxyou_prompt(text, parent=None):
    if parent is None:
        return (f"Rate the toxicity of the comment below.\n\n{TOX_SCALE}\n\n{YOU_NOTE}\n\nComment:\n\"\"\"{_q(text)}\"\"\"\n\n"
                "Answer with a single letter (A, B, C or D).")
    return (f"The comment below is a reply to an earlier comment, which is shown first for context. Rate the toxicity of the reply only.\n\n"
            f"{TOX_SCALE}\n\n{YOU_NOTE}\n\nEarlier comment (context):\n\"\"\"{_q(parent)}\"\"\"\n\nReply to rate:\n\"\"\"{_q(text)}\"\"\"\n\n"
            "Answer with a single letter (A, B, C or D).")
TASKS["tox_you"] = (toxyou_prompt, TOX_OPTS)
SCORE_MAPS["tox_you"] = {"A": 1.0, "B": 2 / 3, "C": 1 / 3, "D": 0.0}
