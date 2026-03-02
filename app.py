import os
import re
import json
import requests
from datetime import datetime, timezone

import streamlit as st
from openai import OpenAI

# ========== 環境変数の読み込み（Cloudはst.secrets、ローカルは.env） ==========
def get_env(name: str, default: str = "") -> str:
    if hasattr(st, "secrets") and name in st.secrets:
        return st.secrets[name]
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return os.getenv(name, default)


OPENAI_API_KEY = get_env("OPENAI_API_KEY")
NOTION_API_KEY = get_env("NOTION_API_KEY")
NOTION_DATABASE_ID = get_env("NOTION_DATABASE_ID")

client = OpenAI(api_key=OPENAI_API_KEY)

# ================== IPA → Stress 生成 ==================
VOWEL_IPA = "aeiouɑɒɔæɪʊəɜ"


def _ensure_dots(ipa_core: str) -> str:
    s = ipa_core
    s = re.sub(r"(?<!\.)(?=[ˈˌ])", ".", s)
    s = re.sub(rf"([{VOWEL_IPA}ː]+[^{VOWEL_IPA}ˈˌ.]+)(?=[{VOWEL_IPA}])", r"\1.", s)
    s = re.sub(r"\.{2,}", ".", s)
    return s


def _romanize_syllable(s: str) -> str:
    rep = s
    C = [("tʃ", "ch"), ("dʒ", "j"), ("ʃ", "sh"), ("ʒ", "zh"), ("θ", "th"), ("ð", "dh"), ("ŋ", "ng")]
    for k, v in C:
        rep = rep.replace(k, v)

    V = [
        ("oʊ", "oh"), ("eɪ", "ay"), ("aɪ", "eye"), ("aʊ", "ow"), ("ɔɪ", "oy"),
        ("iː", "ee"), ("uː", "oo"),
        ("ɜː", "er"), ("ɑː", "ah"), ("ɔː", "aw"),
        ("ɪ", "i"), ("ʊ", "u"), ("ʌ", "uh"), ("ə", "uh"), ("æ", "a"),
        ("ɑ", "ah"), ("ɒ", "o"), ("ɔ", "aw"),
    ]
    for k, v in V:
        rep = rep.replace(k, v)

    rep = rep.replace("ː", "").replace("ɡ", "g").replace("ɫ", "l").replace("j", "y")
    return (rep.lower().strip() or s)


def accent_from_ipa(ipa: str) -> str:
    core = ipa.strip().strip("/[] ")
    if not core:
        return ""

    tokens = [t for t in core.split() if t]
    outs = []
    for tok in tokens:
        tok = _ensure_dots(tok)
        parts = []
        for syl in [x for x in tok.split(".") if x]:
            primary = syl.startswith("ˈ") or syl.startswith("ˌ")
            bare = syl.lstrip("ˈˌ")
            roman = _romanize_syllable(bare)
            parts.append(roman.upper() if primary else roman.lower())

        res = "-".join(parts).replace("ˈ", "").replace("ˌ", "")
        if len(parts) == 1:
            res = res.upper()
        outs.append(res)

    return " ".join(outs)


# ================== タグ / 判定 ==================
ALLOWED_TAGS = {
    "社会問題", "口語OK", "書き言葉・報道", "フォーマル",
    "専門用語", "法律用語", "ビジネス", "Football",
    "医学", "科学・技術", "IT", "スポーツ", "文化・芸術",
    "食べ物・料理", "歴史", "政治", "自然・環境"
}


def is_phrase(term: str) -> bool:
    return bool(re.search(r"[\s\-]", term.strip()))


def is_gerund_phrase(term: str) -> bool:
    t = term.strip()
    parts = re.findall(r"[A-Za-z']+", t)
    return len(parts) >= 2 and parts[0].lower().endswith("ing")


def is_verb_phrase(term: str) -> bool:
    """
    超軽量ヒューリスティック:
    - 2語以上
    - 先頭語が -ing でない
    - 先頭語が冠詞/前置詞/代名詞/接続詞でない
    """
    t = term.strip()
    parts = re.findall(r"[A-Za-z']+", t)
    if len(parts) < 2:
        return False

    first = parts[0].lower()
    if first.endswith("ing"):
        return False

    stop = {
        "a", "an", "the", "to", "of", "in", "on", "at", "for", "from", "with", "by", "as", "about",
        "and", "or", "but", "nor", "so", "yet",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"
    }
    return first not in stop


def heuristic_tags(word: str) -> set:
    """LLMが空だった時だけの最小保険（デフォでビジネスは付けない）"""
    w = word.lower()
    tags = set()
    if any(k in w for k in ["summit", "sanction", "minister", "administration", "diplomacy", "election", "cabinet"]):
        tags |= {"書き言葉・報道", "政治"}
    if any(k in w for k in ["goal", "assist", "midfielder", "pressing", "striker", "offside", "penalty"]):
        tags.add("Football")
    return tags & ALLOWED_TAGS


def _parse_tags_line(s: str) -> set:
    # 区切りゆれ: , / ， 、 ／ を許容
    parts = [t.strip() for t in re.split(r"[,\u3001\uFF0C/／]+", (s or "")) if t.strip()]
    return set(parts) & ALLOWED_TAGS


def classify_tags_llm(word: str, pos_text: str, definition_jp: str, coll1: str, coll2: str) -> set:
    """
    タグだけは別プロンプトでニュートラルに分類する（ここが「ビジネスばっかり」問題の本丸対策）
    - 一般語は Tags: (empty) を許容
    - 複数領域でよく使う語は最大2つまで付ける（例: goal -> ビジネス,Football）
    """
    prompt = f"""
You are classifying tags for a Japanese learner's Notion vocabulary database.

Term: {word}
POS: {pos_text}
JP definition: {definition_jp}
Collocation 1: {coll1}
Collocation 2: {coll2}

Choose up to TWO tags from this fixed list (or choose none):
社会問題, 口語OK, 書き言葉・報道, フォーマル,
専門用語, 法律用語, ビジネス, Football,
医学, 科学・技術, IT, スポーツ,
文化・芸術, 食べ物・料理, 歴史, 政治, 自然・環境

Rules:
Tag rules (IMPORTANT):
- Do NOT default to ビジネス.
- If you can confidently judge register, prioritize ONE of these:
  * 口語OK (casual / conversational / informal)
  * 書き言葉・報道 (news / report / written style)
  * フォーマル (formal / official / polite)
- Use domain tags only when strongly applicable (e.g., Football, IT, 医学, 法律用語, etc.).
- ビジネス is allowed ONLY when the term is clearly business/office-centric (not just “can be used at work”).
- If none apply or you are not confident, output empty.
- Output at most 2 tags.
Tags: <comma-separated or empty>
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip()
    # 期待フォーマット: "Tags: ..."
    m = re.search(r"^Tags:\s*(.*)$", text, flags=re.M)
    tags_raw = m.group(1).strip() if m else ""
    return _parse_tags_line(tags_raw)


# ================== プロンプト（本体） ==================
def build_prompt(word: str, strict_idiom: bool = False) -> str:
    # 入力がフレーズなら、POSの候補を絞ってLLMのブレを潰す
    forced_pos = None
    if is_gerund_phrase(word):
        forced_pos = "Gerund Phr."
    elif is_verb_phrase(word):
        forced_pos = "Verb Phr."
    elif is_phrase(word):
        forced_pos = "Phrase"

    pos_line = "Noun | Verb | Adjective | Adverb | Preposition | Phrase | Verb Phr. | Gerund Phr."
    if forced_pos:
        pos_line = forced_pos

    base = f"""
You are a TOEIC vocabulary book editor (「金のフレーズ」style).

Provide the following for "{word}".

1) Parts of Speech:
{pos_line}
- Output only from the list above (comma-separated if multiple).

2) Definition in Japanese (accurate, concise)
- If there is only ONE part of speech, do NOT add POS labels (e.g., do NOT use 【N】).
- If there are multiple parts of speech, format like: 【N】... / 【V】... / 【Adj】...

3) TOEIC Collocation (Gold Phrase style, English only)

You MUST output up to TWO lines:
- Collocation 1: for the most important/common POS in TOEIC context
- Collocation 2: for the second POS (if applicable); otherwise leave it empty.

Critical rules:
- For Noun/Verb/Adjective/Adverb/Preposition/Verb Phr./Gerund Phr.:
  * Output a collocation chunk, NOT a full sentence.
  * Use lowercase (unless proper noun/acronym).
  * Do NOT add a period.

- For Phrase / Verb Phr. / Gerund Phr.:
  * Output a fixed chunk (lowercase, no period). Do NOT output a noun-only meaning.

Style guidance:
- Prefer business/workplace context for collocations (NOT for tags).
- Prefer impersonal/report style.
- Avoid "I/we".

4) IPA with syllable dots and stress marks (ˈ primary, ˌ secondary), Cambridge style. Example: ˌpɑːr.ləˈmen.tri
5) Katakana (Japanese reading)

Return output exactly in the format below (no extra lines, no extra labels):

Parts of Speech: <comma-separated>
Definition (JP): <text>
Collocation 1: <text>
Collocation 2: <text or empty>
IPA: <ipa>
Katakana: <カタカナ>
""".strip()

    if forced_pos:
        base += f"\n\nIMPORTANT:\n- The input is treated as {forced_pos}. Do NOT include other parts of speech.\n"
    elif is_phrase(word) or strict_idiom:
        base += "\n\nIMPORTANT:\n- This is likely a multi-word expression. Prefer the most common TOEIC usage.\n"

    return base


# ================== Notion helpers ==================
def notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }


def get_db_schema():
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}"
    r = requests.get(url, headers=notion_headers(), timeout=20)
    if r.status_code != 200:
        return {}
    return r.json().get("properties", {}) or {}


def db_has_property(prop_name: str) -> bool:
    props = get_db_schema()
    return prop_name in props


def first_existing_property(candidates: list[str]):
    props = get_db_schema()
    for c in candidates:
        if c in props:
            return c
    return None


def find_existing_page_by_word(word: str):
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    payload = {"filter": {"property": "Word", "title": {"equals": word}}, "page_size": 1}
    r = requests.post(url, headers=notion_headers(), data=json.dumps(payload), timeout=30)
    if r.status_code != 200:
        return None
    results = r.json().get("results", [])
    return results[0]["id"] if results else None


def update_page_properties(page_id: str, properties: dict):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    payload = {"properties": properties}
    return requests.patch(url, headers=notion_headers(), data=json.dumps(payload), timeout=30)


def safe_property_add(props, key, value, is_title=False, is_multi=False):
    """
    重要:
    - multi_select は空配列 [] を送ることで「既存タグをクリア」できる。
      ここをスキップすると、昔の「ビジネス」が残り続ける。
    """
    if value is None:
        return

    if is_title:
        if isinstance(value, str) and not value.strip():
            return
        props[key] = {"title": [{"text": {"content": value}}]}
        return

    if is_multi:
        # 空でも送る（= クリアできる）
        value_set = set(value) if isinstance(value, (set, list, tuple)) else set()
        props[key] = {"multi_select": [{"name": v} for v in sorted(value_set)]}
        return

    if isinstance(value, str) and not value.strip():
        return
    props[key] = {"rich_text": [{"text": {"content": value}}]}


# ================== 1件処理本体 ==================
def process_word(word: str) -> dict:
    word = re.sub(
        r"\bbring\s+.+?\s+to the table\b",
        "bring something to the table",
        word.strip(),
        flags=re.I,
    )

    prompt = build_prompt(word)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=360,
        temperature=0,
    )
    output_text = resp.choices[0].message.content or ""

    lines = [ln.strip() for ln in output_text.split("\n") if ln.strip()]

    def pick(prefix, default=""):
        for ln in lines:
            if ln.startswith(prefix):
                return ln.replace(prefix, "").strip()
        return default

    # ===== POSの決定 =====
    default_pos = (
        "Gerund Phr." if is_gerund_phrase(word)
        else ("Verb Phr." if is_verb_phrase(word)
              else ("Phrase" if is_phrase(word) else "Noun"))
    )

    pos_raw = pick("Parts of Speech:", "") or pick("Part of Speech:", default_pos)
    pos_items = [p.strip() for p in pos_raw.split(",") if p.strip()]

    # 入力がフレーズならPOSを強制（Verb Phr. を優先）
    if is_gerund_phrase(word):
        pos_items = ["Gerund Phr."]
    elif is_verb_phrase(word):
        pos_items = ["Verb Phr."]
    elif is_phrase(word):
        pos_items = ["Phrase"]

    definition_jp = pick("Definition (JP):", "")

    # ===== 定義JPラベルの後処理 =====
    POS_JP_LABEL = {
        "Noun": "【N】",
        "Verb": "【V】",
        "Adjective": "【Adj】",
        "Adverb": "【Adv】",
        "Preposition": "【Prep】",
        "Phrase": "【Phr】",
        "Verb Phr.": "【Verb Phr.】",
        "Gerund Phr.": "【Gerund Phr.】",
        "Verb Phrase": "【Verb Phr.】",
        "Gerund Phrase": "【Gerund Phr.】",
    }

    def _has_pos_label(s: str) -> bool:
        return bool(re.match(r"^【.+?】", (s or "").strip()))

    # 複数POSなのにラベルが無い場合に補完（単語向け）
    if len(pos_items) >= 2 and definition_jp and not _has_pos_label(definition_jp):
        parts = [p.strip() for p in re.split(r"\s*(?:/|／)\s*", definition_jp) if p.strip()]
        if 2 <= len(parts) <= len(pos_items) <= 3:
            labeled = []
            for p, part in zip(pos_items, parts):
                labeled.append(f"{POS_JP_LABEL.get(p, '【?】')}{part}")
            definition_jp = " / ".join(labeled)

    coll1 = pick("Collocation 1:", "")
    coll2 = pick("Collocation 2:", "")
    ipa = pick("IPA:", "").strip("[]/ ")
    katakana = pick("Katakana:", "")

    # ===== ガード1: Collocation 2は複数POSのときだけ =====
    if len(pos_items) < 2:
        coll2 = ""

    # ===== ガード2: フレーズ系は「定義のスラッシュ分割」を1個に丸める =====
    # 例: take measures -> 「対策を講じる / 対策」みたいに混ざったら左側だけ残す
    if len(pos_items) == 1 and pos_items[0] in ("Verb Phr.", "Gerund Phr.", "Phrase") and definition_jp:
        parts = [p.strip() for p in re.split(r"\s*(?:/|／)\s*", definition_jp) if p.strip()]
        if len(parts) >= 1:
            definition_jp = parts[0]

    # 単一POSなら【N】などのラベルを削除
    if len(pos_items) == 1 and definition_jp:
        definition_jp = re.sub(r"^【.*?】\s*", "", definition_jp)

    pron_stress = accent_from_ipa(ipa)

    # ===== タグ（LLM分類 + 最小保険）=====
    pos_text_for_tag = ", ".join(pos_items) if pos_items else default_pos
    tags = classify_tags_llm(
        word=word,
        pos_text=pos_text_for_tag,
        definition_jp=definition_jp,
        coll1=coll1,
        coll2=coll2,
    )

    # ===== POS表記（Notion側）=====
    pos_map = {
        "Noun": "Noun",
        "Verb": "V[I/T]",
        "Adjective": "Adj.",
        "Adverb": "Adv.",
        "Preposition": "Prep.",
        "Phrase": "Phr.",
        "Verb Phr.": "Verb Phr.",
        "Gerund Phr.": "Gerund Phr.",
        "Verb Phrase": "Verb Phr.",
        "Gerund Phrase": "Gerund Phr.",
    }
    pos_multi = [pos_map.get(p, p) for p in pos_items]

    if not pos_multi:
        pos_multi = (
            ["Gerund Phr."] if is_gerund_phrase(word)
            else (["Verb Phr."] if is_verb_phrase(word)
                  else (["Phr."] if is_phrase(word) else ["Noun"]))
        )

    # ===== Notion: Example Sentence 2 の実プロパティ名を探す =====
    ex1_prop = "Example Sentence"
    ex2_prop = first_existing_property([
        "Example Sentence 2",
        "Example Sentence Ⅱ",
        "Example Sentence (2)",
        "Example Sentence2",
        "Example Sentence ②",
        "Example Sentence (Second)",
        "Example Sentence - 2",
    ])
    notes_prop = first_existing_property(["Notes", "Note", "Memo", "メモ", "ノート"])

    props = {}
    safe_property_add(props, "Word", word, is_title=True)
    props["A Part of Speech"] = {"multi_select": [{"name": p} for p in pos_multi]}
    safe_property_add(props, "Definition (JP)", definition_jp)

    # 1本目は必ず Example Sentence
    safe_property_add(props, ex1_prop, coll1)

    # 2本目：複数POSのときだけ入れる（上でcoll2落としてるのでここは安全）
    if coll2:
        if ex2_prop:
            safe_property_add(props, ex2_prop, coll2)
        elif notes_prop:
            safe_property_add(props, notes_prop, f"Collocation 2: {coll2}")
        else:
            if coll1:
                safe_property_add(props, ex1_prop, f"{coll1} / {coll2}")
            else:
                safe_property_add(props, ex1_prop, coll2)

    safe_property_add(props, "Stress", pron_stress)
    safe_property_add(props, "IPA", ipa)
    safe_property_add(props, "Katakana", katakana)

    # 重要: tags が空でも multi_select=[] を送って「過去タグを消す」
    safe_property_add(props, "Tags", tags, is_multi=True)

    if db_has_property("Last Updated"):
        props["Last Updated"] = {"date": {"start": datetime.now(timezone.utc).isoformat()}}

    page_id = find_existing_page_by_word(word)
    if page_id:
        r = update_page_properties(page_id, props)
        status = ("update", r.status_code, r.text[:1000])
    else:
        r = requests.post(
            "https://api.notion.com/v1/pages",
            headers=notion_headers(),
            data=json.dumps({"parent": {"database_id": NOTION_DATABASE_ID}, "properties": props}),
            timeout=30,
        )
        status = ("create", r.status_code, r.text[:1000])

    return {
        "word": word,
        "pos": ", ".join(pos_multi),
        "pos_multi": pos_multi,
        "definition_jp": definition_jp,
        "collocation_1": coll1,
        "collocation_2": coll2,
        "ipa": ipa,
        "stress": pron_stress,
        "katakana": katakana,
        "tags": ", ".join(sorted(tags)) if tags else "",
        "notion_result": status,
        "ex2_property_used": ex2_prop or "",
        "notes_property_used": notes_prop or "",
    }


# ================== Streamlit UI ==================
st.set_page_config(page_title="Notion Vocab App", page_icon="📘")
st.title("📘 Notion Vocab App")

with st.expander("🔑 接続状態", expanded=False):
    ok = all([OPENAI_API_KEY, NOTION_API_KEY, NOTION_DATABASE_ID])
    st.write("OPENAI_API_KEY:", "✅" if OPENAI_API_KEY else "❌")
    st.write("NOTION_API_KEY:", "✅" if NOTION_API_KEY else "❌")
    st.write("NOTION_DATABASE_ID:", "✅" if NOTION_DATABASE_ID else "❌")
    if not ok:
        st.warning("Secrets もしくは .env を設定してください。")

if "term_input" not in st.session_state:
    st.session_state["term_input"] = ""


def _clear_term():
    st.session_state["term_input"] = ""


term = st.text_input(
    "追加したい単語・フレーズを入力（例: bring something to the table）",
    key="term_input"
)

col1, col2 = st.columns([2, 1])
run = col1.button("📌 Notion に登録 / 更新")
col2.button("🫧 クリア", help="入力を空にします", on_click=_clear_term)

if run:
    term_val = st.session_state["term_input"].strip()
    if not term_val:
        st.error("📌 単語・フレーズを入力してください。")
    else:
        with st.spinner("OpenAI → Notion 連携中…"):
            try:
                result = process_word(term_val)
                st.success("☑️ 処理が完了しました！ 🎉")
                st.write("**Word**:", result["word"])
                st.write("**POS**:", result["pos"])
                st.write("**Definition (JP)**:", result["definition_jp"])
                st.write("**Collocation 1**:", result["collocation_1"])
                if result["collocation_2"]:
                    st.write("**Collocation 2**:", result["collocation_2"])
                    if result["ex2_property_used"]:
                        st.caption(f"→ Notion: `{result['ex2_property_used']}` に保存")
                    elif result["notes_property_used"]:
                        st.caption(f"→ Notion: `{result['notes_property_used']}` に退避")
                    else:
                        st.caption("→ Notion: Example Sentence に追記（2枠目が見つからなかったため）")
                st.write("**IPA**:", result["ipa"])
                st.write("**Stress**:", result["stress"])
                st.write("**Katakana**:", result["katakana"])
                st.write("**Tags**:", result["tags"] if result["tags"] else "(none)")
                kind, code, body = result["notion_result"]
                st.write(f"**Notion**: {kind} → status {code}")
                if code not in (200, 201):
                    st.code(body, language="json")
            except Exception as e:
                st.error(f"エラー: {e}")
