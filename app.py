import os
import re
import json
import requests
from datetime import datetime, timezone

import streamlit as st

# ========== 環境変数の読み込み（Cloudはst.secrets、ローカルは.env） ==========
def get_env(name: str, default: str = "") -> str:
    # Streamlit Cloud では st.secrets を優先
    if hasattr(st, "secrets") and name in st.secrets:
        return st.secrets[name]
    # ローカルは .env を許可
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return os.getenv(name, default)

OPENAI_API_KEY      = get_env("OPENAI_API_KEY")
NOTION_API_KEY      = get_env("NOTION_API_KEY")
NOTION_DATABASE_ID  = get_env("NOTION_DATABASE_ID")

# OpenAI（v1系）
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ================== IPA → Stress 生成（あなたのロジックそのまま） ==================
VOWEL_IPA = "aeiouɑɒɔæɪʊəɜ"

def _ensure_dots(ipa_core: str) -> str:
    s = ipa_core
    s = re.sub(r"(?<!\.)(?=[ˈˌ])", ".", s)
    s = re.sub(rf"([{VOWEL_IPA}ː]+[^{VOWEL_IPA}ˈˌ.]+)(?=[{VOWEL_IPA}])", r"\1.", s)
    s = re.sub(r"\.{2,}", ".", s)
    return s

def _romanize_syllable(s: str) -> str:
    rep = s
    C = [("tʃ","ch"),("dʒ","j"),("ʃ","sh"),("ʒ","zh"),("θ","th"),("ð","dh"),("ŋ","ng")]
    for k,v in C: rep = rep.replace(k,v)
    V = [
        ("oʊ","oh"),("eɪ","ay"),("aɪ","eye"),("aʊ","ow"),("ɔɪ","oy"),
        ("iː","ee"),("uː","oo"),
        ("ɜː","er"),("ɑː","ah"),("ɔː","aw"),
        ("ɪ","i"),("ʊ","u"),("ʌ","uh"),("ə","uh"),("æ","a"),
        ("ɑ","ah"),("ɒ","o"),("ɔ","aw"),
    ]
    for k,v in V: rep = rep.replace(k,v)
    rep = rep.replace("ː","").replace("ɡ","g").replace("ɫ","l").replace("j","y")
    return (rep.lower().strip() or s)

def accent_from_ipa(ipa: str) -> str:
    core = ipa.strip().strip("/[] ")
    if not core: return ""
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
        res = "-".join(parts).replace("ˈ","").replace("ˌ","")
        if len(parts) == 1: res = res.upper()
        outs.append(res)
    return " ".join(outs)

ALLOWED_TAGS = {
    "社会問題","口語OK","書き言葉・報道","フォーマル",
    "専門用語","法律用語","ビジネス","Football",
    "医学","科学・技術","IT","スポーツ","文化・芸術",
    "食べ物・料理","歴史","政治","自然・環境"
}

def is_phrase(term: str) -> bool:
    return bool(re.search(r"[\s\-]", term.strip()))

def build_prompt(word: str, strict_idiom: bool=False) -> str:
    base = f"""
You are a lexicographer and register expert. Provide the following for '{word}'.

1) Part of Speech (choose exactly one): Noun | Verb | Adjective | Adverb | Preposition | Phrase
2) Definition in Japanese (accurate, concise)
3) A simple example sentence in **English only**
4) IPA with syllable dots and stress marks (ˈ primary, ˌ secondary), Cambridge style. Example: ˌpɑːr.ləˈmen.tri
5) Katakana (Japanese reading)
6) Tags: choose ANY from this fixed set only:
   社会問題, 口語OK, 書き言葉・報道, フォーマル,
   専門用語, 法律用語, ビジネス, Football,
   医学, 科学・技術, IT, スポーツ,
   文化・芸術, 食べ物・料理, 歴史, 政治, 自然・環境
   - Choose up to 2 tags: ideally 1 register tag and 1 domain tag.

Return output exactly in the format below (no extra punctuation):

Part of Speech: <one>
Definition (JP): <text>
Example Sentence: <English only>
IPA: <ipa>
Katakana: <カタカナ>
Tags: <comma-separated or empty>
""".strip()
    if is_phrase(word) or strict_idiom:
        base += """
IMPORTANT:
- This is likely a multi-word expression. Prefer idiomatic meanings over literal ones.
- If a domain-specific idiom exists, output that and choose an appropriate domain tag.
""".strip()
    return base

def heuristic_tags(word: str) -> set:
    # （略）— 長いのであなたの元コードのままでもOK
    # 短くするため、最低限だけ：
    w = word.lower()
    tags = set()
    if any(k in w for k in ["summit","sanction","minister","administration","diplomacy"]):
        tags.add("書き言葉・報道")
    if any(k in w for k in ["goal","assist","midfielder","pressing"]):
        tags.add("Football")
    if not tags:
        tags.add("口語OK")
    return tags

def db_has_property(prop_name: str) -> bool:
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
    }
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        return False
    return prop_name in r.json().get("properties", {})

def find_existing_page_by_word(word: str):
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    payload = {"filter":{"property":"Word","title":{"equals":word}}, "page_size":1}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
    if r.status_code != 200:
        return None
    results = r.json().get("results", [])
    return results[0]["id"] if results else None

def update_page_properties(page_id: str, properties: dict):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    payload = {"properties": properties}
    return requests.patch(url, headers=headers, data=json.dumps(payload), timeout=30)

def safe_property_add(props, key, value, is_title=False, is_multi=False):
    if not value:
        return
    if is_title:
        props[key] = {"title":[{"text":{"content":value}}]}
    elif is_multi:
        props[key] = {"multi_select":[{"name":v} for v in sorted(value)]}
    else:
        props[key] = {"rich_text":[{"text":{"content":value}}]}

# ========== 1件処理の本体（input() は使わず関数化） ==========
def process_word(word: str) -> dict:
    word = re.sub(r"\bbring\s+.+?\s+to the table\b", "bring something to the table", word.strip(), flags=re.I)
    prompt = build_prompt(word)

    # OpenAI 呼び出し
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # Cloudは3.5よりこっちが安定
        messages=[{"role":"user","content":prompt}],
        max_tokens=280,
        temperature=0,
    )
    output_text = resp.choices[0].message.content

    lines = [ln.strip() for ln in output_text.split("\n") if ln.strip()]
    def pick(prefix, default=""):
        for ln in lines:
            if ln.startswith(prefix):
                return ln.replace(prefix,"").strip()
        return default

    pos_raw         = pick("Part of Speech:", "Phrase" if is_phrase(word) else "Noun")
    definition_jp   = pick("Definition (JP):", "")
    example_sent    = pick("Example Sentence:", "")
    ipa             = pick("IPA:", "").strip("[]/ ")
    katakana        = pick("Katakana:", "")
    tags_raw        = pick("Tags:", "")

    pron_stress = accent_from_ipa(ipa)

    gpt_tags = {t.strip() for t in tags_raw.split(",") if t.strip()} & ALLOWED_TAGS
    if not gpt_tags:
        gpt_tags = heuristic_tags(word)

    pos_mapping = {"Noun":"Noun","Verb":"V[I/T]","Adjective":"Adj.","Adverb":"Adv.","Preposition":"Prep.","Phrase":"Phr."}
    pos = pos_mapping.get(pos_raw, "Phr." if is_phrase(word) else "Noun")

    # Notion 送信
    props = {}
    safe_property_add(props, "Word", word, is_title=True)
    props["A Part of Speech"] = {"multi_select":[{"name":pos}]}
    safe_property_add(props, "Definition (JP)", definition_jp)
    safe_property_add(props, "Example Sentence", example_sent)
    safe_property_add(props, "Stress", pron_stress)
    safe_property_add(props, "IPA", ipa)
    safe_property_add(props, "Katakana", katakana)
    safe_property_add(props, "Tags", gpt_tags, is_multi=True)
    if db_has_property("Last Updated"):
        props["Last Updated"] = {"date":{"start": datetime.now(timezone.utc).isoformat()}}

    page_id = find_existing_page_by_word(word)
    if page_id:
        r = update_page_properties(page_id, props)
        status = ("update", r.status_code, r.text[:1000])
    else:
        r = requests.post(
            "https://api.notion.com/v1/pages",
            headers={
                "Authorization": f"Bearer {NOTION_API_KEY}",
                "Content-Type": "application/json",
                "Notion-Version": "2022-06-28",
            },
            data=json.dumps({"parent":{"database_id":NOTION_DATABASE_ID}, "properties":props}),
            timeout=30,
        )
        status = ("create", r.status_code, r.text[:1000])

    return {
        "word": word,
        "pos": pos,
        "definition_jp": definition_jp,
        "example": example_sent,
        "ipa": ipa,
        "stress": pron_stress,
        "katakana": katakana,
        "tags": ", ".join(sorted(gpt_tags)) if gpt_tags else "",
        "notion_result": status,
    }

# ========== Streamlit UI ==========
st.set_page_config(page_title="Notion Vocab App", page_icon="📘")
st.title("📘 Notion Vocab App")

with st.expander("🔑 接続状態", expanded=False):
    ok = all([OPENAI_API_KEY, NOTION_API_KEY, NOTION_DATABASE_ID])
    st.write("OPENAI_API_KEY:", "✅" if OPENAI_API_KEY else "❌")
    st.write("NOTION_API_KEY:", "✅" if NOTION_API_KEY else "❌")
    st.write("NOTION_DATABASE_ID:", "✅" if NOTION_DATABASE_ID else "❌")
    if not ok:
        st.warning("Secrets もしくは .env を設定してください。")

term = st.text_input("追加したい単語・フレーズを入力（例: bring something to the table）")
col1, col2 = st.columns(2)
run = col1.button("Notion に登録 / 更新")
demo = col2.button("サンプルでテスト", help="network, latency でテストします")

if demo and not term:
    term = "network latency"

if run:
    if not term.strip():
        st.error("単語・フレーズを入力してください。")
    else:
        with st.spinner("OpenAI → Notion 連携中…"):
            try:
                result = process_word(term.strip())
                st.success("処理が完了しました。")
                st.write("**Word**:", result["word"])
                st.write("**POS**:", result["pos"])
                st.write("**Definition (JP)**:", result["definition_jp"])
                st.write("**Example**:", result["example"])
                st.write("**IPA**:", result["ipa"])
                st.write("**Stress**:", result["stress"])
                st.write("**Katakana**:", result["katakana"])
                st.write("**Tags**:", result["tags"])
                kind, code, body = result["notion_result"]
                st.write(f"**Notion**: {kind} → status {code}")
                if code not in (200, 201):
                    st.code(body, language="json")
            except Exception as e:
                st.error(f"エラー: {e}")
