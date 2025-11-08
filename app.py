from dotenv import load_dotenv
import os
import openai
import requests
import json
import re
from datetime import datetime, timezone
import streamlit as st

# --- .envファイルを読み込む ---
load_dotenv()

# --- 環境変数からAPIキーを取得 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

# --- OpenAIのキーをセット ---
openai.api_key = OPENAI_API_KEY

# ================== IPA → Stress 生成（辞書風） ==================

VOWEL_IPA = "aeiouɑɒɔæɪʊəɜ"

def _ensure_dots(ipa_core: str) -> str:
    s = ipa_core
    s = re.sub(r"(?<!\.)(?=[ˈˌ])", ".", s)
    s = re.sub(rf"([{VOWEL_IPA}ː]+[^{VOWEL_IPA}ˈˌ.]+)(?=[{VOWEL_IPA}])", r"\1.", s)
    s = re.sub(r"\.{2,}", ".", s)
    return s

def _romanize_syllable(s: str) -> str:
    rep = s
    C = [
        ("tʃ", "ch"), ("dʒ", "j"), ("ʃ", "sh"), ("ʒ", "zh"),
        ("θ", "th"), ("ð", "dh"), ("ŋ", "ng"),
    ]
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
    rep = rep.replace("ː", "")
    rep = rep.replace("ɡ", "g").replace("ɫ", "l")
    rep = rep.replace("j", "y")
    rep = rep.lower().strip()
    return rep or s

def accent_from_ipa(ipa: str) -> str:  # CHANGED: フレーズ対応（単語ごとに処理し、スペース維持）
    core = ipa.strip().strip("/[] ")
    if not core:
        return ""
    tokens = [t for t in core.split() if t]  # 単語境界を維持
    outs = []
    for tok in tokens:
        tok = _ensure_dots(tok)
        parts = []
        for syl in [x for x in tok.split(".") if x]:
            primary = syl.startswith("ˈ")
            secondary = syl.startswith("ˌ")
            bare = syl.lstrip("ˈˌ")
            roman = _romanize_syllable(bare)
            parts.append(roman.upper() if (primary or secondary) else roman.lower())
        res = "-".join(parts).replace("ˈ", "").replace("ˌ", "")
        if len(parts) == 1:
            res = res.upper()
        outs.append(res)
    return " ".join(outs)

# ====== タグ定義 ======
ALLOWED_TAGS = {
    "社会問題", "口語OK", "書き言葉・報道", "フォーマル",
    "専門用語", "法律用語", "ビジネス", "Football",
    "医学", "科学・技術", "IT", "スポーツ", "文化・芸術",
    "食べ物・料理", "歴史", "政治", "自然・環境"
}

# ====== 単語ベースのタグ自動判定（GPTが空の時の保険） ======
def heuristic_tags(word: str) -> set:
    w = word.lower()
    tags = set()
    # ドメインカテゴリ判定
    if w in {"democracy","feminism","inequality","racism","poverty","refugee",
             "gender","discrimination","immigration","homelessness","opioid",
             "climate","activism"}:
        tags.add("社会問題")
    if w in {"lawsuit","litigation","plaintiff","defendant","statute","ordinance",
             "subpoena","appeal","jurisdiction","precedent","constitution",
             "tort","contract"} or w.endswith("act"):
        tags.add("法律用語")
    if w in {"revenue","profit","margin","kpi","roi","stakeholder","synergy",
             "merger","acquisition","quarterly","fiscal","okr","pipeline",
             "invoice","cashflow","ebitda","churn","retention"}:
        tags.add("ビジネス")
    if w in {"goal","assist","midfielder","striker","forward","defender","winger",
             "offside","penalty","header","fixture","derby","counterattack","pressing"}:
        tags.add("Football")
    if w in {"algorithm","protocol","quantum","neural","latency","throughput",
             "container","orchestration","kubernetes","syntax","blockchain"}:
        tags.add("専門用語")
    # 医学
    if any(k in w for k in ["doctor","medicine","health","disease","virus","vaccine","hospital","clinic"]):
        tags.add("医学")
    # 科学・技術
    if any(k in w for k in ["physics","chemistry","biology","experiment","science","scientific","technology","engineering"]):
        tags.add("科学・技術")
    # IT
    if any(k in w for k in ["computer","algorithm","program","coding","software","hardware","server","database","network","internet"]):
        tags.add("IT")
    # スポーツ
    if any(k in w for k in ["baseball","basketball","tennis","cricket","golf","athletic","athlete","sports"]):
        tags.add("スポーツ")
    # 文化・芸術
    if any(k in w for k in ["music","art","painting","film","movie","literature","theater","novel","artist","culture","dance"]):
        tags.add("文化・芸術")
    # 食べ物・料理
    if any(k in w for k in ["food","meal","cuisine","recipe","chef","restaurant","dish","ingredient","cook"]):
        tags.add("食べ物・料理")
    # 歴史
    if any(k in w for k in ["history","historical","ancient","empire","dynasty","revolution","historian"]):
        tags.add("歴史")
    # 政治
    if any(k in w for k in ["politic","politics","government","election","policy","democracy","diplomacy"]):
        tags.add("政治")
    # 自然・環境
    if any(k in w for k in ["nature","natural","environment","ecology","climate","forest","wildlife","plant","animal"]):
        tags.add("自然・環境")
    # レジスタ（口語／フォーマル／書き言葉）
    colloquial = {"hi","yeah","okay","ok","gonna","wanna","dude","bro","buddy",
                  "cool","kinda","sorta","ain't","y'all"}
    formal_keywords = {"therefore","hence","pursuant","notwithstanding","hereby",
                       "whereas","aforementioned","heretofore","therein","thereof"}
    news_keywords = {"summit","ceasefire","sanction","parliament","minister","administration",
                     "diplomacy","alliance","spokesperson","cease-fire"}
    if any(k == w for k in colloquial):
        tags.add("口語OK")
    elif any(k == w for k in formal_keywords):
        tags.add("フォーマル")
    elif any(k == w for k in news_keywords):
        tags.add("書き言葉・報道")
    # タグ選択（ドメイン優先、次にレジスタ）
    PRIORITY = [
        "法律用語","ビジネス","専門用語","Football",
        "医学","科学・技術","IT","スポーツ","文化・芸術",
        "食べ物・料理","歴史","政治","自然・環境",
        "社会問題",
        "フォーマル","書き言葉・報道","口語OK"
    ]
    domain_tags = {"法律用語","ビジネス","専門用語","Football",
                   "医学","科学・技術","IT","スポーツ","文化・芸術",
                   "食べ物・料理","歴史","政治","自然・環境","社会問題"}
    register_tags = {"口語OK","フォーマル","書き言葉・報道"}
    domain = [t for t in tags if t in domain_tags]
    register = [t for t in tags if t in register_tags]
    picked = []
    if domain:
        picked.append(sorted(domain, key=lambda x: PRIORITY.index(x))[0])
    if register:
        picked.append(sorted(register, key=lambda x: PRIORITY.index(x))[0])
    if len(picked) < 2:
        for t in sorted(tags, key=lambda x: PRIORITY.index(x)):
            if t not in picked:
                picked.append(t)
                if len(picked) == 2:
                    break
    return set(picked)

# ====== ここから：フレーズ検出＆プロンプト生成を追加 ======
def is_phrase(term: str) -> bool:  # ADDED
    """空白やハイフンを含むとフレーズ扱い"""
    return bool(re.search(r"[\s\-]", term.strip()))

def build_prompt(word: str, strict_idiom: bool = False) -> str:  # ADDED
    base = f"""
You are a lexicographer and register expert. Provide the following for '{word}'.

1) Part of Speech (choose exactly one): Noun | Verb | Adjective | Adverb | Preposition | Phrase
2) Definition in Japanese (accurate, concise)
3) A simple example sentence in **English only**
4) IPA with syllable dots and stress marks (ˈ primary, ˌ secondary), *Cambridge style*. Example: ˌpɑːr.ləˈmen.tri
5) Katakana (Japanese reading)
6) Tags: choose ANY from this fixed set only:
   社会問題, 口語OK, 書き言葉・報道, フォーマル,
   専門用語, 法律用語, ビジネス, Football,
   医学, 科学・技術, IT, スポーツ,
   文化・芸術, 食べ物・料理, 歴史, 政治, 自然・環境
   - Choose up to 2 tags: ideally 1 register tag (口語OK / 書き言葉・報道 / フォーマル) and 1 domain tag.

Return output exactly in the format below (no extra punctuation, no brackets):

Part of Speech: <one of the six>
Definition (JP): <text>
Example Sentence: <English only>
IPA: <IPA with dots and ˈ/ˌ>
Katakana: <カタカナ>
Tags: <comma-separated (<=2) from the allowed set or empty>
""".strip()
    if is_phrase(word) or strict_idiom:
        extra = """
IMPORTANT:
- This looks like a MULTI-WORD EXPRESSION (idiom / set phrase / phrasal or fixed expression).
- Prefer idiomatic or set-phrase meanings over literal word-by-word translation.
- If a domain-specific idiom exists (e.g., football/business/news), output THAT sense and select an appropriate domain tag.
- Do NOT output literal meanings when idiomatic use is common.
""".strip()
        return base + "\n\n" + extra
    return base

# ====== Notion 重複チェック & 更新ヘルパー ======
def db_has_property(prop_name: str) -> bool:
    url = f"https://api.notion.com/v1/databases/{DATABASE_ID}"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
    }
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        # 取れなかったら安全側で False
        return False
    props = r.json().get("properties", {})
    return prop_name in props

def find_existing_page_by_word(word: str) -> str | None:
    """DB内の 'Word' タイトルが完全一致する既存ページIDを返す（なければ None）。"""
    url = f"https://api.notion.com/v1/databases/{DATABASE_ID}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    payload = {
        "filter": {
            "property": "Word",
            "title": {"equals": word}
        },
        "page_size": 1
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload))
    if r.status_code != 200:
        print(f"⚠️ Notion検索失敗: {r.status_code} {r.text}")
        return None
    results = r.json().get("results", [])
    return results[0]["id"] if results else None

def update_page_properties(page_id: str, properties: dict) -> requests.Response:
    """既存ページのプロパティだけを更新。"""
    url = f"https://api.notion.com/v1/pages/{page_id}"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    payload = {"properties": properties}
    return requests.patch(url, headers=headers, data=json.dumps(payload))

# ====== 空更新防止ヘルパー ======
def safe_property_add(props, key, value, is_title=False, is_multi=False):
    """値が空でないときだけプロパティを追加"""
    if not value:
        return
    if is_title:
        props[key] = {"title": [{"text": {"content": value}}]}
    elif is_multi:
        props[key] = {"multi_select": [{"name": v} for v in sorted(value)]}
    else:
        props[key] = {"rich_text": [{"text": {"content": value}}]}

# ================== メインループ ==================
while True:
    word = input("📌 追加したい単語を入力してください（終了するには 'exit' と入力）： ").strip()
    if word.lower() == "exit":
        print("👋 終了します。")
        break
    if not word:
        continue

    norm = re.sub(r"\bbring\s+.+?\s+to the table\b", "bring something to the table", word.strip(), flags=re.I)
    word = norm

    # CHANGED: 固定のprompt文字列をやめ、フレーズ時は直訳禁止を強調
    prompt = build_prompt(word)  # CHANGED

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=280,
            temperature=0
        )
        output_text = response.choices[0].message.content
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        continue

    lines = [ln.strip() for ln in output_text.split("\n") if ln.strip()]

    def pick(prefix, default=""):
        for ln in lines:
            if ln.startswith(prefix):
                return ln.replace(prefix, "").strip()
        return default

    # CHANGED: フレーズなら既定POSを Phrase に寄せる
    pos_raw = pick("Part of Speech:", "Phrase" if is_phrase(word) else "Noun")  # CHANGED
    definition_jp = pick("Definition (JP):", "")
    example_sentence = pick("Example Sentence:", "")
    ipa = pick("IPA:", "")
    katakana = pick("Katakana:", "")
    tags_raw = pick("Tags:", "")

    # CHANGED: 空白は消さない（単語境界を保つため）
    ipa = ipa.strip("[]/ ")  # CHANGED（.replace(" ", "") を削除）
    pron_stress = accent_from_ipa(ipa)

    gpt_tags = {t.strip() for t in tags_raw.split(",") if t.strip()} & ALLOWED_TAGS
    if not gpt_tags:
        gpt_tags = heuristic_tags(word)

    pos_mapping = {
        "Noun": "Noun",
        "Verb": "V[I/T]",
        "Adjective": "Adj.",
        "Adverb": "Adv.",
        "Preposition": "Prep.",
        "Phrase": "Phr."
    }
    # CHANGED: 未知POS時もフレーズなら最終的に Phr. に寄せる
    pos = pos_mapping.get(pos_raw, "Phr." if is_phrase(word) else "Noun")  # CHANGED
    
    props = {}
    safe_property_add(props, "Word", word, is_title=True)
    props["A Part of Speech"] = {"multi_select": [{"name": pos}]}  # 品詞は必須
    
    safe_property_add(props, "Definition (JP)", definition_jp)
    safe_property_add(props, "Example Sentence", example_sentence)
    safe_property_add(props, "Stress", pron_stress)
    safe_property_add(props, "IPA", ipa)
    safe_property_add(props, "Katakana", katakana)
    safe_property_add(props, "Tags", gpt_tags, is_multi=True)
    
    if db_has_property("Last Updated"):
        props["Last Updated"] = {"date": {"start": datetime.now(timezone.utc).isoformat()}}
    
    notion_data = {
        "parent": {"database_id": DATABASE_ID},
        "properties": props
    }

    notion_url = "https://api.notion.com/v1/pages"
    notion_headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    # ====== 重複チェック：同じ 'Word' が既にあれば新規作成せず更新 ======
    try:
        existing_id = find_existing_page_by_word(word)
        if existing_id:
            # 既存レコードを更新
            upd_res = update_page_properties(existing_id, notion_data["properties"])
            if upd_res.status_code in (200, 201):
                print(f"🔁 Notionの『{word}』を更新しました！")
            else:
                print(f"❌ Notion更新失敗: {upd_res.status_code} {upd_res.text}")
        else:
            # 新規作成
            crt_res = requests.post(
                "https://api.notion.com/v1/pages",
                headers={
                    "Authorization": f"Bearer {NOTION_API_KEY}",
                    "Content-Type": "application/json",
                    "Notion-Version": "2022-06-28"
                },
                data=json.dumps(notion_data)
            )
            if crt_res.status_code in (200, 201):
                print(f"✅ Notionに『{word}』が追加されました！🎉")
            else:
                print(f"❌ Notionへの追加に失敗: {crt_res.status_code} {crt_res.text}")

        # ===== 共通の出力処理 =====
        print(f"📖 品詞: {pos}")
        if definition_jp:
            print(f"📜 日本語の意味: {definition_jp}")
        if example_sentence:
            print(f"📝 例文: {example_sentence}")
        if pron_stress:
            print(f"🔊 発音: {pron_stress}")
        if ipa:
            print(f"🎯 IPA: {ipa}")
        if katakana:
            print(f"🈺 カタカナ: {katakana}")
        if gpt_tags:
            print(f"🏷️ タグ: {', '.join(sorted(gpt_tags))}")
        print()

    except Exception as e:
        print(f"❌ Notion error: {e}")
