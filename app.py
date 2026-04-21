"""
NOTAM PDF を 1 件アップロードし、Gemini で日本語に要約・整理する Streamlit アプリ。
"""

from __future__ import annotations

import glob
import io
import json
import os
import re
import secrets
import time
import unicodedata
from xml.sax.saxutils import escape
from typing import Optional, Tuple

import fitz  # PyMuPDF
import streamlit as st

from kml_export import augment_spatial_json_with_psn_regex, build_kml_bytes_from_spatial_json

# app.py と同じディレクトリを基準に knowledge/*.pdf を探す
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
from google import genai
from google.genai import types

# ページ設定（タブ名・レイアウト）
st.set_page_config(
    page_title="NOTAM 解析",
    page_icon="📋",
    layout="wide",
)

# Google AI Studio のモデル ID は時期で変わるため、まず既定は新しめの Flash にする。
# 404 のときは MODEL_FALLBACKS の順に自動で試す。
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

MODEL_FALLBACKS: list[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# 前提知識 PDF / NOTAM PDF から Gemini に渡す最大文字数（モデル上限・速度のバランス）
MAX_REFERENCE_CHARS = 90_000
MAX_NOTAM_INPUT_CHARS = 100_000
MULTI_NOTAM_DOWNLOADS_KEY = "_multi_notam_downloads"

# 画面表示順（JSON の各値は「項目名なし」でこの順に箇条書きする）
NOTAM_RESULT_KEYS: list[str] = [
    "notam_number",
    "jst_period",
    "implementation",
    "content_type",
    "applicable_area",
    "equipment",
    "other_conditions",
    "altitude",
]

NOTAM_PARSE_CORE = """
【NOTAM 解析の目的】
送られた NOTAM の原文を、次の要領に従いわかりやすい日本語に解析する。

【解析する項目（この順で考える。ユーザー向けの出力では項目名を付けない）】
1. NOTAM番号（**国内NOTAM番号のみ**。下記「国内と国際の区別」を厳守）
2. 開始終了時間（JST）
3. 実施時間
4. 内容／種別
5. 適用区域
6. 使用機材の説明
7. その他条件
8. 適用高度

【国内NOTAM番号と国際NOTAM番号の区別】
- "notam_number"（画面上の1行目）には **国内NOTAM番号** を記す。**ここに国際番号形式（例 V0333/26）を書くことは誤回答であり禁止**（国内欄が無いなら「不明」または空の運用は JSON ルールに従う）。
- **国際NOTAM番号**（例: `V0333/26`、`A1234/25` のような「英1文字＋3〜5桁の数字＋/＋年の下2桁」形式）は、**notam_number には書かない**（括弧内・注釈としても禁止）。
- **重要**: 国際NOTAM番号は **8項目すべての値の中に一切含めない**（other_conditions にも書かない。転記・引用・「国際: …」形式も禁止）。解釈の内部参照にのみ用いる。
- 原文に国内番号と国際番号の両方がある場合は、**国内側のみ**を notam_number に用いる。
- 国内NOTAM番号が原文から特定できない場合のみ notam_number を「不明」とする。**原文に「国内NOTAM番号」「国内番号」「国内ノータム」「国内通報」等の欄や、その直後の英数字・ハイフン付き番号（例: H26-00333）、または **空港4レター+半角スペース+連番+/+年2桁（例: RJTU 0038/26）** があれば、必ずそこから写経し、安易に「不明」にしない**（推測で国際番号を国内扱いにすることは引き続き禁止）。

【国内NOTAM番号の正確な取り方（最重要）】
- **写経する**: 国内NOTAM番号として採用する文字列は、原文（または国内通報の所定欄）に**実際に印字されている表記を一字一句変えず**転記する。桁数・記号・ハイフンの有無を勝手に補正・推測・整形しない。
- **前提知識 PDF がある場合**: そこに国内番号の欄名・読み取り手順・記載例があれば**必ず参照**し、原文のどの行が国内番号に相当するかを照合する。一般論と原文の表記が食い違うときは**原文の印字を最優先**する。
- **国際番号と混同しない**: `（V0333/26 NOTAMN` のような括弧内のシリーズは国際側。**notam_number に使わない**。
- **電文先頭の6桁などに注意**: 行頭の `271249` や `180034` のように **6桁数字＋半角スペース＋8レター**（例: `271249 RJAAYNYX`、`180034 RJAAYNYX`）の形は、多くの場合 **電文の日時・通報番号・宛先等のヘッダ**であり、国内NOTAM番号と**同一視しない**（notam_number に書かない）。国内番号が別行・別欄にある場合はそちらを使う。
- **複数候補があるとき**: 当該 NOTAM 1 件のブロックのうち、**国内NOTAM番号・国内番号・NOTAM 番号（国内）・国内通報番号** 等のラベルに付随する所定欄の表記を1つだけ採用する。**ラベル付きの欄が読み取れる場合は「不明」にしない**。国際番号や電文先頭6桁+8レター形式で埋めない。

【先頭 NOTAM（"notams" 配列の先頭＝1件目）】
- 配列の **先頭要素は、原文の先頭から数えて最初の NOTAM 通報ブロック**に対応させる（順序の取り違え禁止）。
- **そのブロックに印字された国内NOTAM番号を必ず** `notam_number` に写経する。先頭だけ「不明」や空にすることは、後続件や KML の **notam_index 対応を狂わせる重大ミス**として禁止する（国内欄が原文に存在する限り）。

【書式ルール】
- 各項目は別々に解析し、値だけを用意する（値の文頭に「NOTAM番号」「開始終了時間」などのラベルを付けない）。
- 時刻は UTC から JST（UTC+9）に換算する。
- 日時表記は「2026/10/12 14:32 JST」の形式（日本語の「年・月・日・時・分」の漢字表記は使わない）。
- 有効期間など期間を日本語で明記し、最後に JST を付ける（例: 解析例の2行目のように）。
- 実施時間は内容に応じて明記（例では「期間中継続」）。
- 内容・種別は明記する。
- 適用区域は明記する。**緯度経度・座標に相当する数値は 8 項目すべてに一切書かない**（度分秒・小数度・393621N1414645E のような連結形式・Q 行の NNNNNEEEE 形式・「PSN:」に続く座標列なども禁止）。地名・施設（4レター）・行政区画名・区域の呼称のみで示す（解析例のとおり）。
- 高度は、原文の F)/G) 等に**具体的な上限・下限の根拠があるときだけ**書く（例: 地表から 2000ft AMSL なら「SFC - 2000ft AMSL」）。**高度が原文から特定できない・該当欄が無いときは altitude を空文字 "" とし、一切表記しない**。「SFC - FL999」「FL999」のみ、UNL だけの便宜表記、Q 行の数字を推測で転記したダミー上限などは**禁止**。
- 運用不能は「U/S」と表記する。
- 空港・航空局などは 4 レター（例: RJSY）のみで表記する（「花巻空港」のような地名併記はしない）。
- 次の定型文は出力から省略する:「次期の正確な時間は別途NOTAMで通知される。」
- 原文の [B)項…] / [C)項…] / [E)項…] / [D)項…] / [G)項] / [RMK] などの欄ラベルや番号付きの出典列挙は、出力には繰り返さず、必要な情報だけ本文に取り込む。
- 「該当なし」「該当無し」「N/A」などの定型は **使わない**。情報が無い項目は **空文字 ""** とする（画面ではその行は表示されない）。

【用語の揃え方（例）】
- 「花巻空港 (RJSI) 西エプロン」→「RJSI 西エプロン」
- 「滑走路」→「RWY」
- 「無人航空機（重量 15.8kg）」→「無人航空機：15.8kg」

【進入灯の一部 U/S など】
進入灯の一部が U/S 等の場合は、ソース記載を根拠に WX MINIMA（最低気象条件）の変更の有無を確認し、変更があるときは変更内容を明記する。ない場合はその旨を簡潔に書く。

【解析例（このトーン・粒度を目安にする）】
■解析前
271249 RJAAYNYX
国内NOTAM番号 H26-00333
(V0333/26 NOTAMN
Q)RJJJ/QOLAS/IV/M/E/000/007/3937N14147E005
A)RJJJ B)2601271249 C)2604240800
E)OBST LGT ON POWER LINE TOWER-U/S
PSN : 393621.2N1414645.1E (MIYAKO-SHI IN IWATE) HGT:648FT AMSL
RMK/1.OBST LGT IN PLACE OF OBST LGT AND OBST DAY MARKING
FOR POWER LINE
2.OBST LGT IN PLACE OF OBST DAY MARKING FOR POWER LINE TOWER
3.OTHER SIDE TOWER
PSN:393630.2N1414654.1E (MIYAKO-SHI IN IWATE) HGT:699FT AMSL
F)SFC G)699FT AMSL)

■解析後（箇条書きの各行に項目名は付けない。該当なし等は書かず、無い項目は JSON では ""）
- H26-00333
- 2026/01/27 21:49 JST - 2026/04/24 17:00 JST
- 期間中継続
- 航空障害灯 U/S
- 岩手県宮古市（送電鉄塔）
- 送電線および鉄塔の航空障害灯が、昼間障害標識等の代わりに設置されている。反対側の鉄塔（699ft AMSL）についても同様。
- SFC - 699ft AMSL
※ 上の例では「使用機材」に相当する内容が無いため、JSON の equipment は "" とし、画面ではその行を出さない（国際番号 V0333/26 はどの行にも書かない）。
""".strip()

REFERENCE_KNOWLEDGE_RULE = """
【前提知識テキストがユーザーメッセージに含まれる場合】
メッセージ内の「前提知識」ブロックは、ユーザーが別途アップロードした PDF から抽出したものである。用語の定義・**国内NOTAM番号の欄の見つけ方・ラベル名・記載例**・社内手順など、解析の根拠として優先的に用いる。
**国内NOTAM番号（notam_number）**について、前提知識の説明と原文の印字が食い違う場合は、**原文に印字されている文字列をそのまま** notam_number に用いる（前提知識は「どの行を読むか」の手掛かりにし、番号本体を書き換えない）。
前提知識と NOTAM 原文が食い違う場合は NOTAM 原文を正とし、必要なら other_conditions に簡潔に触れる。
""".strip()

JSON_NOTAM_OUTPUT_RULES = """
【返却形式（厳守）】
- 返答は有効な JSON オブジェクト 1 つのみ。前後に説明文や Markdown を付けない。
- ルートキーは "notams" とし、値は配列とする。
- 入力に複数の NOTAM が含まれる場合は、それぞれ 1 オブジェクトずつ "notams" に**原文の先頭からの出現順**で入れる。1 件のみなら要素は 1 つ。**先頭要素の notam_number 見落としは後続・地図対応を破壊するため禁止**。
- 各オブジェクトは次の 8 キーを必ず持ち、値はすべて文字列とする。情報が無い項目は **空文字 ""**。「該当なし」「該当無し」「N/A」等は **禁止**（不明なら notam_number のみ「不明」可）。キー名と順序は固定:
  1) "notam_number"        … **国内NOTAM番号のみ**（例: H26-00333 や **空港4レター+連番/年2桁の RJTU 0038/26** 等、**国内所定欄の印字を写経**）。**V0333/26 型の国際表記を notam_number に入れた場合は無効**（改変禁止・国際形式は**どのキーにも書かない**）。電文先頭6桁+8レター形式は国内番号とみなさない。「不明」は、原文に国内番号欄・上記いずれかの国内番号らしき表記が**一切無い**ときのみ。
  2) "jst_period"          … 開始終了を JST で「YYYY/MM/DD HH:MM JST - YYYY/MM/DD HH:MM JST」形式
  3) "implementation"      … 実施時間（例: 期間中継続）
  4) "content_type"        … 内容／種別
  5) "applicable_area"      … 適用区域（**緯度経度・座標数値は一切書かない**。地名・施設の 4 レター等のみ）
  6) "equipment"            … 使用機材の説明
  7) "other_conditions"     … その他条件
  8) "altitude"             … 適用高度。原文に根拠があるときのみ「SFC - xxxft AMSL」等で書く。**根拠が無いときは必ず ""**。「SFC - FL999」や FL999 のみなどのダミー・便宜上限は**禁止**
- 上記 8 値は、ユーザー画面では項目名なしの箇条書きとしてそのまま表示するため、値自体に「NOTAM番号：」などの接頭辞を付けない。
- 国際NOTAM番号（英1文字＋数字+/＋年2桁の典型形式）は **いずれの値にも含めない**。
- **緯度経度・座標の数値**（十進度、度分秒、電文の座標列、Q 行の区域コードに相当する座標トークン等）は **いずれの値にも含めない**（高度の ft AMSL 等の表記は可）。
""".strip()

READER_HINTS: dict[str, str] = {
    "私用パイロット向け": (
        "読み手は自家用操縦士・私用飛行を想定する。現場で使う前提で、判断に効く情報を優先する。"
    ),
    "運航・整備・事務向け": (
        "読み手は運航管理者・整備・航空事務を想定する。運航制限・手続・連絡先が分かるよう明確に。"
    ),
    "学習用（やさしい言葉）": (
        "読み手は初学者を想定する。専門用語はできるだけ平易な言葉に言い換え、必要なら括弧で補足する。"
    ),
}

LENGTH_HINTS: dict[str, str] = {
    "短め": "各項目は短く。実施時間・その他条件は一言でよい。",
    "標準（解析例に近い）": "解析例と同等の情報量を目安にする。",
    "詳しく": "content_type / other_conditions を中心に、運用上必要なら詳述してよい。",
}


def build_system_instruction(reader_label: str, length_label: str, extra_user_notes: str) -> str:
    """サイドバーの「解析スタイル」から Gemini 用システム指示を組み立てる。"""
    reader = READER_HINTS.get(reader_label, READER_HINTS["私用パイロット向け"])
    length = LENGTH_HINTS.get(length_label, LENGTH_HINTS["標準（解析例に近い）"])
    parts = [
        "あなたは航空情報（NOTAM）の専門アシスタントです。",
        "ユーザーから渡されるのは、PDF から抽出された NOTAM 関連のテキストです。\n\n",
        NOTAM_PARSE_CORE,
        "\n\n",
        JSON_NOTAM_OUTPUT_RULES,
        "\n\n",
        REFERENCE_KNOWLEDGE_RULE,
        "\n\n【読み手の前提】\n",
        reader,
        "\n\n【詳しさ】\n",
        length,
    ]
    extra = (extra_user_notes or "").strip()
    if extra:
        parts.extend(
            [
                "\n\n【ユーザーからの追加指示（他と矛盾する場合はこちらを優先）】\n",
                extra,
            ]
        )
    return "".join(parts)


_NOTAM_ACCESS_SESSION_KEY = "_notam_app_access_granted"


def get_notam_app_access_password() -> str:
    """
    公開用の入口パスワード（任意）。未設定なら誰でも UI に入れる。
    環境変数 NOTAM_APP_PASSWORD または Streamlit secrets の NOTAM_APP_PASSWORD。
    ※これは Gemini API キーとは別。Gemini キーを URL で共有しないこと。
    """
    env = os.environ.get("NOTAM_APP_PASSWORD", "").strip()
    if env:
        return env
    try:
        return str(st.secrets.get("NOTAM_APP_PASSWORD", "")).strip()
    except Exception:
        return ""


def get_api_key() -> str:
    """サイドバー入力、環境変数、Streamlit secrets の順で API キーを取得。"""
    key = st.session_state.get("gemini_api_key_input", "").strip()
    if key:
        return key
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """PyMuPDF で PDF からプレーンテキストを抽出。"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    parts: list[str] = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()
    return "\n".join(parts).strip()


def reference_pdf_paths_for_autoload() -> list[str]:
    """前提知識として読む PDF のパス一覧（先頭から順に連結する）。

    - 環境変数 NOTAM_REFERENCE_PDF に**ファイル**のパスがある場合: その1件のみ。
    - 上記がなく、knowledge フォルダ内の *.pdf がある場合: **ファイル名の昇順**（大文字小文字は区別しない並び）の全件。
    """
    env = os.environ.get("NOTAM_REFERENCE_PDF", "").strip().strip('"')
    if env and os.path.isfile(env):
        return [env]
    pattern = os.path.join(_APP_DIR, "knowledge", "*.pdf")
    paths = sorted(
        (p for p in glob.glob(pattern) if os.path.isfile(p)),
        key=lambda p: os.path.basename(p).lower(),
    )
    return paths


def load_disk_reference_text() -> Tuple[str, str]:
    """ディスク上の前提知識 PDF を読み込む。(連結テキスト, 人間向けのパス説明)。未設定なら ("", "")。"""
    paths = reference_pdf_paths_for_autoload()
    if not paths:
        return "", ""
    try:
        fingerprint = tuple((p, os.path.getmtime(p)) for p in paths)
    except OSError:
        return "", "\n".join(paths)
    cache = st.session_state.get("_disk_ref_cache")
    if isinstance(cache, dict) and cache.get("fingerprint") == fingerprint and "text" in cache:
        return str(cache["text"]), _format_reference_paths_for_ui(paths)

    chunks: list[str] = []
    total = 0
    # メモリと速度のため、連結の目安上限（Gemini 側ではさらに先頭だけ使用）
    cap = max(MAX_REFERENCE_CHARS * 2, 200_000)
    for p in paths:
        with open(p, "rb") as f:
            blob = f.read()
        t = extract_text_from_pdf(blob)
        header = f"\n\n--- PDF: {os.path.basename(p)} ---\n\n"
        piece = header + t
        if total + len(piece) > cap:
            piece = piece[: max(0, cap - total)]
        chunks.append(piece)
        total += len(piece)
        if total >= cap:
            break
    text = "".join(chunks).strip()
    st.session_state["_disk_ref_cache"] = {"fingerprint": fingerprint, "text": text}
    return text, _format_reference_paths_for_ui(paths)


def _format_reference_paths_for_ui(paths: list[str]) -> str:
    if len(paths) == 1:
        return paths[0]
    return f"{len(paths)} 件: " + "; ".join(os.path.basename(p) for p in paths)


def normalize_model_name(name: str) -> str:
    """UI 用のモデル名を API 向けに整える（空白除去・'models/' 接頭辞の除去）。"""
    n = name.strip()
    if n.startswith("models/"):
        n = n[len("models/") :]
    return n


# 国際NOTAM番号に典型的なトークン（例: V0333/26）を表示から取り除く
_INTERNATIONAL_NOTAM_TOKEN_RE = re.compile(r"\b[A-Z]\d{3,5}/\d{2}\b")
# 括弧付き（V0333/26）等
_INTERNATIONAL_NOTAM_IN_PARENS_RE = re.compile(
    r"[\(（]\s*[A-Z]\d{3,5}\s*/\s*\d{2}\s*[\)）]",
    re.I,
)
# 電文ヘッダー「6桁 + 空白 + 8レター」（例 180034 RJAAYNYX）。国内NOTAM番号ではない。
_NOTAM_ADDRESSEE_HEADER_RE = re.compile(
    r"^\d{6}\s+[A-Z]{8}\b",
    re.I,
)


def _is_notam_addressee_header_value(t: str) -> bool:
    u = (t or "").strip()
    return bool(u) and bool(_NOTAM_ADDRESSEE_HEADER_RE.match(u))


def strip_international_notam_tokens(text: str) -> str:
    """値の中に紛れ込んだ国際NOTAM番号形式のトークンを削除する。"""
    if not (text or "").strip():
        return (text or "").strip()
    s = _INTERNATIONAL_NOTAM_TOKEN_RE.sub("", text)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip(" ,;、").strip()


def clean_domestic_notam_number_value(nn: str) -> str:
    """notam_number 専用。国際番号表記・括弧内国際を除去し、国内表記のみ残す。"""
    t = (nn or "").strip()
    if not t:
        return ""
    if _is_notam_addressee_header_value(t):
        return ""
    t = _INTERNATIONAL_NOTAM_IN_PARENS_RE.sub("", t)
    t = strip_international_notam_tokens(t)
    t = re.sub(r"\s{2,}", " ", t).strip(" ,;、。()（）[]【】")
    collapsed = re.sub(r"\s+", "", t)
    if collapsed and _INTERNATIONAL_NOTAM_TOKEN_RE.fullmatch(collapsed):
        return ""
    if _INTERNATIONAL_NOTAM_TOKEN_RE.search(t) and len(collapsed) <= 22:
        t = _INTERNATIONAL_NOTAM_TOKEN_RE.sub("", t).strip(" ,;、。()（）[]【】")
    t = re.sub(r"\s{2,}", " ", t).strip()
    if collapsed and _INTERNATIONAL_NOTAM_TOKEN_RE.fullmatch(re.sub(r"\s+", "", t)):
        return ""
    return t


def sanitize_domestic_notam_numbers_on_items(notam_items: list[dict]) -> list[dict]:
    """各要素の notam_number から国際表記を物理的に取り除く。"""
    return [
        {**it, "notam_number": clean_domestic_notam_number_value(str(it.get("notam_number") or ""))}
        for it in notam_items
    ]


# 解析画面・テキスト出力から緯度経度らしき表記を除く（KML 用の座標抽出は原文から別処理）
_COMPACT_DMS_COORD_RE = re.compile(
    r"(?:PSN\s*:\s*)?\d{2}\s*\d{2}\s*\d+(?:\.\d+)?\s*[NS]\s*\d{3}\s*\d{2}\s*\d+(?:\.\d+)?\s*[EW]",
    re.IGNORECASE,
)
_Q_LINE_COORD_TOKEN_RE = re.compile(r"\b\d{4,5}[NS]\d{5,7}[EW]\d+\b", re.IGNORECASE)
_DECIMAL_NE_COORD_RE = re.compile(
    r"\d{1,3}(?:\.\d+)?\s*[NS]\s*,?\s*\d{1,3}(?:\.\d+)?\s*[EW]",
    re.IGNORECASE,
)


def strip_coordinate_like_from_text(text: str) -> str:
    """緯度経度の数値表記（度分秒・Q 行風・小数度など）を取り除く。"""
    if not (text or "").strip():
        return (text or "").strip()
    s = _COMPACT_DMS_COORD_RE.sub("", text)
    s = _Q_LINE_COORD_TOKEN_RE.sub("", s)
    s = _DECIMAL_NE_COORD_RE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\s*([,;、])\s*", r"\1 ", s)
    return s.strip(" ,;、").strip()


def strip_placeholder_altitude_text(text: str) -> str:
    """モデルが高度なしで入れがちな SFC - FL999 等を空にする（画面・PDF 用）。"""
    t = (text or "").strip()
    if not t:
        return ""
    norm = re.sub(r"\s+", " ", t)
    for ch in ("－", "–", "—", "―"):
        norm = norm.replace(ch, "-")
    if re.fullmatch(r"(?i)(SFC|地表|GND)\s*-\s*FL\s*999(\s.*)?", norm):
        return ""
    if re.fullmatch(r"(?i)FL\s*999(\s.*)?", norm):
        return ""
    return t


def sanitize_notam_items_altitude_placeholders(notam_items: list[dict]) -> list[dict]:
    """各 NOTAM の altitude からダミー上限表記を除去する。"""
    return [
        {**it, "altitude": strip_placeholder_altitude_text(str(it.get("altitude") or ""))}
        for it in notam_items
    ]


def should_omit_notam_display_line(text: str) -> bool:
    """該当なし系・空は箇条書きに出さない。"""
    t = (text or "").strip()
    if not t:
        return True
    if re.fullmatch(r"該当[な無无]し", t.replace(" ", "").replace("　", "")):
        return True
    collapsed = t.replace(" ", "").replace("　", "").lower()
    if collapsed in ("該当なし", "該当無し", "該当无し"):
        return True
    if collapsed in ("n/a", "none", "nil"):
        return True
    if collapsed == "na" and len(t) <= 4:
        return True
    if t in ("―", "—", "-", "…"):
        return True
    return False


_ICAO4_IN_TEXT_RE = re.compile(r"\b([A-Z]{4})\b")
_DOMESTIC_NOTAM_AIRPORT_RE = re.compile(r"\b([A-Z]{4})\s+\d{3,5}/\d{2}\b")


def infer_airport_label_for_notam_item(item: dict) -> str:
    """解析項目から主たる空港の4レターを推定（PDF の枠見出し用）。取れなければ「その他」。"""
    # 国内 RJXX 連番/年 形式は空港が明確なことが多い
    for key in ("notam_number",):
        val = unicodedata.normalize("NFKC", str(item.get(key) or ""))
        m = _DOMESTIC_NOTAM_AIRPORT_RE.search(val)
        if m:
            return m.group(1)
    # 適用区域・本文から 4 レター（FIR の RJJJ 単独より実空港を優先）
    for key in ("applicable_area", "content_type", "equipment", "other_conditions", "notam_number"):
        val = unicodedata.normalize("NFKC", str(item.get(key) or ""))
        codes = _ICAO4_IN_TEXT_RE.findall(val)
        if not codes:
            continue
        if len(codes) >= 2:
            for c in codes:
                if c != "RJJJ":
                    return c
        return codes[0]
    return "その他"


def group_notam_items_by_airport(notam_items: list[dict]) -> list[tuple[str, list[dict]]]:
    """原文中の出現順を保ち、空港ラベルの初出順でまとめる。"""
    order: list[str] = []
    buckets: dict[str, list[dict]] = {}
    for item in notam_items:
        lab = infer_airport_label_for_notam_item(item)
        if lab not in buckets:
            order.append(lab)
            buckets[lab] = []
        buckets[lab].append(item)
    return [(lb, buckets[lb]) for lb in order]


def reorder_airport_sections_for_pdf(
    sections: list[tuple[str, list[str]]],
) -> list[tuple[str, list[str]]]:
    """
    PDF 出力の空港枠の並び順を固定ルールで並べ替える。
    優先:
    - 1ページ目: RJSF（あれば）
    - 2ページ目: RJSN または RJSS（あれば）。両方あれば RJSN → RJSS
    - 最後: RJTU / RJAH（最優先で最後）。両方あれば RJTU → RJAH（RJAHが最終）
    - それ以外: 元の出現順のまま
    """
    if not sections:
        return sections
    by_label: dict[str, list[str]] = {}
    original_order: list[str] = []
    for label, blocks in sections:
        if label not in by_label:
            original_order.append(label)
        by_label[label] = blocks

    out: list[tuple[str, list[str]]] = []

    def _take(label: str) -> None:
        if label in by_label:
            out.append((label, by_label.pop(label)))

    # 強制優先
    _take("RJSF")
    _take("RJSN")
    _take("RJSS")

    # 中間（元の順序を維持）
    for lb in original_order:
        if lb in ("RJSF", "RJSN", "RJSS", "RJTU", "RJAH"):
            continue
        _take(lb)

    # 最後（最優先で末尾に回す）
    _take("RJTU")
    _take("RJAH")

    # 念のため取りこぼしがあれば最後に（通常ここには来ない）
    for lb, blocks in by_label.items():
        out.append((lb, blocks))
    return out


def format_one_notam_item_export_block(item: dict) -> str:
    """1 NOTAM の箇条書き本文（項目名なし）。表示対象行が無ければ空文字。"""
    lines: list[str] = []
    for key in NOTAM_RESULT_KEYS:
        val = item.get(key)
        text = str(val).strip() if val is not None else ""
        if key == "notam_number":
            text = clean_domestic_notam_number_value(text)
        else:
            text = strip_international_notam_tokens(text)
        text = strip_coordinate_like_from_text(text)
        if should_omit_notam_display_line(text):
            continue
        lines.append(f"- {text}")
    return "\n".join(lines).strip()


def build_notam_pdf_sections(notam_items: list[dict]) -> list[tuple[str, list[str]]]:
    """PDF 用: [(空港ラベル, [NOTAM本文ブロック...]), ...]。NOTAM単位でページ跨ぎを防ぐため分割して持つ。"""
    sections: list[tuple[str, list[str]]] = []
    for label, items in group_notam_items_by_airport(notam_items):
        blocks = [format_one_notam_item_export_block(it) for it in items]
        blocks = [b for b in blocks if b]
        if not blocks:
            continue
        sections.append((label, blocks))
    if not sections:
        return [("解析結果", ["（解析結果の表示対象がありません）"])]
    return reorder_airport_sections_for_pdf(sections)


def format_notam_items_for_export(notam_items: list[dict]) -> str:
    """画面と同じルールで解析結果を1本のテキストにする（PDF・共有用）。空港ごとに見出しを付ける。"""
    parts: list[str] = []
    for label, items in group_notam_items_by_airport(notam_items):
        blocks = [format_one_notam_item_export_block(it) for it in items]
        blocks = [b for b in blocks if b]
        if not blocks:
            continue
        inner = "\n\n".join(blocks)
        parts.append(f"■ {label}\n\n{inner}")
    out = "\n\n".join(parts).strip()
    return out if out else "（解析結果の表示対象がありません）"


def _split_text_chunks(text: str, chunk_size: int) -> list[str]:
    t = text or ""
    if not t:
        return [""]
    return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]


def _to_reportlab_flowable_text(s: str) -> str:
    """Paragraph 用に XML エスケープし、改行を <br/> にする。"""
    t = escape(s or "")
    return t.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")


_EXPORT_JP_FONT_NAME = "NotamExportJP"


def _register_reportlab_jp_font() -> str:
    """reportlab 用に日本語フォントを登録。失敗時は Helvetica。"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont

    if _EXPORT_JP_FONT_NAME in pdfmetrics.getRegisteredFontNames():
        return _EXPORT_JP_FONT_NAME

    env_font = os.environ.get("NOTAM_PDF_EXPORT_FONT", "").strip().strip('"')
    if env_font and os.path.isfile(env_font):
        try:
            pdfmetrics.registerFont(TTFont(_EXPORT_JP_FONT_NAME, env_font, subfontIndex=0))
            return _EXPORT_JP_FONT_NAME
        except Exception:
            pass

    for fname in (
        "NotoSansJP-Regular.otf",
        "NotoSansJP-Regular.ttf",
        "NotoSansCJKjp-Regular.otf",
    ):
        p = os.path.join(_APP_DIR, "fonts", fname)
        if os.path.isfile(p):
            try:
                pdfmetrics.registerFont(
                    TTFont(_EXPORT_JP_FONT_NAME, p, subfontIndex=0)
                )
                return _EXPORT_JP_FONT_NAME
            except Exception:
                continue

    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates: list[tuple[str, int]] = [
        (os.path.join(windir, "Fonts", "meiryo.ttc"), 0),
        (os.path.join(windir, "Fonts", "YuGothM.ttc"), 0),
        (os.path.join(windir, "Fonts", "msgothic.ttc"), 0),
        (os.path.join(windir, "Fonts", "meiryob.ttc"), 0),
    ]
    for path, subidx in candidates:
        if os.path.isfile(path):
            try:
                pdfmetrics.registerFont(
                    TTFont(_EXPORT_JP_FONT_NAME, path, subfontIndex=subidx)
                )
                return _EXPORT_JP_FONT_NAME
            except Exception:
                continue

    # Windows 以外・Streamlit Cloud 等: Adobe-Japan 系 CID（追加ファイル不要）
    for cid in ("HeiseiKakuGo-W5", "HeiseiMin-W3"):
        try:
            if cid not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(UnicodeCIDFont(cid))
            return cid
        except Exception:
            continue
    return "Helvetica"


def _export_pdf_chars_per_chunk(col_width_pt: float, leading: float, page_h_pt: float, margin_tb_pt: float) -> int:
    """1ページの表本文セルに収めそうな文字数のおおよその上限（はみ出し防止のため控えめ）。"""
    # タイトル・脚注・表ヘッダ・余白の分を差し引いた本文の目安高さ（pt）
    reserved = 120.0
    body_h = max(180.0, page_h_pt - margin_tb_pt - reserved)
    lines = max(10.0, body_h / leading)
    # 和文は幅あたりの字数はフォント次第なので保守的に見積もる
    chars_per_line = max(18.0, col_width_pt / 10.0)
    n = int(lines * chars_per_line * 0.72)
    # 1セルが1ページを超えないよう上限を抑える（splitInRow 併用でも安全側）
    return max(280, min(n, 680))


def _pdf_draw_page_header(
    canvas: object,
    header_title: str,
    font_name: str,
    page_w: float,
    page_h: float,
) -> None:
    """各ページ上部に文書タイトルとページ番号を描画する。"""
    from reportlab.lib import colors
    from reportlab.lib.units import mm as mm_unit

    canvas.saveState()
    try:
        canvas.setFont(font_name, 11)
    except Exception:
        canvas.setFont("Helvetica", 11)
    canvas.setFillColor(colors.HexColor("#263238"))
    try:
        pg = int(canvas.getPageNumber())
    except Exception:
        pg = 1
    txt = f"{header_title}　（ページ {pg}）"
    canvas.drawCentredString(page_w / 2.0, page_h - 9 * mm_unit, txt)
    canvas.restoreState()


def build_analysis_export_pdf(
    *,
    header_title: str,
    airport_sections: list[tuple[str, list[str]]],
) -> bytes:
    """解析結果の PDF（縦A4・空港ごとに枠で区切る）。NOTAM本文はページを跨がない。"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font = _register_reportlab_jp_font()
    page_size = A4
    page_w, page_h = page_size[0], page_size[1]
    lm, rm, tm, bm = 16 * mm, 16 * mm, 22 * mm, 14 * mm
    # Table の幅がページごとに微妙にズレるのを防ぐため、明示的に float へ落として固定
    usable_w = float(page_w - lm - rm)

    styles = getSampleStyleSheet()
    hdr_style = ParagraphStyle(
        "ExportHdr",
        parent=styles["Normal"],
        fontName=font,
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1b5e20"),
        spaceAfter=2,
    )
    body_leading = 13.0
    body_style = ParagraphStyle(
        "ExportBody",
        parent=styles["Normal"],
        fontName=font,
        fontSize=9.5,
        leading=body_leading,
        spaceAfter=0,
    )

    margin_tb_pt = float(tm + bm)
    chunk = _export_pdf_chars_per_chunk(
        float(usable_w - 6 * mm), body_leading, float(page_h), margin_tb_pt
    )
    chunk = min(chunk + 200, 1200)

    sections_in = airport_sections or [("解析結果", [""])]
    story: list = []
    def _airport_table(label: str, blocks: list[str]) -> Table:
        hdr_txt = f"■ {label}"
        ph = Paragraph(_to_reportlab_flowable_text(hdr_txt), hdr_style)

        # Table の「行」を NOTAM 単位にすることで、行ごとに NOSPLIT を指定できる
        rows: list[list[object]] = [[ph]]
        blocks_in = blocks if isinstance(blocks, list) and blocks else [""]
        for bi, block in enumerate(blocks_in):
            # 1つのNOTAMが極端に長いと1ページに収まらないため、ページ相当で分割（この場合は分割は許容）
            pieces = _split_text_chunks(block, chunk) if len(block or "") > chunk else [block]
            for pi, piece in enumerate(pieces):
                para = Paragraph(
                    _to_reportlab_flowable_text(piece) if (piece or "").strip() else " ",
                    body_style,
                )
                rows.append([para])

        tbl = Table(
            rows,
            colWidths=[usable_w],
            hAlign="CENTER",
            repeatRows=1,
            splitByRow=1,
            splitInRow=0,
        )
        style_cmds = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("BOX", (0, 0), (-1, -1), 0.85, colors.HexColor("#546e7a")),
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#c8e6c9")),
        ]
        # 本文行の背景と余白、分割禁止（NOTAM 1 ブロック = 1 行）
        if len(rows) >= 2:
            style_cmds.append(("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f1f8e9")))
            for r in range(1, len(rows)):
                # 行単位でページ分割しない（収まらなければ次ページへ送る）
                style_cmds.append(("NOSPLIT", (0, r), (0, r)))
                # NOTAM 間の余白（行の上/下）
                style_cmds.append(("TOPPADDING", (0, r), (0, r), 6))
                style_cmds.append(("BOTTOMPADDING", (0, r), (0, r), 6))
        tbl.setStyle(TableStyle(style_cmds))
        tbl.hAlign = "CENTER"
        return tbl

    for sec_i, (label, blocks) in enumerate(sections_in):
        if sec_i > 0:
            # 空港ごとに必ず改ページ（1空港=1ページ以上）
            story.append(PageBreak())
        tbl = _airport_table(label, blocks if isinstance(blocks, list) else [""])
        story.append(tbl)
        story.append(Spacer(0, 4 * mm))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=page_size,
        leftMargin=lm,
        rightMargin=rm,
        topMargin=tm,
        bottomMargin=bm,
        title="NOTAM 解析結果",
    )

    def _on_page(canv: object, doc: object) -> None:
        _pdf_draw_page_header(canv, header_title, font, page_w, page_h)

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    buf.seek(0)
    return buf.getvalue()


def parse_json_from_response(text: str) -> Optional[dict]:
    """モデル応答から JSON オブジェクトを取り出す。"""
    text = text.strip()
    # ```json ... ``` 形式への対応
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def normalize_parsed_notams(parsed: Optional[dict]) -> list[dict]:
    """Gemini の JSON を、画面表示用の NOTAM オブジェクト配列に正規化する。"""
    if not parsed or not isinstance(parsed, dict):
        return []
    items = parsed.get("notams")
    if isinstance(items, list) and items:
        return [x for x in items if isinstance(x, dict)]
    if all(k in parsed for k in NOTAM_RESULT_KEYS):
        return [parsed]
    return []


# 原文から国内NOTAM番号を拾い、モデルが「不明」にした項目を補う
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff\u2060]")


def _prepare_text_for_domestic_scan(raw_text: str) -> str:
    """PDF 抽出で混ざりがちなゼロ幅文字を除き、全角英数字等を NFKC で半角寄せする。"""
    t = (raw_text or "").strip()
    t = _ZERO_WIDTH_RE.sub("", t)
    return unicodedata.normalize("NFKC", t)


_HYPH_CLASS = r"[-－﹣−‐‑‒–—\u2212\u30fc]"
# ラベル直後にスペース・コロン無しで H26-00333 が続く表記（PDF 抽出で多い）。先に試す。
_DOM_LABELED_GLUE_CORE = (
    rf"([A-HJ-UW-Za-hj-uw-z]\d{{2}}{_HYPH_CLASS}\d{{3,8}})(?=[\s\n\r\(（]|$)"
)
# 空港4レター + 連番/年2桁（例 RJTU 0038/26）。印刷物「国内ノータム」欄付近で使われることが多い。
_DOM_LABELED_AERO_SLASH_GLUE = (
    rf"([A-Z]{{4}}\s+\d{{3,5}}/\d{{2}})(?=[\s\n\r\(（]|$)"
)
_DOMESTIC_NUMBER_LABELED_RES: tuple[re.Pattern[str], ...] = (
    re.compile(rf"国内\s*ノータム\s*{_DOM_LABELED_AERO_SLASH_GLUE}", re.I),
    re.compile(
        r"国内\s*ノータム\s*(?:[：:=\s]*)?[\r\n]+\s*([A-Z]{4}\s+\d{3,5}/\d{2})\b",
        re.I,
    ),
    re.compile(rf"国内\s*NOTAM\s*番号\s*{_DOM_LABELED_AERO_SLASH_GLUE}", re.I),
    re.compile(rf"国内NOTAM番号\s*{_DOM_LABELED_AERO_SLASH_GLUE}", re.I),
    re.compile(rf"国内\s*NOTAM\s*番号\s*{_DOM_LABELED_GLUE_CORE}", re.I),
    re.compile(rf"国内NOTAM番号\s*{_DOM_LABELED_GLUE_CORE}", re.I),
    re.compile(rf"国内\s*番号\s*{_DOM_LABELED_GLUE_CORE}", re.I),
    re.compile(rf"国内通報\s*番号\s*{_DOM_LABELED_GLUE_CORE}", re.I),
    re.compile(rf"国内通報番号\s*{_DOM_LABELED_GLUE_CORE}", re.I),
    re.compile(r"国内\s*NOTAM\s*番号\s*(?:[：:=\s]*)?[\r\n]+\s*([^\s\n\r]+)", re.I),
    re.compile(r"国内NOTAM番号\s*(?:[：:=\s]*)?[\r\n]+\s*([^\s\n\r]+)", re.I),
    re.compile(r"国内\s*番号\s*(?:[：:=\s]*)?[\r\n]+\s*([^\s\n\r]+)", re.I),
    re.compile(r"国内通報\s*番号\s*(?:[：:=\s]*)?[\r\n]+\s*([^\s\n\r]+)", re.I),
    re.compile(r"国内通報番号\s*(?:[：:=\s]*)?[\r\n]+\s*([^\s\n\r]+)", re.I),
    re.compile(r"国内\s*NOTAM\s*番号\s*[：:=\s]\s*([^\s\n\r]+)", re.I),
    re.compile(r"国内NOTAM番号\s*[：:=\s]\s*([^\s\n\r]+)", re.I),
    re.compile(r"国内\s*番号\s*[：:=\s]\s*([^\s\n\r]+)", re.I),
    re.compile(r"国内通報\s*番号\s*[：:=\s]\s*([^\s\n\r]+)", re.I),
    re.compile(r"国内\s*No\.?\s*[：:=\s]\s*([^\s\n\r]+)", re.I),
    re.compile(r"国内\s*ＮＯ\.?\s*[：:=\s]\s*([^\s\n\r]+)"),
    re.compile(r"NOTAM\s*番号\s*[（(]\s*国内\s*[）)]\s*[：:=\s]\s*([^\s\n\r]+)", re.I),
    re.compile(r"NOTAM番号\s*[（(]国内[）)]\s*[：:=\s]\s*([^\s\n\r]+)", re.I),
    re.compile(r"(?:SERIAL|シリアル)\s*No\.?\s*[：:=\s]\s*([^\s\n\r]+)", re.I),
    re.compile(r"Domestic\s+NOTAM\s+(?:No\.?|Number)\s*[：:=\s]\s*([^\s\n\r]+)", re.I),
)
# 国際形式 V0333/26 と区別しやすい「英1文字 + 2桁年 + ハイフン + 桁」
# 先頭は \b だと「国内NOTAM番号H26-…」の H 前で境界にならないため、ASCII 英数字以外の直後も許容する。
_DOMESTIC_HYPHEN_BODY_RE = re.compile(
    rf"(?<![0-9A-Za-z])([A-HJ-UW-Za-hj-uw-z]\d{{2}}{_HYPH_CLASS}\d{{3,8}})\b",
)
# 4レター空港 + 連番/年（国際の「英1文字+数字+/年」とは別系統）
_DOMESTIC_AERODROME_SLASH_SERIAL_RE = re.compile(
    rf"\b([A-Z]{{4}}\s+\d{{3,5}}/\d{{2}})\b",
    re.I,
)
# 先頭付近の「行頭〜国内ハイフン形式」だけ拾う（ラベルが取れない PDF 向け・先頭ブロック優先）
_LOOSE_HEAD_DOMESTIC_LINE_RE = re.compile(
    rf"(?m)^[ \t\u3000]*([A-HJ-UW-Za-hj-uw-z]\d{{2}}{_HYPH_CLASS}\d{{3,8}})\b",
)
# 行頭の「RJTU 0038/26」型（E) ブロック直後など）
_LOOSE_HEAD_AERODROME_SERIAL_LINE_RE = re.compile(
    r"(?m)^[ \t\u3000]*([A-Z]{4}\s+\d{3,5}/\d{2})\s*$",
    re.I,
)


def _is_domestic_aerodrome_slash_serial(t: str) -> bool:
    """国内表記の RJTU 0038/26 型（4レター + 連番 + / + 年2桁）。"""
    u = re.sub(r"\s+", " ", (t or "").strip())
    return bool(re.fullmatch(r"[A-Z]{4} \d{3,5}/\d{2}", u, re.I))


def _normalize_domestic_candidate_token(raw: str) -> str:
    t = (raw or "").strip()
    t = t.strip("()（）[]【】「」『』")
    if not t:
        return ""
    # 「RJTU 0038/26」型はスペース以降も番号の一部
    if _is_domestic_aerodrome_slash_serial(t):
        return re.sub(r"\s+", " ", t).strip()
    t = re.split(r"\s+", t, maxsplit=1)[0]
    return t.rstrip(".,;:、。）)）]}")


def _is_plausible_domestic_notam_number(t: str) -> bool:
    if not t or len(t) < 4 or len(t) > 42:
        return False
    if t in ("不明", "NONE", "N/A", "n/a", "-", "―", "…"):
        return False
    if "不明" in t.replace(" ", ""):
        return False
    if _INTERNATIONAL_NOTAM_TOKEN_RE.search(t):
        return False
    if _is_notam_addressee_header_value(t):
        return False
    if _is_domestic_aerodrome_slash_serial(t):
        return True
    if re.fullmatch(r"\d{4,10}", t):
        return False
    if re.fullmatch(r"[A-Z]{4}", t, re.I):
        return False
    return True


def _collect_domestic_notam_candidates_from_raw_text(raw_text: str) -> list[str]:
    """原文から国内NOTAM番号らしきトークンを出現順に（重複除去）で返す。"""
    text = _prepare_text_for_domestic_scan(raw_text)
    if not text:
        return []
    hits: list[tuple[int, str]] = []

    for rx in _DOMESTIC_NUMBER_LABELED_RES:
        for m in rx.finditer(text):
            cand = _normalize_domestic_candidate_token(m.group(1))
            if _is_plausible_domestic_notam_number(cand):
                hits.append((m.start(1), cand))

    for m in _DOMESTIC_HYPHEN_BODY_RE.finditer(text):
        cand = _normalize_domestic_candidate_token(m.group(1))
        if _is_plausible_domestic_notam_number(cand):
            hits.append((m.start(1), cand))

    for m in _DOMESTIC_AERODROME_SLASH_SERIAL_RE.finditer(text):
        cand = _normalize_domestic_candidate_token(m.group(1))
        if _is_plausible_domestic_notam_number(cand):
            hits.append((m.start(1), cand))

    head_lim = min(48_000, len(text))
    for m in _LOOSE_HEAD_DOMESTIC_LINE_RE.finditer(text):
        if m.start(1) >= head_lim:
            break
        cand = _normalize_domestic_candidate_token(m.group(1))
        if _is_plausible_domestic_notam_number(cand):
            hits.append((m.start(1), cand))

    for m in _LOOSE_HEAD_AERODROME_SERIAL_LINE_RE.finditer(text):
        if m.start(1) >= head_lim:
            break
        cand = _normalize_domestic_candidate_token(m.group(1))
        if _is_plausible_domestic_notam_number(cand):
            hits.append((m.start(1), cand))

    hits.sort(key=lambda x: x[0])
    ordered: list[str] = []
    seen: set[str] = set()
    for _, c in hits:
        key = c.upper().replace("－", "-").replace("﹣", "-").replace("−", "-").replace("‐", "-")
        if key in seen:
            continue
        seen.add(key)
        ordered.append(c)
    return ordered


def _notam_number_is_unknown_or_empty(val: object) -> bool:
    t = str(val or "").strip()
    if not t:
        return True
    if t == "不明":
        return True
    if t in ("不明（）", "不明()"):
        return True
    if t in ("―", "—", "–", "-", "…", "N/A", "n/a", "NONE", "none", "NULL", "null"):
        return True
    if re.fullmatch(r"-+", t):
        return True
    return False


def _notam_number_should_use_regex_augment(val: object) -> bool:
    """空・不明に加え、国際番号形式が混じっている notam_number も原文から差し替え対象にする。"""
    if _notam_number_is_unknown_or_empty(val):
        return True
    t = str(val or "").strip()
    if _is_notam_addressee_header_value(t):
        return True
    if _INTERNATIONAL_NOTAM_TOKEN_RE.search(t):
        return True
    return False


def _normalize_domestic_for_match(s: str) -> str:
    """ハイフン類・空白を除いた比較用キー（大文字化）。"""
    if not s:
        return ""
    t = str(s).strip().upper()
    for a, b in (
        ("－", "-"),
        ("﹣", "-"),
        ("−", "-"),
        ("‐", "-"),
        ("–", "-"),
        ("—", "-"),
        ("―", "-"),
        ("\u2212", "-"),
        ("\u30fc", "-"),
    ):
        t = t.replace(a, b)
    t = re.sub(r"\s+", "", t)
    return t


def _domestic_number_appears_in_raw(nn: str, raw: str) -> bool:
    """原文に国内番号として実質同一の表記が含まれるか（全角ハイフン等のゆらぎ許容）。"""
    if not nn or not raw:
        return False
    nn_nf = _prepare_text_for_domestic_scan(nn)
    raw_nf = _prepare_text_for_domestic_scan(raw)
    n = _normalize_domestic_for_match(nn_nf)
    if len(n) < 4:
        return (nn_nf in raw_nf) or (str(nn).strip() in raw)
    return n in _normalize_domestic_for_match(raw_nf)


def _first_notam_domestic_must_pin_to_head_candidate(
    nn_raw: object, c0: str, raw_s: str
) -> bool:
    """
    notams[0] の国内番号を、本文先頭の第1候補に必ず合わせるべきか。
    見落とし・幻覚・国際混入を抑え、KML の notam_index=1 と整合させる。
    """
    if not c0 or not _is_plausible_domestic_notam_number(c0):
        return False
    if _is_notam_addressee_header_value(str(nn_raw or "").strip()):
        return True
    if _notam_number_should_use_regex_augment(nn_raw):
        return True
    nn = clean_domestic_notam_number_value(str(nn_raw or ""))
    if not nn or nn == "不明":
        return True
    if not _domestic_number_appears_in_raw(nn, raw_s) and _domestic_number_appears_in_raw(
        c0, raw_s
    ):
        return True
    return False


def _leading_candidate_skip_for_known_prefix(
    notam_items: list[dict], candidates: list[str]
) -> int:
    """
    先頭から連続する「既に確定している国内番号」が candidates の先頭と一致するぶんだけ
    候補インデックスを進める（0 件目が埋まっているのに 1 件目へ先頭候補を誤割当しないため）。
    """
    k = 0
    for it in notam_items:
        if _notam_number_should_use_regex_augment(it.get("notam_number")):
            break
        if k >= len(candidates):
            break
        cur = clean_domestic_notam_number_value(str(it.get("notam_number") or ""))
        if not cur:
            break
        if _normalize_domestic_for_match(cur) == _normalize_domestic_for_match(candidates[k]):
            k += 1
        else:
            break
    return k


def augment_notam_domestic_numbers_from_raw_text(
    notam_items: list[dict], raw_text: str
) -> list[dict]:
    """
    notam_number が空・不明・国際番号混入の行に、原文から抽出した国内番号を当てはめる。
    先頭件は本文先頭の第1国内候補にピン留めし、既知の行に合わせて候補オフセットを進める。
    """
    if not notam_items or not (raw_text or "").strip():
        return notam_items
    candidates = _collect_domestic_notam_candidates_from_raw_text(raw_text)
    if not candidates:
        return notam_items
    raw_s = (raw_text or "").strip()
    out = [dict(it) for it in notam_items]
    c0 = candidates[0]
    if _first_notam_domestic_must_pin_to_head_candidate(
        out[0].get("notam_number"), c0, raw_s
    ):
        out[0]["notam_number"] = c0
    c_i = _leading_candidate_skip_for_known_prefix(out, candidates)
    unknown_idx = [
        i
        for i, it in enumerate(out)
        if _notam_number_should_use_regex_augment(it.get("notam_number"))
    ]
    if not unknown_idx:
        return out
    unknown_set = frozenset(unknown_idx)
    for i in range(len(out)):
        if i in unknown_set and c_i < len(candidates):
            out[i]["notam_number"] = candidates[c_i]
            c_i += 1
    return out


def _is_model_not_found_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "404" in msg or "not found" in msg or "is not found" in msg


def _is_gemini_429_error(exc: BaseException) -> bool:
    msg = str(exc)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return True
    return "quota" in msg.lower() and "exceed" in msg.lower()


def _retry_delay_seconds_from_gemini_error(exc: BaseException) -> float:
    """API メッセージの retry in Xs から待機秒。無ければ既定。"""
    m = re.search(r"retry in ([\d.]+)\s*s", str(exc), re.I)
    if m:
        try:
            return min(60.0, max(1.5, float(m.group(1))))
        except ValueError:
            pass
    return 12.0


GEMINI_429_USER_HINT = """
**429（クォータ・レート制限）について**

- **無料枠**では、モデルごとに **1 日あたりの `generate_content` 回数に上限**があります（メッセージの例では `gemini-2.5-flash` が **20 回/日**）。  
  このアプリは **PDF 1 件あたり メイン解析 ＋ KML 用解析 の最低 2 回** API を呼ぶため、**短時間で複数 PDF を解析するとすぐ上限に達しやすい**です。
- **対処**: [Google AI Studio の料金・プラン](https://ai.google.dev/pricing) で課金・有効化する、[利用状況](https://ai.dev/rate-limit) を確認する、**しばらく時間を空ける（翌日まで待つ）**、**一度に扱う PDF を減らす**、サイドバーの **モデル名を変える**（例: `gemini-2.0-flash` は別枠のことがある）など。
- 短時間の連続制限の場合は、**数十秒待って再試行**すると通ることがあります（アプリ側でも自動で数回待機します）。
""".strip()


_NOTAM_NUMBER_USER_REMINDER = (
    "\n\n【notam_number の再確認（最重要）】"
    "JSON の notam_number には、原文に**国内NOTAM番号として印字されている文字列だけ**を写経すること（改変・推測禁止）。"
    "国内通報の**冒頭付近**を必ず読み、「国内NOTAM番号」「国内番号」「国内通報番号」「NOTAM番号（国内）」等のラベルの直後・次行にある英数字とハイフンの組（例: H26-00333）を最優先で採用する。"
    "行頭の「6桁数字＋半角スペース＋8レター」（例: 271249 RJAAYNYX、180034 RJAAYNYX）の形は**国内NOTAM番号に使わない**。"
    "上記の欄やハイフン付き番号が読み取れるのに「不明」にすることは禁止。"
)


def build_user_prompt_for_analysis(notam_text: str, reference_text: str) -> str:
    """NOTAM 本文と、任意の前提知識テキストからユーザープロンプトを組み立てる。"""
    notam_trim = (notam_text or "").strip()[:MAX_NOTAM_INPUT_CHARS]
    ref = (reference_text or "").strip()
    if ref:
        ref_trim = ref[:MAX_REFERENCE_CHARS]
        return (
            "【前提知識（ユーザーが指定した PDF から抽出したテキスト）】\n"
            "以下を用語・手順・番号体系などの根拠として参照すること。\n"
            "---\n"
            f"{ref_trim}\n"
            "---\n\n"
            "【NOTAM 本文（解析対象。PDF から抽出）】\n"
            "上記前提知識を踏まえ、次のテキストを解析し、システム指示で定めた JSON のみで回答せよ。\n"
            "---\n"
            f"{notam_trim}\n"
            "---"
            f"{_NOTAM_NUMBER_USER_REMINDER}"
        )
    return (
        "以下は PDF から抽出したテキストです。指定の JSON 形式のみで回答してください。\n\n"
        "---\n"
        f"{notam_trim}\n"
        "---"
        f"{_NOTAM_NUMBER_USER_REMINDER}"
    )


def analyze_with_gemini(
    api_key: str,
    raw_text: str,
    model_name: str,
    system_instruction: str,
    reference_text: str = "",
) -> Tuple[Optional[dict], str, str]:
    """Gemini に渡し、(構造化結果, 生テキスト, 実際に使ったモデルID) を返す。404 時はフォールバックを試す。"""
    client = genai.Client(api_key=api_key)
    primary = normalize_model_name(model_name) or MODEL_FALLBACKS[0]
    try_order = [primary] + [m for m in MODEL_FALLBACKS if m != primary]

    user_prompt = build_user_prompt_for_analysis(raw_text, reference_text)

    last_error: Optional[BaseException] = None
    for mid in try_order:
        model_last_error: Optional[BaseException] = None
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=mid,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        temperature=0.08,
                    ),
                )
                raw = (getattr(response, "text", None) or "").strip()
                parsed = parse_json_from_response(raw)
                return parsed, raw, mid
            except Exception as e:
                model_last_error = e
                last_error = e
                if _is_model_not_found_error(e):
                    break
                if _is_gemini_429_error(e) and attempt + 1 < 5:
                    time.sleep(_retry_delay_seconds_from_gemini_error(e))
                    continue
                if _is_gemini_429_error(e):
                    break
                raise
        if model_last_error is not None and _is_gemini_429_error(model_last_error):
            continue
    assert last_error is not None
    raise last_error


_SPATIAL_EXTRACTION_SYSTEM = """
あなたは NOTAM プレーンテキストから地理情報だけを取り出す専門家です。
返答は有効な JSON オブジェクト 1 つのみ。前後に説明文や Markdown を付けない。

JSON スキーマ:
{
  "has_positions": boolean,
  "features": [
    {
      "notam_index": 1,
      "domestic_notam_number": "string（国際番号 V0333/26 形式は入れない。不明なら空）",
      "points": [ {"lat": number, "lon": number, "comment": "任意・短い日本語"} ],
      "lower_ft_amsl": number | null,
      "upper_ft_amsl": number | null,
      "notes": "string | null"
    }
  ]
}

ルール:
- has_positions: 少なくとも1点の緯度経度を WGS84 十進度で示せる場合 true。全く無い場合 false（features は [] でよい）。
- **lat は緯度（-90〜90）、lon は経度（-180〜180）**。日本周辺では lat が約20〜46、lon が約122〜154に収まる。値が逆になっていないか必ず確認すること。
- PSN の DDMMSS(.ss)N DDDMMSS(.ss)E は必ず十進度に変換して points に含める。
- 「382346N1411242E」のように **緯度・経度を連結した度分秒**（DDMMSS(.ss)N の直後に DDDMMSS(.ss)E、空白なし）で並ぶ頂点も、PSN と同様に十進度へ変換して points に含める（「BOUNDED BY …」の `-` 区切りの順で辿る）。
- **points の並び順は重要**: 同一 feature（同一 NOTAM）の points は、その NOTAM が適用される区域の**境界の頂点を、多角形の辺に沿って周る順**に並べる（反時計回りでも時計回りでもよいが、交差しないよう原文・図・Q 行の並びに合わせる）。先頭と末尾を結ぶと閉じた多角形になること。
- 同一 NOTAM に複数の離れた区域がある場合は **feature を複数**に分け、それぞれに points を置く。単一区域なら 1 NOTAM につき 1 feature。
- 同一 NOTAM に複数 PSN があり同一区域の頂点なら、points に周順で複数要素。別 NOTAM は別 feature に分ける。
- F) SFC, G) 699FT AMSL のように上下限がある場合は lower_ft_amsl / upper_ft_amsl に feet AMSL の数値（SFC は 0）。片方しか無い場合は null を使う。
- notam_index は本文中の NOTAM の並び（先頭から 1）。不明なら 1。
- domestic_notam_number は入力の対応表を優先し、原文から補ってよい。国際 NOTAM 番号は絶対に入れない。
- **notam_index=1** の feature には、原文で最初に現れる NOTAM ブロックの座標・PSN だけを入れる。入力表の 1 行目の国内番号と domestic_notam_number を必ず一致させる（後続 index との取り違えは KML を破壊する）。
""".strip()


def extract_spatial_features_gemini(
    api_key: str,
    raw_text: str,
    domestic_numbers_in_order: list[str],
    model_name: str,
) -> Tuple[Optional[dict], str, str]:
    """原文から緯度経度・高度の候補を JSON で取得する。"""
    client = genai.Client(api_key=api_key)
    primary = normalize_model_name(model_name) or MODEL_FALLBACKS[0]
    try_order = [primary] + [m for m in MODEL_FALLBACKS if m != primary]
    table = [
        {"notam_index": i + 1, "domestic_notam_number": (n or "").strip()}
        for i, n in enumerate(domestic_numbers_in_order)
    ]
    user = (
        "既に解析済みの国内NOTAM番号の対応（不明は空）:\n"
        f"{json.dumps(table, ensure_ascii=False, indent=2)}\n\n"
        "次の NOTAM 原文から、上記スキーマの JSON のみを返してください。\n"
        "**重要**: 各 feature の notam_index・domestic_notam_number は上表と厳密に一致させること。"
        "特に **notam_index=1** は原文先頭の最初の NOTAM ブロックの座標のみに対応させ、国内番号は表の 1 行目と同一文字列にする（取り違え禁止）。\n"
        "---\n"
        f"{(raw_text or '').strip()[:85000]}\n"
        "---"
    )
    last_error: Optional[BaseException] = None
    for mid in try_order:
        model_last_error: Optional[BaseException] = None
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=mid,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=_SPATIAL_EXTRACTION_SYSTEM,
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                raw = (getattr(response, "text", None) or "").strip()
                parsed = parse_json_from_response(raw)
                return parsed, raw, mid
            except Exception as e:
                model_last_error = e
                last_error = e
                if _is_model_not_found_error(e):
                    break
                if _is_gemini_429_error(e) and attempt + 1 < 5:
                    time.sleep(_retry_delay_seconds_from_gemini_error(e))
                    continue
                if _is_gemini_429_error(e):
                    break
                raise
        if model_last_error is not None and _is_gemini_429_error(model_last_error):
            continue
    assert last_error is not None
    raise last_error


_REFINE_DOMESTIC_NOTAM_NUMBER_SYSTEM = """
あなたは NOTAM 電文の**国内NOTAM番号**だけを、原文に照らして査読する専門家です。
返答は有効な JSON オブジェクト 1 つのみ。前後に説明文や Markdown を付けない。

JSON スキーマ:
{
  "corrections": [
    { "notam_index": 1, "notam_number": "string" }
  ]
}

ルール:
- 入力テーブルの **すべての notam_index** について、corrections にちょうど 1 行ずつ含める（欠け・余分を作らない）。
- notam_number には、原文に**国内NOTAM番号として印字されている文字列を一字一句写経**する。推測・補完・整形禁止。
- **国際NOTAM番号**（英1文字＋3〜5桁数字＋/＋年2桁、例 V0333/26）は **notam_number に絶対に含めない**（括弧内の国際番号も転記禁止）。
- 電文先頭の「6桁数字＋半角スペース＋8レター」（例 271249 RJAAYNYX、180034 RJAAYNYX）は**国内番号ではない**。notam_number に書いてはならない（書いた場合は無効）。
- **notam_index=1** は必ず原文先頭の最初の NOTAM ブロックの国内番号と照合する。先頭だけ「不明」にすることは後続・地図の対応を狂わせるため禁止（原文に国内欄があれば写経）。
- 原文に国内番号の所定欄が存在しない NOTAM だけ notam_number を「不明」とする。それ以外は「不明」にしない。
- 入力の「現在の国内NOTAM番号欄」が誤り・空・**国際番号形式の混入**のときは、原文の**国内所定欄の印字だけ**を返して必ず修正する。
- notam_number の文字列に **英大文字1文字＋数字＋/＋年2桁** のパターンが含まれる場合は**必ず削除し**、国内表記に置き換える（置換不能なら「不明」）。
""".strip()


def _merge_domestic_refine_corrections(
    notam_items: list[dict], by_idx: dict[int, str]
) -> list[dict]:
    """査読結果を notam_items に反映する（国際番号っぽい値は無視）。"""
    out = [dict(it) for it in notam_items]
    for i, it in enumerate(out):
        idx = i + 1
        if idx not in by_idx:
            continue
        new_n = str(by_idx[idx] or "").strip()
        old = str(it.get("notam_number") or "").strip()
        if _INTERNATIONAL_NOTAM_TOKEN_RE.search(new_n):
            continue
        if _is_notam_addressee_header_value(new_n):
            continue
        if new_n and new_n != "不明":
            out[i]["notam_number"] = new_n
        elif new_n == "不明" and _notam_number_is_unknown_or_empty(old):
            out[i]["notam_number"] = "不明"
    for j in range(len(out)):
        out[j]["notam_number"] = clean_domestic_notam_number_value(
            str(out[j].get("notam_number") or "")
        )
    return out


def refine_domestic_notam_numbers_with_gemini(
    api_key: str,
    raw_text: str,
    notam_items: list[dict],
    model_name: str,
    *,
    strict_international_reminder: bool = False,
) -> list[dict]:
    """
    メイン解析の直後に、原文だけを根拠に国内NOTAM番号を再査読する（追加の API 呼び出し）。
    失敗時は元の notam_items をそのまま返す。
    """
    if not notam_items:
        return notam_items
    client = genai.Client(api_key=api_key)
    primary = normalize_model_name(model_name) or MODEL_FALLBACKS[0]
    try_order = [primary] + [m for m in MODEL_FALLBACKS if m != primary]

    table = [
        {
            "notam_index": i + 1,
            "現在の国内NOTAM番号欄": (str(it.get("notam_number") or "").strip() or "（空または不明）"),
            "JST期間の冒頭": (str(it.get("jst_period") or "").strip()[:100]),
        }
        for i, it in enumerate(notam_items)
    ]
    strict_head = ""
    if strict_international_reminder:
        strict_head = (
            "【最終確認・絶対遵守】\n"
            "notam_number に **V0333/26 型（英大文字1文字＋数字＋/＋年2桁）** を含めた回答は**無効**です。"
            "必ず原文の **国内NOTAM番号の所定欄**（例: 国内NOTAM番号 H26-00333）に印字された表記**のみ**を返してください。"
            "国際番号は括弧内外・注釈としても**1文字も** notam_number に含めないでください。\n\n"
        )
    user = (
        strict_head
        + "次の表は NOTAM を JSON 解析した直後の状態です。**下の原文のみ**を根拠に、各行の国内NOTAM番号を査読してください。\n"
        "訂正不要なら、notam_number に現在と同じ正しい値を返してください。\n"
        "表の **すべての notam_index** について、corrections に必ず 1 行ずつ出力してください。\n\n"
        "【現在の値】\n"
        f"{json.dumps(table, ensure_ascii=False, indent=2)}\n\n"
        "【NOTAM 原文（PDF 抽出テキスト）】\n---\n"
        f"{(raw_text or '').strip()[:92000]}\n---"
    )

    last_error: Optional[BaseException] = None
    for mid in try_order:
        model_last_error: Optional[BaseException] = None
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=mid,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=_REFINE_DOMESTIC_NOTAM_NUMBER_SYSTEM,
                        response_mime_type="application/json",
                        temperature=0.02,
                    ),
                )
                raw = (getattr(response, "text", None) or "").strip()
                parsed = parse_json_from_response(raw)
                if not parsed or not isinstance(parsed, dict):
                    return notam_items
                corrections = parsed.get("corrections")
                if not isinstance(corrections, list):
                    return notam_items
                by_idx: dict[int, str] = {}
                for c in corrections:
                    if not isinstance(c, dict):
                        continue
                    try:
                        ix = int(c.get("notam_index") or 0)
                        nn = str(c.get("notam_number") or "").strip()
                        if ix >= 1:
                            by_idx[ix] = nn
                    except (TypeError, ValueError):
                        continue
                need = set(range(1, len(notam_items) + 1))
                if not need.issubset(by_idx.keys()):
                    return notam_items
                return _merge_domestic_refine_corrections(notam_items, by_idx)
            except Exception as e:
                model_last_error = e
                last_error = e
                if _is_model_not_found_error(e):
                    break
                if _is_gemini_429_error(e) and attempt + 1 < 5:
                    time.sleep(_retry_delay_seconds_from_gemini_error(e))
                    continue
                if _is_gemini_429_error(e):
                    break
                return notam_items
        if model_last_error is not None and _is_gemini_429_error(model_last_error):
            continue
    return notam_items


def generate_analysis_pdf_and_kml_bytes(
    *,
    pdf_sections: list[tuple[str, str]],
    header_title: str,
    extracted: str,
    domestic_list: list[str],
    notam_items_for_kml: list[dict],
    model: str,
    api_key: str,
) -> tuple[Optional[bytes], Optional[bytes], Optional[str]]:
    """
    1 件の NOTAM 解析結果から解析 PDF と KML を生成する。
    戻り値: (pdf_bytes, kml_bytes, pdf_error_message)。KML は座標が無い等で None になり得る。
    """
    pdf_bytes: Optional[bytes] = None
    kml_bytes: Optional[bytes] = None
    pdf_err: Optional[str] = None
    try:
        pdf_bytes = build_analysis_export_pdf(
            airport_sections=pdf_sections,
            header_title=header_title,
        )
    except Exception as ex:
        pdf_err = str(ex)
    try:
        spatial, _, _ = extract_spatial_features_gemini(
            api_key,
            extracted,
            domestic_list,
            model,
        )
        spatial = spatial or {}
        spatial = augment_spatial_json_with_psn_regex(
            spatial, extracted, domestic_list
        )
        notam_meta_by_index = [dict(it) for it in (notam_items_for_kml or []) if isinstance(it, dict)]
        kml_bytes = build_kml_bytes_from_spatial_json(
            spatial,
            document_title=header_title,
            fallback_domestic_by_index=domestic_list,
            notam_meta_by_index=notam_meta_by_index,
        )
    except Exception:
        kml_bytes = None
    return pdf_bytes, kml_bytes, pdf_err


def _clear_notam_export_session_state() -> None:
    """解析実行前に、単体・複数いずれのエクスポート用セッションも消す。"""
    for k in (
        "_export_pdf_bytes",
        "_export_pdf_filename",
        "_kml_bytes",
        "_kml_filename",
        "_export_txt_bytes",
        "_export_txt_filename",
        "_heavy_export_pending",
        "_heavy_extracted",
        "_heavy_domestic_list",
        "_heavy_base",
        "_heavy_header_title",
        "_heavy_model",
        MULTI_NOTAM_DOWNLOADS_KEY,
    ):
        st.session_state.pop(k, None)


def main() -> None:
    required_access_pw = get_notam_app_access_password()
    if required_access_pw and not st.session_state.get(_NOTAM_ACCESS_SESSION_KEY):
        st.title("NOTAM 解析（PDF）")
        st.info(
            "この URL は **招待制**です。共有された **アクセス用パスワード**（環境変数 "
            "`NOTAM_APP_PASSWORD` / secrets と同じ文字列）を入力してください。"
        )
        st.caption(
            "※ Gemini の API キーはここではなく、通過後の画面サイドバーで入力します（キーを知っている人＝使える、という運用にしたい場合は、パスワードと同じ値にする運用も可能ですが、**漏れたときの損失が大きい**ので別パスワードを推奨します）。"
        )
        with st.form("notam_access_gate"):
            gate_pw = st.text_input("アクセス用パスワード", type="password")
            if st.form_submit_button("続ける"):
                if secrets.compare_digest((gate_pw or "").strip(), required_access_pw):
                    st.session_state[_NOTAM_ACCESS_SESSION_KEY] = True
                    st.rerun()
                else:
                    st.error("パスワードが違います。")
        st.stop()

    st.title("NOTAM 解析（PDF）")
    st.caption(
        "NOTAM の PDF は **1 ファイルずつ**アップロードしてください。テキスト抽出・Gemini 解析・解析PDF／KML を生成します。"
    )

    with st.sidebar:
        st.header("設定")
        if required_access_pw:
            if st.button("アクセス終了（ログアウト）", help="このブラウザでの通過状態を消します"):
                st.session_state.pop(_NOTAM_ACCESS_SESSION_KEY, None)
                st.rerun()
        st.text_input(
            "Gemini API キー",
            type="password",
            key="gemini_api_key_input",
            help="https://aistudio.google.com/apikey から取得できます。空欄の場合は環境変数 GEMINI_API_KEY または .streamlit/secrets.toml を参照します。",
        )
        model_name = st.text_input(
            "モデル名",
            value=DEFAULT_MODEL,
            help="例: gemini-2.5-flash。古い gemini-1.5-flash は 404 になることがあります。",
        )
        st.divider()
        st.subheader("解析スタイル")
        reader_label = st.selectbox(
            "読み手",
            options=list(READER_HINTS.keys()),
            index=0,
            help="出てくる用語の難しさや、何を優先して書くかが変わります。",
        )
        length_label = st.selectbox(
            "詳しさ",
            options=list(LENGTH_HINTS.keys()),
            index=1,
            help="各項目の分量の目安です。",
        )
        extra_notes = st.text_area(
            "追加の指示（任意）",
            height=100,
            placeholder="例: Qコードがあれば必ず書く／英語の原文キーワードも併記 など",
            help="ここに書いた内容は、上のスタイルより優先して守るよう指示します。",
        )
        st.divider()
        st.markdown(
            "**ヒント**: スキャン画像のみの PDF は文字が取れず解析が難しいことがあります。"
        )

    uploaded_one = st.file_uploader(
        "NOTAM の PDF（1 ファイル）",
        type=["pdf"],
        accept_multiple_files=False,
        help="複数の PDF を解析するときは、ファイルを入れ替えて再度「解析する」を押してください。",
    )
    notam_uploads = [uploaded_one] if uploaded_one is not None else []

    disk_ref_text, disk_ref_path = load_disk_reference_text()
    with st.expander("前提知識の自動読込（毎回アップロード不要）", expanded=not bool(disk_ref_path)):
        if disk_ref_path:
            st.text(disk_ref_path)
            st.caption(
                f"抽出済み {len(disk_ref_text):,} 文字（解析では先頭 {MAX_REFERENCE_CHARS:,} 文字まで使用）。"
                "ファイルを差し替えた場合はページを再読み込みすると再取り込みされます。"
            )
        else:
            st.markdown(
                "次のどちらかで、**毎回自動**で前提知識を読み込めます（NOTAM PDF だけアップロードすればよい）。"
            )
            st.markdown(
                "1. **`knowledge`** フォルダに、使いたい PDF を **1 本以上**置く（名前は自由で `.pdf` なら可）。"
                "複数あるときは **ファイル名の A→Z 順**に連結して読み込みます。順番を決めたいときは先頭に数字を付けるとよいです（例: `01_要領.pdf`, `02_用語.pdf`）。  \n"
                "2. 単一ファイルだけ別パスにしたい場合は、環境変数 **`NOTAM_REFERENCE_PDF`** にその PDF のフルパスを設定する（このときは knowledge 内は使いません）。"
            )

    reference_upload = st.file_uploader(
        "前提知識 PDF の上書き（任意）",
        type=["pdf"],
        help="アップロードした場合のみ、その内容を固定ファイルより優先して使います。空なら上の自動読込を使用。",
    )

    # 解析前でも「前提知識プレビュー」を出せるよう、ここで一度だけ抽出して使い回す
    reference_preview_raw = ""
    if reference_upload is not None:
        reference_preview_raw = extract_text_from_pdf(reference_upload.getvalue())
    elif (disk_ref_text or "").strip():
        reference_preview_raw = disk_ref_text

    if reference_upload is not None and not (reference_preview_raw or "").strip():
        st.warning("前提知識（上書き）PDF からテキストを取得できませんでした。スキャン PDF の可能性があります。")
    elif disk_ref_path and reference_upload is None and not (disk_ref_text or "").strip():
        st.warning(
            f"前提知識ファイルは見つかりましたがテキストが空です: `{disk_ref_path}`（スキャン PDFの可能性があります）"
        )

    if (reference_preview_raw or "").strip():
        with st.expander("前提知識（抽出プレビュー）", expanded=False):
            rprev = reference_preview_raw[:6000] + ("…" if len(reference_preview_raw) > 6000 else "")
            st.text_area(
                "前提知識の先頭",
                rprev,
                height=200,
                disabled=True,
                label_visibility="collapsed",
            )
            if len(reference_preview_raw) > MAX_REFERENCE_CHARS:
                st.caption(
                    f"前提知識は先頭 **{MAX_REFERENCE_CHARS:,}** 文字のみ Gemini に渡します（全 {len(reference_preview_raw):,} 文字）。"
                )

    # ダウンロードボタンを緑に（アプリ内CSSで上書き）
    st.markdown(
        """
<style>
/* download_button 全般を緑に */
div[data-testid="stDownloadButton"] > button {
  background-color: #43a047 !important; /* 明るめの緑 */
  color: white !important;
  border: 1px solid #2e7d32 !important;
}
div[data-testid="stDownloadButton"] > button:hover {
  background-color: #2e7d32 !important;
  border-color: #1b5e20 !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    def _render_downloads(downloads_saved: object) -> None:
        if not downloads_saved:
            return
        if not isinstance(downloads_saved, list):
            return
        st.subheader("ダウンロード")
        st.caption("解析PDFとKMLを別ボタンで保存できます。")
        for i, row in enumerate(downloads_saved):
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or f"ファイル{i + 1}")
            st.markdown(f"**{label}**")
            c_pdf, c_kml = st.columns(2)
            with c_pdf:
                pb = row.get("pdf_bytes")
                if isinstance(pb, (bytes, bytearray)) and len(pb) > 0:
                    st.download_button(
                        "解析PDFをダウンロード",
                        data=pb,
                        file_name=str(row.get("pdf_filename") or "notam_解析.pdf"),
                        mime="application/pdf",
                        key=f"multi_pdf_dl_{i}",
                    )
                    st.caption("縦向き A4・解析結果のみ。")
                else:
                    st.caption("解析PDF なし")
            with c_kml:
                kb = row.get("kml_bytes")
                if isinstance(kb, (bytes, bytearray)) and len(kb) > 0:
                    st.download_button(
                        "KMLをダウンロード",
                        data=kb,
                        file_name=str(row.get("kml_filename") or "notam.kml"),
                        mime="application/vnd.google-earth.kml+xml",
                        key=f"multi_kml_dl_{i}",
                    )
                    st.caption("Google Earth 等で開けます。ピンは NOTAM ごとに 1 本（国内番号）。")
                else:
                    st.caption("KML なし（座標なし等）")

    # ダウンロード欄は「前提知識（抽出プレビュー）」の直下に固定表示
    downloads_slot = st.empty()
    with downloads_slot.container():
        _render_downloads(st.session_state.get(MULTI_NOTAM_DOWNLOADS_KEY))

    # 進捗メッセージも上に固定（spinner を下に出さない）
    progress_slot = st.empty()

    col_a, col_b = st.columns([1, 4])
    with col_a:
        run = st.button("解析する", type="primary", disabled=len(notam_uploads) == 0)
    with col_b:
        if st.button("前回の生成物をクリア（PDF/KML）", help="ダウンロード欄に古いPDFが残るときに使います。"):
            _clear_notam_export_session_state()
            st.session_state.pop(MULTI_NOTAM_DOWNLOADS_KEY, None)
            with downloads_slot.container():
                _render_downloads(st.session_state.get(MULTI_NOTAM_DOWNLOADS_KEY))

    if notam_uploads:
        f0 = notam_uploads[0]
        st.info(f"NOTAM PDF: **{f0.name}**（{f0.size:,} バイト）")
    if reference_upload is not None:
        st.caption(f"前提知識の上書き: **{reference_upload.name}**（{reference_upload.size:,} バイト）")
    elif disk_ref_path:
        st.caption("前提知識: 固定ファイルを自動使用しています。")

    if run and notam_uploads:
        api_key = get_api_key()
        if not api_key:
            st.error(
                "Gemini API キーが未設定です。サイドバーに入力するか、"
                "環境変数 `GEMINI_API_KEY` または `.streamlit/secrets.toml` に設定してください。"
            )
            st.stop()

        _clear_notam_export_session_state()

        disk_txt, disk_p = load_disk_reference_text()
        # 解析用の前提知識は上で抽出した値をそのまま使う（配置を固定するため二重描画しない）
        reference_raw = reference_preview_raw

        system_instruction = build_system_instruction(reader_label, length_label, extra_notes)
        mdl = model_name.strip() or DEFAULT_MODEL
        downloads: list[dict] = []
        last_used_model: Optional[str] = None

        for uploaded in notam_uploads:
            base = os.path.splitext(uploaded.name)[0] or "notam"
            header_title = base or "NOTAM"
            st.divider()
            st.subheader(uploaded.name)

            with st.spinner("PDF からテキストを抽出しています…"):
                extracted = extract_text_from_pdf(uploaded.getvalue())

            with st.expander(f"抽出テキスト（プレビュー）: {uploaded.name}", expanded=False):
                if not extracted:
                    st.warning("テキストがほとんど取得できませんでした。スキャン PDF の可能性があります。")
                else:
                    preview = extracted[:8000] + ("…" if len(extracted) > 8000 else "")
                    st.text_area(
                        "先頭のみ表示",
                        preview,
                        height=240,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    if len(extracted) > MAX_NOTAM_INPUT_CHARS:
                        st.caption(
                            f"NOTAM 本文は先頭 **{MAX_NOTAM_INPUT_CHARS:,}** 文字のみ Gemini に渡します（全 {len(extracted):,} 文字）。"
                        )

            empty_row = {
                "label": uploaded.name,
                "pdf_bytes": None,
                "kml_bytes": None,
                "pdf_filename": f"{base}_解析.pdf",
                "kml_filename": f"{base}.kml",
            }

            if not extracted.strip():
                st.warning("このファイルは本文が空のためスキップします。")
                downloads.append(empty_row)
                continue

            raw_response = ""
            parsed: Optional[dict] = None
            used_model = mdl
            with st.spinner("Gemini で解析しています…"):
                try:
                    parsed, raw_response, used_model = analyze_with_gemini(
                        api_key,
                        extracted,
                        mdl,
                        system_instruction,
                        (reference_raw or "").strip(),
                    )
                    last_used_model = used_model
                except Exception as e:
                    st.error(f"**{uploaded.name}** — Gemini API でエラー: {e}")
                    if _is_gemini_429_error(e):
                        st.markdown(GEMINI_429_USER_HINT)
                    else:
                        st.info(
                            "モデル名を **gemini-2.5-flash** または **gemini-2.0-flash** にし、"
                            "`python -m pip install -U google-genai` で SDK を更新してから再試行してください。"
                        )
                    downloads.append(empty_row)
                    continue

            st.subheader("解析結果")

            notam_items = normalize_parsed_notams(parsed)
            if notam_items:
                notam_items = augment_notam_domestic_numbers_from_raw_text(notam_items, extracted)
                notam_items = sanitize_domestic_notam_numbers_on_items(notam_items)
                with st.spinner("国内NOTAM番号を原文で再査読しています（第1段階）…"):
                    notam_items = refine_domestic_notam_numbers_with_gemini(
                        api_key, extracted, notam_items, mdl,
                        strict_international_reminder=False,
                    )
                notam_items = sanitize_domestic_notam_numbers_on_items(notam_items)
                notam_items = augment_notam_domestic_numbers_from_raw_text(notam_items, extracted)
                with st.spinner("国内NOTAM番号を原文で再査読しています（第2段階・国際表記禁止の最終確認）…"):
                    notam_items = refine_domestic_notam_numbers_with_gemini(
                        api_key, extracted, notam_items, mdl,
                        strict_international_reminder=True,
                    )
                notam_items = sanitize_domestic_notam_numbers_on_items(notam_items)
                notam_items = augment_notam_domestic_numbers_from_raw_text(notam_items, extracted)
                notam_items = sanitize_notam_items_altitude_placeholders(notam_items)
                # 高度の整形のあともう一度（先頭の「不明」取りこぼし対策）
                notam_items = sanitize_domestic_notam_numbers_on_items(notam_items)
                notam_items = augment_notam_domestic_numbers_from_raw_text(notam_items, extracted)
            if notam_items:
                for idx, item in enumerate(notam_items, start=1):
                    bullets: list[str] = []
                    for key in NOTAM_RESULT_KEYS:
                        val = item.get(key)
                        text = str(val).strip() if val is not None else ""
                        if key == "notam_number":
                            text = clean_domestic_notam_number_value(text)
                        else:
                            text = strip_international_notam_tokens(text)
                        text = strip_coordinate_like_from_text(text)
                        if should_omit_notam_display_line(text):
                            continue
                        bullets.append(f"- {text}")
                    if bullets:
                        st.markdown("\n".join(bullets))
                    else:
                        st.caption("表示する項目がありません（空または省略対象のみ）。")
                    if len(notam_items) > 1 and idx < len(notam_items):
                        st.divider()
                pdf_sections = build_notam_pdf_sections(notam_items)
                domestic_list = [
                    clean_domestic_notam_number_value(str(it.get("notam_number") or ""))
                    for it in notam_items
                ]
                with progress_slot.container():
                    c_msg, c_gif = st.columns([6, 1])
                    with c_msg:
                        st.info("解析PDF・KML を生成しています…")
                    with c_gif:
                        # ロード中が視覚的に分かるよう GIF を表示
                        st.image(
                            "https://media1.tenor.com/m/qYpzX7uvYFcAAAAC/pixel-popcat.gif",
                            width=72,
                        )
                try:
                    pdf_b, kml_b, pdf_err = generate_analysis_pdf_and_kml_bytes(
                        pdf_sections=pdf_sections,
                        header_title=header_title,
                        extracted=extracted,
                        domestic_list=domestic_list,
                        notam_items_for_kml=notam_items,
                        model=mdl,
                        api_key=api_key,
                    )
                finally:
                    progress_slot.empty()
                if pdf_err:
                    st.warning(f"**{uploaded.name}** — 解析PDFの生成に失敗: {pdf_err}")
                downloads.append(
                    {
                        "label": uploaded.name,
                        "pdf_bytes": pdf_b,
                        "kml_bytes": kml_b,
                        "pdf_filename": f"{base}_解析.pdf",
                        "kml_filename": f"{base}.kml",
                    }
                )
            else:
                st.warning("JSON 形式の解析結果を自動認識できませんでした。以下はモデルの生の応答です。")
                st.code(raw_response, language="text")
                downloads.append(empty_row)

        if last_used_model and normalize_model_name(model_name) != last_used_model:
            st.success(
                f"指定モデルが利用できなかったため、最後の解析では **{last_used_model}** を使用しました。"
            )

        st.session_state[MULTI_NOTAM_DOWNLOADS_KEY] = downloads
        with downloads_slot.container():
            _render_downloads(st.session_state.get(MULTI_NOTAM_DOWNLOADS_KEY))

    # 下部への再描画は行わない（上部の downloads_slot に集約）


if __name__ == "__main__":
    main()
