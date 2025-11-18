import os
import re
import math
import requests
import pandas as pd
import streamlit as st
import torch
import timm
import chromadb
from torchvision import transforms
from PIL import Image
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

#---CONFIGURATION---
EXCEL_PATH = os.environ.get("SKIN_EXCEL", "skincare.db.xlsx")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "skincare.db")
COLLECTION_NAME = "products"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SKIN_TYPES = {"oily", "normal", "dry"}
TOPK_LABELS = 3
LABEL_MIN_PROB = 0.001
OTHER_MIN_PROB = 0.5
VIT_MODEL_PATH  = r"C:\Users\SARA\Downloads\Acne and skin type.v1i.folder\best_vit_multilabel.pt"
IMG_SIZE = 224

#routine building slots
ROUTINE_SLOTS = {
    "Cleanser": ["cleanser", "wash", "face wash"],
    "Serum": ["serum", "essence", "ampoule"],
    "Moisturizer": ["moisturiser", "moisturizer", "cream", "lotion", "gel"],
    "Sunscreen": ["sunscreen", "spf", "sun cream", "sun-cream", "sun fluid"],
}

#canonical issue names
ISSUE_CANON = {
    "blackheads": ["blackheads", "comedones"],
    "whiteheads": ["whiteheads", "milia"],
    "papules": ["papules"],
    "pustules": ["pustules"],
    "nodules": ["nodules"],
    "dark spot": ["dark spot", "hyperpigmentation", "dark spots", "spots"],
}

#---ISSUE HELPERS---
def _issue_terms(conditions: list[str]) -> list[str]:
    terms = set()
    for c in conditions or []:
        c_low = c.lower().strip()
        if c_low:
            terms.add(c_low)
        for canon, syns in ISSUE_CANON.items():
            if canon == c_low:
                for s in syns:
                    terms.add(s.lower())
    return sorted(terms)

def _match_count(haystack: str, needle: str) -> int:
    try:
        return 1 if re.search(r"\b" + re.escape(needle) + r"\b", haystack.lower()) else 0
    except Exception:
        return 0

def _condition_score(meta: dict, terms: list[str]) -> int:
    sf = _lower(meta.get("suitable_for", ""))
    ft = _lower(meta.get("features", ""))
    pt = _lower(meta.get("product_type", ""))
    nm = _lower(meta.get("name", ""))
    score = 0
    for t in terms:
        score += 2 * _match_count(sf, t)
        score += 2 * _match_count(ft, t)
        score += 1 * _match_count(pt, t)
        score += 1 * _match_count(nm, t)
    return score

#---GENERIC HELPERS---
def _to_float_eur(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    s = s.replace("€", "").replace("£", "").strip().replace(",", ".")
    m = re.findall(r"[0-9.]+", s)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None

def _norm(s):
    try:
        if s is None:
            return ""
        if isinstance(s, float) and math.isnan(s):
            return ""
    except Exception:
        pass
    return str(s).strip()

def _lower(s):
    try:
        if s is None:
            return ""
        if isinstance(s, float) and math.isnan(s):
            return ""
    except Exception:
        pass
    return str(s).lower().strip()

def extract_issues_from_label(label: str) -> set[str]:
    text = label.lower()
    found = set()
    for canon, syns in ISSUE_CANON.items():
        for s in syns:
            patt = r"\b" + re.escape(s.lower()) + r"\b"
            if re.search(patt, text):
                found.add(canon)
                break
    return found

def extract_skin_from_label(label: str) -> str | None:
    text = label.lower()
    for t in SKIN_TYPES:
        if re.search(r"\b" + re.escape(t) + r"\b", text):
            return t
    return None

def score_for(m, slot, cond_terms, skin_type):
    base = _condition_score(m, cond_terms) if cond_terms else 0

    blob = (
        f"{m.get('product_type','')} "
        f"{m.get('name','')} "
        f"{m.get('features','')}"
    ).lower()
    slot_bonus = 2 if any(h in blob for h in ROUTINE_SLOTS.get(slot, [])) else 0
    skin_bonus = 0
    if skin_type:
        if re.search(r"\b" + re.escape(skin_type.lower()) + r"\b", m.get("suitable_for","").lower()):
            skin_bonus = 2

    price = float(m.get("price", 0.0) or 0.0)
    price_penalty = 0.05 * price

    return max(0.0, base + slot_bonus + skin_bonus - price_penalty)

def _strict_issue_match(meta: dict, terms: list[str]) -> bool:
    fields = [
        _lower(meta.get("suitable_for", "")),
        _lower(meta.get("features", "")),
        _lower(meta.get("product_type", "")),
        _lower(meta.get("name", "")),
    ]
    for t in terms:
        patt = r"\b" + re.escape(t) + r"\b"
        if any(re.search(patt, f) for f in fields):
            return True
    return False

#---CHROMA SETUP---
@st.cache_resource
def load_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    try:
        count = col.count()
    except Exception:
        count = len(col.get(limit=1).get("ids", []))

    if count == 0:
        bootstrap_from_excel(col)

    return col

def bootstrap_from_excel(col):
    if not os.path.exists(EXCEL_PATH):
        st.error(f"Excel not found at {EXCEL_PATH}.")
        st.stop()

    df = pd.read_excel(EXCEL_PATH)
    req = ["id", "brand", "product_name", "price", "features", "suitable_for", "product_type"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        st.error(f"Missing columns in Excel: {missing}")
        st.stop()

    df["price"] = df["price"].apply(_to_float_eur)
    df = df.drop_duplicates(subset=["brand", "product_name"]).reset_index(drop=True)

    ids, docs, metas = [], [], []
    for _, row in df.iterrows():
        pid = str(row["id"])
        brand = _norm(row["brand"])
        name = _norm(row["product_name"])
        price = row["price"] if row["price"] is not None else 0.0
        features = _norm(row["features"])
        suitable = _lower(row["suitable_for"])
        ptype = _lower(row["product_type"])

        text = (
            f"{brand} {name}. Features: {features}. "
            f"Suitable for: {suitable}. Product type: {ptype}."
        )

        ids.append(pid)
        docs.append(text)
        metas.append({
            "product_id": int(row["id"]) if pd.notna(row["id"]) else None,
            "brand": brand,
            "name": name,
            "price": float(price),
            "suitable_for": suitable,
            "features": features,
            "product_type": ptype,
        })

    BATCH = 512
    for i in range(0, len(ids), BATCH):
        col.upsert(
            ids=ids[i:i + BATCH],
            documents=docs[i:i + BATCH],
            metadatas=metas[i:i + BATCH],
        )

#---CHROMA QUERIES---
def chroma_query(col, skin_type, total_budget, topk=10,
                 extra_hint="", conditions_hint=""):
    terms = [t.strip() for t in conditions_hint.split(",") if t.strip()]
    cond_terms = _issue_terms(terms) if terms else []

    parts = []
    if skin_type:
        parts.append(f"for {skin_type} skin")
    if conditions_hint:
        parts.append(f"with issues: {conditions_hint}")
    if extra_hint:
        parts.append(extra_hint)

    query_text = " ".join(["Best skincare products"] + parts + [f"under {total_budget} euros"]).strip()

    where_doc = None
    if cond_terms:
        where_doc = {"$contains": cond_terms[0]}
    elif skin_type:
        where_doc = {"$contains": _lower(skin_type)}

    n_results = max(topk, 24)
    res = col.query(
        query_texts=[query_text],
        n_results=n_results,
        where={"price": {"$lte": float(total_budget)}},
        where_document=where_doc,
    )

    candidates = []
    if res and res.get("metadatas") and res["metadatas"]:
        candidates = list(res["metadatas"][0])

    if not candidates:
        return []

    if cond_terms:
        for m in candidates:
            m["__match_score"] = _condition_score(m, cond_terms)
        candidates.sort(key=lambda x: (-x.get("__match_score", 0), x.get("price", 0.0)))
    else:
        candidates.sort(key=lambda x: x.get("price", 0.0))

    return candidates[:topk]

#---ROUTINE BUILDER---
def build_routine(col, skin_type, total_budget, per_slot=1, conditions=None):
    conditions = conditions or []
    cond_terms = _issue_terms(conditions) if conditions else []
    budget = int(round(float(total_budget) * 100))

    slot_order = ["Cleanser", "Serum", "Moisturizer", "Sunscreen"]
    slot_cands = {s: [] for s in slot_order}

    for slot in slot_order:
        gathered = []
        TOPK_STRICT = 25 if conditions else 12
        for h in ROUTINE_SLOTS[slot]:
            extra_hint = f"focusing on {h}"
            res = chroma_query(
                col, skin_type, total_budget, topk=TOPK_STRICT,
                extra_hint=extra_hint,
                conditions_hint=", ".join(conditions))

            all_zero = (len(res) > 0 and all(m.get("__match_score", 0) == 0 for m in res))
            if not res or all_zero:
                res = chroma_query(
                    col, skin_type, total_budget, topk=TOPK_STRICT,
                    extra_hint=extra_hint
                )
            gathered.extend(res)

        if not gathered:
            gathered = chroma_query(
                col, skin_type, total_budget, topk=TOPK_STRICT,
                conditions_hint=", ".join(conditions)
            )

        seen = set()
        cleaned = []
        for m in gathered:
            pid = m.get("product_id")
            if pid in seen:
                continue
            seen.add(pid)
            price_eur = float(m.get("price", 0.0) or 0.0)
            if price_eur <= float(total_budget):
                cleaned.append(m)

        pre_strict = list(cleaned)

        if conditions:
            required_terms = _issue_terms(conditions)
            cleaned = [m for m in pre_strict if _strict_issue_match(m, required_terms)]

            if not cleaned:
                cleaned = [m for m in pre_strict if _condition_score(m, required_terms) > 0]

            if not cleaned:
                cleaned = pre_strict

        cleaned.sort(key=lambda x: float(x.get("price", 0.0) or 0.0))
        slot_cands[slot] = cleaned

    import math
    dp = [(-math.inf, None) for _ in range(budget + 1)]
    dp[budget] = (0.0, [])

    for slot in slot_order:
        cands = slot_cands[slot]
        nxt = [(-math.inf, None) for _ in range(budget + 1)]

        for b in range(budget + 1):
            if dp[b][0] > nxt[b][0]:
                nxt[b] = dp[b]

        for m in cands:
            price_cents = int(round(float(m.get("price", 0.0) or 0.0) * 100))
            util = score_for(m, slot, cond_terms, skin_type)

            for b in range(price_cents, budget + 1):
                if dp[b][0] == -math.inf:
                    continue
                nb = b - price_cents
                cand_util = dp[b][0] + util
                if cand_util > nxt[nb][0]:
                    picks = list(dp[b][1])
                    picks.append((slot, m))
                    nxt[nb] = (cand_util, picks)

        dp = nxt

    best_util, best_picks = -math.inf, []
    best_b = 0
    for b in range(budget + 1):
        if dp[b][0] > best_util:
            best_util, best_picks, best_b = dp[b][0], dp[b][1], b

    routine = {s: [] for s in slot_order}
    total_chosen = 0
    if best_picks:
        used_slot = set()
        for slot, m in best_picks:
            if slot in used_slot:
                continue
            routine[slot] = [m]
            used_slot.add(slot)
            total_chosen += 1

    return routine

#--- VIT MODEL ---
@st.cache_resource
def load_vit_and_labels():
    device = torch.device("cpu")
    ckpt = torch.load(VIT_MODEL_PATH, map_location=device)
    classes = ckpt.get("classes")
    if classes is None:
        raise RuntimeError("El checkpoint no contiene 'classes'.")

    model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=len(classes))
    state = {k.replace("module.", "").replace("model.", ""): v for k, v in ckpt["model"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("Missing:", missing, "Unexpected:", unexpected)

    model.eval()
    return model, classes

_vit_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def vit_predict_probs(pil_img: Image.Image):
    model, classes = load_vit_and_labels()
    x = _vit_tf(pil_img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy().tolist()
    return sorted(list(zip(classes, probs)), key=lambda t: t[1], reverse=True)

#---LLM MODEL---
MODEL_NAME = "gemma3:4b"

def llm_advice(prompt, num_predict=800, timeout_s=900):
    system_msg = (
        "You are an expert skincare advisor. Be precise and practical. "
        "Only reference products in the list."
    )

    url_chat = "http://localhost:11434/api/chat"
    payload_chat = {
        "model": MODEL_NAME,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.5, "num_predict": num_predict, "num_ctx": 1024},
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        r = requests.post(url_chat, json=payload_chat, timeout=timeout_s)
        if r.status_code == 404:
            url_gen = "http://localhost:11434/api/generate"
            prompt_full = f"System: {system_msg}\n\nUser: {prompt}\nAssistant:"
            payload_gen = {
                "model": MODEL_NAME,
                "prompt": prompt_full,
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0.5, "num_predict": num_predict, "num_ctx": 1024},
            }
            rg = requests.post(url_gen, json=payload_gen, timeout=timeout_s)
            rg.raise_for_status()
            data = rg.json()
            return (data.get("response") or "").strip()

        r.raise_for_status()
        data = r.json()
        return data["message"]["content"].strip()

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")

#---UI SETUP---
st.set_page_config(page_title="Skin Care Vector DB", layout="centered")
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif !important;
}
.stApp {
    background-color: #ffffff;
    color: #333;
    text-align: center;
}
h1, h2, h3 {
    color: #d63384;
    font-weight: 600;
}
.stButton>button {
    background-color: #ffb3c6;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 8px 16px;
    font-weight: 500;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #ff99b6;
}
.product-box {
    background-color: #ffd1dc;
    padding: 14px;
    border-radius: 12px;
    margin: 8px 0;
    text-align: left;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)
st.title("💄 Skin Care AI")
st.subheader("Skin detector with personalized routine")

#---DATA SOURCE PANEL---
with st.expander("Data source"):
    st.write(f"Excel path: **{EXCEL_PATH}**")
    st.write(f"Chroma path: **{CHROMA_PATH}**")
    if st.button("Rebuild Chroma from Excel"):
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)
        col = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        bootstrap_from_excel(col)
        st.success("Rebuilt vector DB ✔️")

#---IMAGE CAPTURE---
camera_image = st.camera_input("📸 Take a photo")

detected_skin_type = None
detected_issues = []
top_issue = None

if camera_image:
    img = Image.open(camera_image)
    st.image(img, caption="Your Beautiful Face", use_container_width=True)

    with st.spinner("Analyzing your skin"):
        preds = vit_predict_probs(img)

        detected_skin_type = None
        for lbl, p in preds:
            if p < LABEL_MIN_PROB:
                continue
            sk = extract_skin_from_label(lbl)
            if sk:
                detected_skin_type = sk
                break

        issue_maxprob = {}
        for lbl, p in preds:
            for iss in extract_issues_from_label(lbl):
                issue_maxprob[iss] = max(issue_maxprob.get(iss, 0.0), float(p))

        sorted_items = sorted(issue_maxprob.items(), key=lambda x: x[1], reverse=True)
        filtered = [(iss, pr) for iss, pr in sorted_items if pr >= OTHER_MIN_PROB]
        detected_issues = [iss for iss, pr in filtered][:2]

        top_issue = filtered[0][0] if filtered else None

# --- DISPLAY SKIN TYPE ---
st.subheader(f"Detected Skin Type: {detected_skin_type.capitalize() if detected_skin_type else 'Unknown'}")

# --- DISPLAY ISSUES ---
if detected_issues:
    issues_text = ", ".join([i.capitalize() for i in detected_issues])
    st.markdown("### Other Detected Conditions:")
    st.markdown(f"**{issues_text}**")

    routine_skin_type = None
    routine_conditions = detected_issues
    st.subheader("Routine based on detected skin issues")

else:
    # No issues detected
    st.info("No extra conditions detected.")

    # Routine based on skin type
    routine_skin_type = detected_skin_type
    routine_conditions = []
    st.subheader("Routine based on skin type")

#---BUDGET INPUT---
total_budget = st.slider("💰 Budget (€)", 20, 400, 120, step=1)
st.write(f"Routine target: ≤ €{total_budget}")

if detected_issues:
    routine_skin_type = None
    routine_conditions = detected_issues
    st.subheader("Routine based on detected skin issues")
else:
    routine_skin_type = detected_skin_type
    routine_conditions = []
    st.subheader("Routine based on skin type")

with st.spinner("Building your routine..."):
    col = load_chroma_collection()
    routine = build_routine(
        col,
        routine_skin_type,
        total_budget,
        per_slot=1,
        conditions=routine_conditions
    )

#---DISPLAY ROUTINE---
st.markdown("## ✨ Suggested Routine")
anything = False
for slot in ["Cleanser", "Serum", "Moisturizer", "Sunscreen"]:
    items = routine.get(slot, [])
    if items:
        anything = True
        m = items[0]
        st.markdown(f"### {slot}")
        st.markdown(f"""
        <div class='product-box'>
            <b>{m.get('brand', '')}</b> — {m.get('name', '')}<br>
            €{m.get('price', 0.0):.2f}<br>
            {m.get('features', '')}<br>
            For: {m.get('suitable_for', '')}
        </div>
        """, unsafe_allow_html=True)

if not anything:
    st.warning("No matching products found. Try increasing the budget.")

#---LLM EXPLANATION---
with st.spinner("Creating instructions..."):
    prods = []
    order = ["Cleanser", "Serum", "Moisturizer", "Sunscreen"]
    for slot in order:
        items = routine.get(slot, [])
        for m in items:
            prods.append(
                f"- [{slot}] {m.get('brand','')} — {m.get('name','')} ("
                f"€{m.get('price',0.0):.2f}); type: {m.get('product_type','')}; "
                f"suitable_for: {m.get('suitable_for','')}; "
                f"features: {m.get('features','')}"
            )

    if prods:
        products_block = "\n".join(prods)
        extra_ctx = ""
        if detected_issues:
            extra_ctx = f"\nDetected issues: {', '.join(detected_issues)}."

        prompt = (
            "You are a professional dermatologist AI. Use ONLY the products listed below.\n"
            "Start with: 'As your dermatologist, I suggest...'\n"
            "Give a short skincare routine (1–2 sentences per product) explaining use order and main benefit.\n"
            "Do NOT invent or add new products.\n\n"
            f"Skin type: {detected_skin_type or 'unknown'} | Budget: €{total_budget}\n"
            f"{extra_ctx}\n\n"
            "Products:\n"
            f"{products_block}"
        )

        try:
            text = llm_advice(prompt, num_predict=900)
            st.markdown(text)
        except Exception as e:
            st.error("LLM unavailable. Check Ollama is running.")
    else:
        st.info("No products selected, so no explanation generated.")
