"""DocAI ETL — Streamlit client application.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from docai_etl import DocumentETLPipeline, PipelineConfig, __version__
from docai_etl.llm import LLMExtractor
from docai_etl.loaders import SQLiteLoader, checks_dataframe, to_dataframe, to_json
from docai_etl.ocr import SUPPORTED_EXTENSIONS, tesseract_available

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "samples"

# Streamlit >= 1.50 uses width="stretch"; older releases use use_container_width=True
_ST_VERSION = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _ST_VERSION >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="DocAI ETL · Intelligent Document Processing", page_icon="🧾", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      .stage {border:1px solid rgba(128,128,128,.25); border-radius:10px; padding:.6rem .8rem; text-align:center;}
      .stage b {display:block; font-size:.95rem}
      .stage span {font-size:.78rem; opacity:.75}
      .pill {display:inline-block; padding:.1rem .55rem; border-radius:999px; font-size:.75rem;
             background:rgba(46,134,222,.12); margin-right:.3rem}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Sidebar — configuration
# --------------------------------------------------------------------------- #
cfg = PipelineConfig()

with st.sidebar:
    st.title("⚙️ Pipeline settings")

    st.subheader("🧠 LLM (Llama 3 8B)")
    cfg.llm.enabled = st.toggle("Use LLM for structuring", value=cfg.llm.enabled,
                                help="Off = deterministic heuristic extractor (no model needed).")
    provider_label = st.selectbox(
        "Provider",
        ["Ollama (local)", "OpenAI-compatible API (Groq, Together, vLLM…)"],
        index=0 if cfg.llm.provider == "ollama" else 1,
        disabled=not cfg.llm.enabled,
    )
    cfg.llm.provider = "ollama" if provider_label.startswith("Ollama") else "openai_compatible"
    default_url = cfg.llm.base_url if cfg.llm.provider == "ollama" else "https://api.groq.com/openai/v1"
    default_model = cfg.llm.model if cfg.llm.provider == "ollama" else "llama-3.1-8b-instant"
    cfg.llm.base_url = st.text_input("Base URL", value=default_url, disabled=not cfg.llm.enabled)
    cfg.llm.model = st.text_input("Model", value=default_model, disabled=not cfg.llm.enabled)
    if cfg.llm.provider == "openai_compatible":
        cfg.llm.api_key = st.text_input("API key", value=cfg.llm.api_key, type="password")
    if st.button("🔌 Test LLM connection", disabled=not cfg.llm.enabled, **STRETCH):
        ok, msg = LLMExtractor(cfg.llm).health_check()
        (st.success if ok else st.error)(msg)

    st.subheader("🖼️ OpenCV preprocessing")
    cfg.preprocess.denoise = st.checkbox("Denoise", value=True)
    cfg.preprocess.deskew = st.checkbox("Auto-deskew", value=True)
    cfg.preprocess.upscale = st.checkbox("Upscale small images", value=True)
    cfg.preprocess.threshold = st.checkbox("Binarise", value=True)
    cfg.preprocess.threshold_method = st.radio(
        "Threshold method", ["otsu", "adaptive"], horizontal=True,
        help="Otsu for clean scans; adaptive for phone photos with shadows.",
    )

    st.subheader("🔤 Tesseract OCR")
    cfg.ocr.lang = st.text_input("Language(s)", value=cfg.ocr.lang, help="e.g. eng, hin, eng+hin")
    psm_options = {3: "3 · Auto page segmentation", 4: "4 · Single column", 6: "6 · Uniform block", 11: "11 · Sparse text"}
    cfg.ocr.psm = st.selectbox("Page segmentation mode", list(psm_options), format_func=psm_options.get)

    st.subheader("✅ Rules")
    st.caption("Dates → **DD/MM/YYYY** · Emails → lower-case RFC format · Phones → E.164")
    order = st.radio("Ambiguous numeric dates (03/04/2024)", ["Day first (DD/MM)", "Month first (MM/DD)"])
    cfg.rules.dayfirst = order.startswith("Day")
    cfg.rules.default_country_code = st.text_input("Default phone country code", value=cfg.rules.default_country_code)

    st.divider()
    st.caption(f"DocAI ETL v{__version__}")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🧾 DocAI ETL — Intelligent Document Processing")
st.markdown(
    "Upload invoices, receipts, letters or forms (images / PDFs). The pipeline **extracts** text with "
    "OpenCV + Tesseract, **structures** it with Llama 3 8B, and **validates** every value with a rule engine "
    "before **loading** it to JSON / CSV / SQLite."
)

stages = [
    ("1 · Ingest", "PNG · JPG · TIFF · PDF"),
    ("2 · Preprocess", "OpenCV denoise · deskew · binarise"),
    ("3 · OCR", "Tesseract LSTM / PDF text layer"),
    ("4 · AI Transform", "Llama 3 8B → JSON"),
    ("5 · Rule Validation", "DD/MM/YYYY · email · phone · math"),
    ("6 · Load", "JSON · CSV · SQLite"),
]
for col, (title, sub) in zip(st.columns(len(stages)), stages):
    col.markdown(f"<div class='stage'><b>{title}</b><span>{sub}</span></div>", unsafe_allow_html=True)
st.write("")

if not tesseract_available(cfg.ocr):
    st.warning(
        "Tesseract OCR was not found. Images and scanned PDFs need it — install it "
        "(Windows: UB-Mannheim installer, macOS: `brew install tesseract`, Linux: `apt install tesseract-ocr`) "
        "or set `TESSERACT_CMD`. Digital PDFs still work.",
        icon="⚠️",
    )

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
if "results" not in st.session_state:
    st.session_state.results = []

left, right = st.columns([3, 1])
with left:
    uploads = st.file_uploader(
        "Upload documents",
        type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
    )
with right:
    st.write("")
    use_samples = st.button("📂 Try sample documents", **STRETCH)
    run = st.button("🚀 Run ETL pipeline", type="primary", **STRETCH, disabled=not uploads)

batch: list[tuple[str, bytes]] = []
if use_samples:
    batch = [(p.name, p.read_bytes()) for p in sorted(SAMPLES_DIR.iterdir()) if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not batch:
        st.error("No samples found. Run `python scripts/generate_samples.py` first.")
elif run and uploads:
    batch = [(u.name, u.getvalue()) for u in uploads]

if batch:
    pipeline = DocumentETLPipeline(cfg)
    results = []
    overall = st.progress(0.0, text="Starting…")
    for i, (name, data) in enumerate(batch):
        with st.status(f"Processing **{name}**", expanded=False) as status:
            def on_progress(msg: str, pct: float, _i=i, _name=name) -> None:
                status.update(label=f"{_name}: {msg}")
                overall.progress((_i + pct) / len(batch), text=f"{_name}: {msg}")

            try:
                res = pipeline.run(data, name, progress=on_progress)
                results.append(res)
                for w in res.warnings:
                    st.write(f"⚠️ {w}")
                status.update(label=f"✅ {name} — quality {res.quality_score}% in {res.timings['total']}s", state="complete")
            except Exception as exc:  # keep the batch going
                status.update(label=f"❌ {name}: {exc}", state="error")
    overall.empty()
    st.session_state.results = results
    if results:
        st.toast(f"Processed {len(results)} document(s)", icon="✅")

results = st.session_state.results

# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
tab_results, tab_history = st.tabs(["📊 Results", "🗄️ Database history"])

with tab_results:
    if not results:
        st.info("Upload documents or click **Try sample documents** to see the pipeline in action.")
    else:
        # ---- batch KPIs
        all_checks = [c for r in results for c in r.checks]
        fixed = sum(c.status == "fixed" for c in all_checks)
        failed = sum(c.status == "fail" for c in all_checks)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Documents", len(results))
        k2.metric("Avg quality score", f"{sum(r.quality_score for r in results) / len(results):.1f}%")
        confs = [r.ocr_confidence for r in results if r.ocr_confidence is not None]
        k3.metric("Avg OCR confidence", f"{sum(confs) / len(confs):.1f}%" if confs else "—")
        k4.metric("Values auto-fixed", fixed)
        k5.metric("Rule failures", failed)

        st.subheader("Batch summary")
        summary = to_dataframe(results)
        st.dataframe(summary, **STRETCH, hide_index=True)

        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ Batch JSON", to_json(results), "docai_batch.json", "application/json", **STRETCH)
        c2.download_button("⬇️ Batch CSV", summary.to_csv(index=False), "docai_batch.csv", "text/csv", **STRETCH)
        if c3.button("💾 Load batch into SQLite", **STRETCH):
            loader = SQLiteLoader(ROOT / cfg.sqlite_path)
            for r in results:
                loader.save(r)
            st.success(f"Saved {len(results)} document(s) to {cfg.sqlite_path}")

        st.divider()
        names = [f"{r.filename} ({r.record['document_type']})" for r in results]
        idx = st.selectbox("Inspect document", range(len(results)), format_func=lambda i: names[i])
        res = results[idx]
        rec = res.record

        st.markdown(
            f"<span class='pill'>🆔 {res.doc_id}</span><span class='pill'>📄 {len(res.pages)} page(s)</span>"
            f"<span class='pill'>🧠 {res.extraction_method}</span><span class='pill'>⏱️ {res.timings.get('total')}s</span>"
            f"<span class='pill'>✅ quality {res.quality_score}%</span>",
            unsafe_allow_html=True,
        )
        if rec.get("summary"):
            st.caption(rec["summary"])

        t_struct, t_rules, t_vision, t_text, t_raw = st.tabs(
            ["🧩 Structured data", "✅ Rule checks", "🖼️ OpenCV preprocessing", "🔤 OCR text", "🧠 LLM raw output"]
        )

        with t_struct:
            a, b = st.columns([1, 1])
            with a:
                st.markdown("**Key fields**")
                fields = rec.get("fields") or {}
                if fields:
                    st.dataframe(pd.DataFrame(fields.items(), columns=["Field", "Value"]), hide_index=True, **STRETCH)
                else:
                    st.caption("No key/value fields found.")
                st.markdown("**Dates (DD/MM/YYYY)**")
                if rec["dates"]:
                    st.dataframe(pd.DataFrame(rec["dates"])[["label", "value", "original"]], hide_index=True, **STRETCH)
                else:
                    st.caption("No dates found.")
            with b:
                st.markdown("**Emails (standard format)**")
                st.write("\n".join(f"- `{e}`" for e in rec["emails"]) or "_None found_")
                st.markdown("**Phones (E.164)**")
                st.write("\n".join(f"- `{p}`" for p in rec["phones"]) or "_None found_")
                st.markdown("**Amounts**" + (f" ({rec['currency']})" if rec.get("currency") else ""))
                if rec["amounts"]:
                    st.dataframe(pd.DataFrame(rec["amounts"]), hide_index=True, **STRETCH)
                else:
                    st.caption("No amounts found.")
            if rec["line_items"]:
                st.markdown("**Line items**")
                st.dataframe(pd.DataFrame(rec["line_items"]), hide_index=True, **STRETCH)
            st.download_button("⬇️ Download this record (JSON)", to_json(res, include_text=True),
                               f"{Path(res.filename).stem}.json", "application/json")

        with t_rules:
            df = checks_dataframe(res)
            if df.empty:
                st.caption("No rule checks were applicable.")
            else:
                counts = df["status"].value_counts()
                r1, r2, r3 = st.columns(3)
                r1.metric("✅ Passed", int(counts.get("pass", 0)))
                r2.metric("🛠️ Auto-fixed", int(counts.get("fixed", 0)))
                r3.metric("❌ Failed", int(counts.get("fail", 0)))
                icon = {"pass": "✅ pass", "fixed": "🛠️ fixed", "fail": "❌ fail"}
                df["status"] = df["status"].map(icon)
                st.dataframe(df[["status", "rule", "field", "original", "result", "message"]],
                             hide_index=True, **STRETCH)

        with t_vision:
            ocr_pages = [p for p in res.pages if p.original_image is not None]
            if not ocr_pages:
                st.info("This PDF has a digital text layer, so no image preprocessing / OCR was needed.")
            for p in ocr_pages:
                st.markdown(f"**Page {p.page_number}** · steps: `{' → '.join(p.preprocessing_steps) or 'none'}` "
                            f"· OCR confidence **{p.confidence}%** · {p.word_count} words")
                o, q = st.columns(2)
                o.image(p.original_image, caption="Original")
                q.image(p.processed_image, caption="After OpenCV preprocessing", clamp=True)

        with t_text:
            for p in res.pages:
                st.markdown(f"**Page {p.page_number}** — `{p.method}`")
                st.code(p.text or "(no text)", language="text")

        with t_raw:
            st.caption("Exactly what the extractor returned, before the rule engine normalised it.")
            st.json(res.llm_output)

with tab_history:
    db_path = ROOT / cfg.sqlite_path
    if not db_path.exists():
        st.info("Nothing loaded yet. Use **Load batch into SQLite** after processing documents.")
    else:
        loader = SQLiteLoader(db_path)
        hist = loader.history()
        st.dataframe(hist, hide_index=True, **STRETCH)
        if not hist.empty:
            pick = st.selectbox("Show extracted values for", hist["doc_id"],
                                format_func=lambda d: f"{d} — {hist.set_index('doc_id').loc[d, 'filename']}")
            st.dataframe(loader.fields(pick), hide_index=True, **STRETCH)
