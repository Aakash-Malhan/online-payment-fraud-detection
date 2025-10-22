import io, os, time, traceback, uuid
import numpy as np, pandas as pd, gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    average_precision_score, roc_auc_score, precision_recall_curve
)
from sklearn.utils import shuffle

# Try XGBoost; fall back to RF if not available in the Space build
try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False
    from sklearn.ensemble import RandomForestClassifier  # fallback

RANDOM_STATE = 7

# ---------------- Synthetic generator ----------------
def _lognorm(n, mean=2000.0, sigma=1.3, rng=None):
    rng = rng or np.random.default_rng(RANDOM_STATE)
    mu = np.log(mean) - 0.5 * sigma**2
    return rng.lognormal(mean=mu, sigma=sigma, size=n)

def generate_synthetic_paysim(n_rows=120_000, seed=7, fraud_rate=0.0012):
    rng = np.random.default_rng(int(seed))
    step = rng.integers(1, 101, size=n_rows)
    types = np.array(["PAYMENT","CASH_OUT","TRANSFER","CASH_IN","DEBIT"])
    probs = np.array([0.58, 0.18, 0.15, 0.07, 0.02])
    t_series = types[rng.choice(len(types), size=n_rows, p=probs)]

    base_amt = _lognorm(n_rows, mean=2000, sigma=1.3, rng=rng)
    mult = np.ones(n_rows)
    mult[t_series == "TRANSFER"] *= rng.uniform(1.5, 3.0, (t_series == "TRANSFER").sum())
    mult[t_series == "CASH_OUT"] *= rng.uniform(1.4, 2.5, (t_series == "CASH_OUT").sum())
    mult[t_series == "CASH_IN"]  *= rng.uniform(0.5, 1.2, (t_series == "CASH_IN").sum())
    amount = (base_amt * mult).clip(1, 5_000_000)

    nameOrig = np.array([f"C{rng.integers(10**6, 10**8)}" for _ in range(n_rows)])
    nameDest = np.array([f"M{rng.integers(10**6, 10**8)}" for _ in range(n_rows)])

    oldbalanceOrg = rng.uniform(0, 6_000_000, n_rows)
    mask_low = rng.random(n_rows) < 0.15
    oldbalanceOrg[mask_low] = rng.uniform(0, 5_000, mask_low.sum())
    newbalanceOrig = (oldbalanceOrg - amount).clip(min=0)
    inconsistent = rng.random(n_rows) < 0.08
    newbalanceOrig[inconsistent] = rng.uniform(0, 6_000_000, inconsistent.sum())

    oldbalanceDest = rng.uniform(0, 6_000_000, n_rows)
    new_accounts = rng.random(n_rows) < 0.10
    oldbalanceDest[new_accounts] = 0.0
    newbalanceDest = oldbalanceDest + amount
    inconsistent_d = rng.random(n_rows) < 0.07
    newbalanceDest[inconsistent_d] = rng.uniform(0, 6_000_000, inconsistent_d.sum())

    p = np.full(n_rows, float(fraud_rate))
    high_amt = amount > 300_000
    dest_empty = oldbalanceDest == 0
    sender_drained = newbalanceOrig == 0
    risky_type = np.isin(t_series, ["TRANSFER", "CASH_OUT"])
    pattern = risky_type & (high_amt | dest_empty | sender_drained)
    p[pattern] += rng.uniform(0.03, 0.2, pattern.sum())
    p[np.isin(t_series, ["PAYMENT","DEBIT"])] *= 0.15
    p[t_series == "CASH_IN"] *= 0.25
    isFraud = (rng.random(n_rows) < p).astype(int)
    isFlaggedFraud = ((t_series == "TRANSFER") & (amount > 600_000) &
                      (rng.random(n_rows) < 0.05)).astype(int)

    df = pd.DataFrame({
        "step": step,
        "type": t_series,
        "amount": np.round(amount, 2),
        "nameOrig": nameOrig,
        "oldbalanceOrg": np.round(oldbalanceOrg, 2),
        "newbalanceOrig": np.round(newbalanceOrig, 2),
        "nameDest": nameDest,
        "oldbalanceDest": np.round(oldbalanceDest, 2),
        "newbalanceDest": np.round(newbalanceDest, 2),
        "isFraud": isFraud,
        "isFlaggedFraud": isFlaggedFraud,
    }).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

# ---------------- Preprocess / Train ----------------
def preprocess(df: pd.DataFrame):
    df = df.copy()
    req = {
        "step","type","amount","nameOrig","oldbalanceOrg","newbalanceOrig",
        "nameDest","oldbalanceDest","newbalanceDest","isFlaggedFraud"
    }
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")
    has_label = "isFraud" in df.columns

    X = pd.concat(
        [
            df[["step","amount","oldbalanceOrg","newbalanceOrig",
                "oldbalanceDest","newbalanceDest","isFlaggedFraud"]].astype(float),
            pd.get_dummies(df["type"], prefix="type", drop_first=True)
        ],
        axis=1
    ).fillna(0.0)
    y = df["isFraud"].astype(int).values if has_label else None
    return X, y, has_label, df

def train_and_eval(X, y, fast_rows=100_000, test_size=0.2):
    if fast_rows and len(X) > fast_rows:
        X, y = shuffle(X, y, random_state=RANDOM_STATE)
        X = X.iloc[:int(fast_rows)].reset_index(drop=True)
        y = y[:int(fast_rows)]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    pos = int(ytr.sum()); neg = len(ytr) - pos
    spw = float(neg / max(pos, 1))

    if XGB_OK:
        model = XGBClassifier(
            n_estimators=320, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            n_jobs=0, random_state=RANDOM_STATE,
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=spw
        )
        model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        prob = model.predict_proba(Xte)[:, 1]
    else:
        model = RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE
        )
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]

    roc = roc_auc_score(yte, prob)
    pr  = average_precision_score(yte, prob)
    prec, rec, thr = precision_recall_curve(yte, prob)
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)
    best_idx = int(np.argmax(f1))
    threshold = float(thr[best_idx]) if best_idx < len(thr) else 0.5
    ypred = (prob >= threshold).astype(int)

    import matplotlib.pyplot as plt
    # Confusion matrix
    fig_cm, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(yte, ypred)).plot(
        values_format='d', cmap="Blues", ax=ax
    )
    fig_cm.tight_layout()

    # Feature importance (top 12)
    feat_fig = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        cols = np.array(X.columns)
        idx = np.argsort(importances)[-12:]
        feat_fig, ax2 = plt.subplots(figsize=(6, 4))
        ax2.barh(range(len(idx)), importances[idx])
        ax2.set_yticks(range(len(idx)))
        ax2.set_yticklabels(cols[idx])
        ax2.set_title("Feature importance (top 12)")
        feat_fig.tight_layout()

    report = classification_report(yte, ypred, digits=4)
    summary = (
        f"Samples (train+test): {len(X)} | Positives in train: {pos} | "
        f"scale_pos_weight: {spw:.2f}\n\n"
        f"ROC-AUC: {roc:.5f}\nPR-AUC: {pr:.5f}\n"
        f"Chosen threshold (max F1): {threshold:.4f}\n\n"
        f"{report}"
    )
    return model, threshold, summary, fig_cm, feat_fig

# ---------------- Helpers ----------------
def _save_csv(df: pd.DataFrame, name_hint: str) -> str:
    """Write df to a temp file and return its path for gr.Files output."""
    fname = f"/tmp/{name_hint}_{uuid.uuid4().hex[:8]}.csv"
    df.to_csv(fname, index=False)
    return fname

# ---------------- Pipeline ----------------
def run_pipeline(data_source, file, has_labels, fast_rows, user_threshold,
                 synth_rows, synth_seed, synth_fraud_rate, debug=False):
    try:
        t0 = time.time()
        source = str(data_source)
        fast_rows = int(float(fast_rows) if fast_rows is not None else 100000)
        synth_rows = int(float(synth_rows) if synth_rows is not None else 120000)
        synth_seed = int(float(synth_seed) if synth_seed is not None else 7)
        synth_fraud_rate = float(synth_fraud_rate) if synth_fraud_rate is not None else 0.0012
        user_thr = None if user_threshold is None else float(user_threshold)

        if source == "Generate Synthetic":
            df = generate_synthetic_paysim(synth_rows, synth_seed, synth_fraud_rate)
        else:
            if file is None:
                return None, "Please upload a CSV or switch to 'Generate Synthetic'.", None, None, [], "", ""
            df = pd.read_csv(file.name if hasattr(file, "name") else file)

        if has_labels and "isFraud" not in df.columns:
            return None, "You selected 'Data includes isFraud', but the column is missing.", None, None, [], "", ""

        X, y, found_label, df_raw = preprocess(df)

        metrics_md, cm_plot, feat_plot = "", None, None
        if (has_labels and found_label) or (source == "Generate Synthetic" and "isFraud" in df.columns):
            model, best_thr, metrics_md, cm_plot, feat_plot = train_and_eval(X, df["isFraud"].values, fast_rows=fast_rows)
            probs = model.predict_proba(X)[:, 1]
            thr_used = user_thr if user_thr is not None else best_thr
            preds = (probs >= thr_used).astype(int)
            metrics_md += f"\nUser threshold: {thr_used:.2f} (model max-F1 was {best_thr:.2f})"
        else:
            probs = np.zeros(len(X)); preds = np.zeros(len(X), dtype=int)
            metrics_md = "No labels detected. Upload data with **isFraud** or use 'Generate Synthetic'."

        out = df_raw.copy()
        out["fraud_probability"] = probs
        out["fraud_pred"] = preds

        top = out.sort_values("fraud_probability", ascending=False).head(20)[
            ["step","type","amount","nameOrig","nameDest","fraud_probability","fraud_pred"]
        ]

        files_to_return = []
        files_to_return.append(_save_csv(out, "fraud_predictions"))
        if source == "Generate Synthetic":
            files_to_return.append(_save_csv(df, "synthetic_dataset"))

        elapsed = time.time() - t0
        summary_md = (
            f"**Run time:** {elapsed:.1f}s\n\n"
            f"**Key takeaways**\n"
            f"- Built-in synthetic dataset mimics PaySim patterns (rare fraud, TRANSFER/CASH_OUT risk).\n"
            f"- Uses class weighting & threshold tuning for imbalanced learning.\n"
            f"- Slide the decision threshold to trade precision vs recall.\n"
            f"- Download predictions (and synthetic data, if generated) to integrate with review queues."
        )
        # No extra debug string here; errors are caught below
        return top, metrics_md, cm_plot, feat_plot, files_to_return, summary_md, ""

    except Exception as e:
        tb = traceback.format_exc()
        return None, f"⚠️ Exception: {type(e).__name__}: {e}", None, None, [], "", tb

# ---------------- UI ----------------
with gr.Blocks(title="Online Payment Fraud Detection") as demo:
    gr.Markdown(
        "# 🛡️ Online Payment Fraud Detection\n"
        "Run a demo with a **synthetic dataset** or upload your CSV. "
        "Trains XGBoost (or RF fallback), shows metrics, a confusion matrix, and feature importance."
    )
    data_source = gr.Radio(["Generate Synthetic", "Upload CSV"], value="Generate Synthetic", label="Data source")

    with gr.Column(visible=True) as synth_box:
        synth_rows = gr.Slider(5_000, 250_000, value=120_000, step=1_000, label="Synthetic rows")
        synth_seed = gr.Number(value=7, precision=0, label="Synthetic seed")
        synth_fraud_rate = gr.Slider(0.0002, 0.01, value=0.0012, step=0.0001, label="Target fraud rate")

    with gr.Column(visible=False) as upload_box:
        file = gr.File(label="Upload PaySim-like CSV", file_types=[".csv"])

    has_labels = gr.Checkbox(label="Data includes `isFraud` label", value=True)
    fast_rows = gr.Number(label="Fast demo: max rows to train on (speed)", value=100000, precision=0)
    threshold_ui = gr.Slider(0.0, 0.99, value=0.90, step=0.01, label="Decision threshold (higher = fewer alerts)")

    run_btn = gr.Button("Run")

    top_table = gr.Dataframe(label="Top 20 most suspicious transactions", interactive=False)
    metrics = gr.Markdown()
    cm_plot = gr.Plot(label="Confusion Matrix (test split)")
    feat_plot = gr.Plot(label="Feature importance")
    dl = gr.Files(label="Download files")
    summary = gr.Markdown()
    debug_box = gr.Textbox(label="Debug log (for troubleshooting)", lines=6)

    def toggle_boxes(choice):
        return gr.update(visible=(choice == "Generate Synthetic")), gr.update(visible=(choice == "Upload CSV"))
    data_source.change(toggle_boxes, inputs=data_source, outputs=[synth_box, upload_box])

    run_btn.click(
        run_pipeline,
        inputs=[data_source, file, has_labels, fast_rows, threshold_ui, synth_rows, synth_seed, synth_fraud_rate, gr.State(False)],
        outputs=[top_table, metrics, cm_plot, feat_plot, dl, summary, debug_box]
    )

if __name__ == "__main__":
    demo.launch()
