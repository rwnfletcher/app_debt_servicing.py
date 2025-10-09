# app_debt_servicing.py
import base64
from io import BytesIO
from datetime import datetime

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---- Optional ReportLab import (PDF) ----
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

st.set_page_config(
    page_title="Debt Servicing Calculator — Bank + Seller Note (Capacity-aware)",
    page_icon="📈",
    layout="wide",
)

# ========= Defaults / Reset =========
DEFAULTS = {
    "sale_price": 5_000_000.0,
    "ebitda": 1_500_000.0,
    "op_salary": 250_000.0,
    "equity_roll_pct": 0.00,
    "deposit_pct": 0.00,
    "split_seller_pct": 0.20,                      # 20%
    "bank_structure": "IO 12m then Amortizing",
    "bank_rate": 6.00,
    "bank_term": 7,
    "seller_structure": "Amortizing (P+I)",
    "seller_rate": 8.00,
    "seller_term": 5,
    "allev_amt": 0.0,
    "allev_month": 6,
    "unsecured_multiple": 2.2,
    "ffe_val": 0.0,
    "ffe_adv_rate": 0.70,                          # 70%
    "cap_bank_to_capacity": True,
    "use_operator_salary": False,
    "theme": "Light",
}

def reset_defaults():
    st.session_state.clear()

# ========= Theme + Reset UI =========
top_controls = st.columns([1, 1, 6])
with top_controls[0]:
    theme_choice = st.radio("Theme", ["Light", "Dark"], key="theme_choice", horizontal=True)
with top_controls[1]:
    if st.button("Reset to defaults"):
        reset_defaults()
        st.rerun()

if theme_choice == "Dark":
    st.markdown(
        """
        <style>
        .stApp, .block-container { background:#0f172a !important; color:#e2e8f0 !important; }
        .stButton>button, .stDownloadButton>button { background:#1f2937; color:#e2e8f0; border:1px solid #374151; }
        </style>
        """,
        unsafe_allow_html=True
    )

# ========= Format helpers =========
def fmt_money(x): return f"${float(x or 0):,.0f}"
def fmt_money2(x): return f"${float(x or 0):,.2f}"
def fmt_pct(x):
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "—"

# ========= Parsing helpers =========
def _parse_money(s, default=0.0):
    try:
        return float(str(s).replace(",", "").strip() or default)
    except Exception:
        return default

def _parse_pct(s, default=0.0):
    s = str(s).replace(",", "").strip()
    if s.endswith("%"):
        s = s[:-1]
    try:
        v = float(s)
        return v / 100 if v > 1 else v
    except Exception:
        return default

def money_input(label, default, key):
    val = st.text_input(label, value=f"{default:,.2f}", key=key)
    return _parse_money(val, default)

def percent_input(label, default, key):
    val = st.text_input(label, value=f"{default*100:.1f}%", key=key)
    return _parse_pct(val, default)

# ========= Finance logic =========
def pmt(r, n, pv):
    if n <= 0: return 0.0
    if r == 0: return pv / n
    return (r * pv) / (1 - (1 + r) ** (-n))

def build_amort(principal, rate, years, struct, label, io_months=0, extra=None):
    """
    Returns monthly amortization DataFrame with columns:
    ['Loan','Period','Payment','Interest','Principal','Ending','Year']
    """
    extra = extra or {}
    r = rate / 100 / 12
    n = int(years * 12)
    rows = []
    bal = principal
    pay_after = None
    full = pmt(r, n, principal)

    for t in range(1, n + 1):
        in_io = (struct == "Interest-Only (Full Term)") or (struct == "IO 12m then Amortizing" and t <= io_months)
        interest = bal * r

        if in_io:
            payment = interest
            pr = 0.0
        else:
            if struct == "Amortizing (P+I)":
                payment = full
            elif struct == "IO 12m then Amortizing":
                if pay_after is None:
                    pay_after = pmt(r, n - io_months, bal)
                payment = pay_after
            else:
                payment = pmt(r, n, bal)
            pr = payment - interest

        add = float(extra.get(t, 0.0))
        if add > 0:
            pr += add
            payment += add

        if t == n:
            pr = bal
            payment = interest + pr

        bal = max(bal - pr, 0.0)
        rows.append((label, t, payment, interest, pr, bal))

    df = pd.DataFrame(rows, columns=["Loan", "Period", "Payment", "Interest", "Principal", "Ending"])
    df["Year"] = ((df["Period"] - 1) // 12) + 1
    return df

def to_annual(df):
    if df.empty:
        return df
    return df.groupby("Year", as_index=False).agg(Payments=("Payment", "sum"))

def perloan_years(df):
    if df.empty:
        return 0.0, 0.0
    y = to_annual(df)
    y1 = float(y.loc[y["Year"] == 1, "Payments"].sum())
    rest = y.loc[y["Year"] >= 2, "Payments"]
    avg = float(rest.mean()) if not rest.empty else 0.0
    return y1, avg

# ========= Sidebar Inputs =========
with st.sidebar:
    st.header("Deal Inputs")
    sale_price = money_input("Sale Price", DEFAULTS["sale_price"], "sale_price")
    ebitda_input = money_input("EBITDA (annual, before any operator salary)", DEFAULTS["ebitda"], "ebitda")

    use_op = st.checkbox("Subtract Operator Salary?", DEFAULTS["use_operator_salary"], key="use_op")
    op_salary = money_input("Operator Salary (annual)", DEFAULTS["op_salary"], "op_salary") if use_op else 0.0

    st.markdown("---")
    st.subheader("Equity / Deposit")
    equity_roll = percent_input("Equity Roll", DEFAULTS["equity_roll_pct"], "equity_roll_pct")
    deposit = percent_input("Deposit / Down Payment", DEFAULTS["deposit_pct"], "deposit_pct")

    st.markdown("---")
    st.subheader("Debt Stack Split")
    # Integer-safe slider (0..100 -> divide by 100)
    split_seller_pct = st.slider(
        "Seller Note (% of financed stack)",
        0, 100, int(DEFAULTS["split_seller_pct"] * 100), 1, key="split_seller_slider"
    ) / 100.0
    st.caption("Bank share = 1 − seller share. Applied after equity roll and deposit.")

    st.markdown("---")
    st.subheader("Bank Loan")
    bank_struct = st.selectbox("Structure", ["Amortizing (P+I)", "Interest-Only (Full Term)", "IO 12m then Amortizing"], index=2)
    bank_rate = st.number_input("Interest Rate (annual %)", 0.0, 20.0, DEFAULTS["bank_rate"], 0.25)
    bank_term = st.number_input("Term (years)", 1, 30, DEFAULTS["bank_term"], 1)

    st.subheader("Seller Note")
    sell_struct = st.selectbox("Structure", ["Amortizing (P+I)", "Interest-Only (Full Term)"], index=0)
    sell_rate = st.number_input("Interest Rate (annual %)", 0.0, 20.0, DEFAULTS["seller_rate"], 0.25)
    sell_term = st.number_input("Term (years)", 1, 30, DEFAULTS["seller_term"], 1)

    st.markdown("##### Seller: Tax Burden Alleviator")
    allev_amt = money_input("Alleviator (extra principal, A$)", DEFAULTS["allev_amt"], "allev_amt")
    allev_month = st.number_input("Month number for Alleviator (1–term months)", 1, int(sell_term * 12), DEFAULTS["allev_month"], 1)

    st.markdown("---")
    st.subheader("Bank Capacity (Guide)")
    unsecured_multiple = st.number_input("Unsecured finance vs EBITDA (× multiple)", 0.0, 10.0, DEFAULTS["unsecured_multiple"], 0.1)
    ffe_value = money_input("FFE / Equipment Value (A$)", DEFAULTS["ffe_val"], "ffe_val")
    ffe_advance = percent_input("Advance rate on FFE", DEFAULTS["ffe_adv_rate"], "ffe_adv_rate")
    cap_bank = st.checkbox("Cap bank loan to capacity & reallocate excess", DEFAULTS["cap_bank_to_capacity"])

# ========= Core Calculations =========
equity_roll_value = sale_price * equity_roll
deposit_value = sale_price * deposit
financed = max(sale_price - equity_roll_value - deposit_value, 0.0)

bank_raw = financed * (1 - split_seller_pct)
seller_raw = financed - bank_raw

unsecured_cap = ebitda_input * unsecured_multiple
secured_cap = ffe_value * ffe_advance
bank_capacity = unsecured_cap + secured_cap

if cap_bank and bank_raw > bank_capacity:
    bank_principal = bank_capacity
    seller_principal = financed - bank_principal
else:
    bank_principal = bank_raw
    seller_principal = seller_raw

# Seller extra principal map
extra_map = {int(allev_month): float(allev_amt)} if allev_amt > 0 else {}

# Build schedules
bank_df = build_amort(
    bank_principal, bank_rate, bank_term, bank_struct,
    "Bank", io_months=12 if bank_struct == "IO 12m then Amortizing" else 0
)
seller_df = build_amort(
    seller_principal, sell_rate, sell_term, sell_struct,
    "Seller", io_months=0, extra=extra_map
)

combined_df = pd.concat([bank_df, seller_df], ignore_index=True).sort_values("Period")
ann = to_annual(combined_df)
y1 = float(ann.loc[ann["Year"] == 1, "Payments"].sum()) if not ann.empty else 0.0
y2plus = float(ann.loc[ann["Year"] >= 2, "Payments"].mean()) if not ann.empty else 0.0

bank_y1, bank_y2plus = perloan_years(bank_df)
seller_y1, seller_y2plus = perloan_years(seller_df)

ebitda_adjusted = max(ebitda_input - (op_salary if use_op else 0.0), 0.0)
buffer_y1 = (y1 / ebitda_adjusted) if ebitda_adjusted > 0 else float("inf")
buffer_y2 = (y2plus / ebitda_adjusted) if ebitda_adjusted > 0 else float("inf")

# ========= Dashboard =========
st.title("📊 Debt Servicing Calculator — Bank + Seller Note")

top = st.columns(5)
top[0].metric("Sale Price", fmt_money(sale_price))
top[1].metric("Financed Amount", fmt_money(financed))
top[2].metric("Bank Principal", fmt_money(bank_principal))
top[3].metric("Seller Principal", fmt_money(seller_principal))
top[4].metric("EBITDA Used", fmt_money(ebitda_adjusted))

st.markdown("### Coverage Summary")
cov = st.columns(3)
cov[0].metric("Year 1 Repayments", fmt_money(y1))
cov[1].metric("Avg. Years 2+ Repayments", fmt_money(y2plus))
cov[2].metric("Debt as % of EBITDA", f"{buffer_y1*100:.1f}% (Y1) / {buffer_y2*100:.1f}% (Y2+)")

st.markdown("### Monthly Repayments by Loan")
m1 = st.columns(3)
m1[0].metric("Bank (Y1 Monthly)", fmt_money(bank_y1 / 12))
m1[1].metric("Seller (Y1 Monthly)", fmt_money(seller_y1 / 12))
m1[2].metric("Total (Y1 Monthly)", fmt_money(y1 / 12))

m2 = st.columns(3)
m2[0].metric("Bank (Y2+ Monthly)", fmt_money(bank_y2plus / 12))
m2[1].metric("Seller (Y2+ Monthly)", fmt_money(seller_y2plus / 12))
m2[2].metric("Total (Y2+ Monthly)", fmt_money(y2plus / 12))

st.caption("🛈 Year-1 may be lower if the bank loan uses an IO (interest-only) first year; "
           "Year-1 may be higher if a Seller ‘Tax Burden Alleviator’ lump-sum occurs in Year-1. "
           "Avg. Y2+ is steady-state after IO and can change again when a loan matures earlier than the other.")

# ========= Print & Exports =========
def build_print_html(summary_dict):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    def m(v): return fmt_money(v)
    def p(v): return fmt_pct(v)
    html = f"""
    <html><head><meta charset='utf-8'><title>Debt Servicing Summary</title>
    <style>
      body{{font-family:Arial, Helvetica, sans-serif; margin:24px; color:#111}}
      table{{width:100%; border-collapse:collapse; margin:8px 0}}
      th,td{{border:1px solid #ccc; padding:6px 8px; font-size:12px}}
      th{{background:#f6f6f6}}
      .btn{{display:inline-block; padding:8px 12px; border:1px solid #bbb; border-radius:6px; text-decoration:none; color:#111}}
    </style></head>
    <body>
      <h2>Debt Servicing Calculator — Summary</h2>
      <div>Generated {now}</div>
      <p><a class="btn" onclick="window.print()">Print / Save as PDF</a></p>

      <h3>Snapshot</h3>
      <table>
        <tr><th>Sale Price</th><td>{m(summary_dict['Sale Price'])}</td>
            <th>Financed Amount</th><td>{m(summary_dict['Financed Amount'])}</td></tr>
        <tr><th>Bank Principal</th><td>{m(summary_dict['Bank Principal (final)'])}</td>
            <th>Seller Principal</th><td>{m(summary_dict['Seller Principal (final)'])}</td></tr>
        <tr><th>EBITDA</th><td>{m(summary_dict['EBITDA (input)'])}</td>
            <th>EBITDA Used</th><td>{m(summary_dict['EBITDA Used for Servicing'])}</td></tr>
      </table>

      <h3>Coverage</h3>
      <table>
        <tr><th>Year 1 Total</th><td>{m(summary_dict['Year 1 Repayments (Total)'])}</td>
            <th>Avg. Y2+ Total</th><td>{m(summary_dict['Avg. Y2+ Repayments (Total)'])}</td></tr>
        <tr><th>Debt as % EBITDA (Y1)</th><td>{p(summary_dict['Debt as % EBITDA (Y1)'])}</td>
            <th>Debt as % EBITDA (Y2+)</th><td>{p(summary_dict['Debt as % EBITDA (Y2+)'])}</td></tr>
      </table>

      <h3>Definitions</h3>
      <ul>
        <li><b>EBITDA</b>: Earnings Before Interest, Taxes, Depreciation & Amortization.</li>
        <li><b>FFE</b>: Furniture, Fixtures & Equipment.</li>
        <li><b>P+I</b>: Principal & Interest.</li>
        <li><b>IO</b>: Interest-Only.</li>
        <li><b>Alleviator</b>: One-off Seller principal payment.</li>
      </ul>
    </body></html>
    """
    return html

summary = {
    "Sale Price": sale_price,
    "Financed Amount": financed,
    "Bank Principal (final)": bank_principal,
    "Seller Principal (final)": seller_principal,
    "EBITDA (input)": ebitda_input,
    "EBITDA Used for Servicing": ebitda_adjusted,
    "Year 1 Repayments (Total)": y1,
    "Avg. Y2+ Repayments (Total)": y2plus,
    "Debt as % EBITDA (Y1)": buffer_y1,
    "Debt as % EBITDA (Y2+)": buffer_y2,
}

# HTML print view (opens new tab)
html_str = build_print_html(summary)
html64 = base64.b64encode(html_str.encode()).decode()
st.markdown(f"<a href='data:text/html;base64,{html64}' target='_blank'>🖨️ Open Print View (HTML)</a>", unsafe_allow_html=True)

# Simple PDF (ReportLab) — optional
if REPORTLAB_AVAILABLE:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    doc.build([
        Paragraph("Debt Servicing Summary", styles["Heading1"]),
        Paragraph(f"Sale Price: {fmt_money(sale_price)}<br/>Financed: {fmt_money(financed)}", styles["Normal"]),
        Paragraph(f"EBITDA: {fmt_money(ebitda_input)}<br/>EBITDA Used: {fmt_money(ebitda_adjusted)}", styles["Normal"]),
        Paragraph(f"Year 1 Repayments: {fmt_money(y1)}<br/>Avg. Y2+ Repayments: {fmt_money(y2plus)}", styles["Normal"]),
        Paragraph(f"Debt as % EBITDA: {buffer_y1*100:.1f}% (Y1) / {buffer_y2*100:.1f}% (Y2+)", styles["Normal"]),
    ])
    buf.seek(0)
    st.download_button("📄 Download PDF", data=buf.getvalue(), file_name="debt_servicing.pdf", mime="application/pdf")
else:
    st.info("Add `reportlab` to requirements.txt to enable the PDF download. HTML Print View still works.")

# ========= Risk Hints =========
st.markdown("---")
warn = []
worst = max(y1, y2plus)
if ebitda_adjusted == 0:
    warn.append("Adjusted EBITDA is zero; no coverage.")
elif worst > ebitda_adjusted:
    warn.append("Repayments exceed adjusted EBITDA in at least one phase (negative coverage).")
elif worst > 0.7 * ebitda_adjusted:
    warn.append("Debt service >~70% of adjusted EBITDA in at least one phase (thin buffer).")
if cap_bank and bank_raw > bank_capacity:
    warn.append("Bank principal capped by capacity; excess shifted to Seller Note.")
if warn:
    st.warning(" • ".join(warn))
else:
    st.info("Coverage looks reasonable based on inputs. Consider working capital, capex, taxes, contingencies.")

# ========= SOP & Definitions =========
st.markdown("---")
st.markdown("## How to Use — SOP")
st.markdown("### Inputs")
st.markdown(
    """
- **Sale Price / EBITDA / FFE value**: Type whole dollars with commas (e.g., `5,000,000`).
- **Equity Roll / Deposit / FFE advance**: Enter as `%` or decimal (e.g., `10%` or `0.10`).
- **Debt Stack Split**: Set what % of the financed amount is **Seller Note** (bank gets the rest).
- **Bank Structure**:
  - *Amortizing (P+I)* — fixed payments across the term.
  - *Interest-Only (Full Term)* — interest only until maturity (balloon at end).
  - *IO 12m then Amortizing* — first 12 months IO, then fixed P+I.
- **Seller Alleviator**: Optional one-off extra principal in the specified month (default Month 6).
- **Bank Capacity**: Unsecured = EBITDA × multiple; Secured = FFE × advance rate. Toggle capping to limit bank loan and push overflow to Seller Note.
- **Operator Salary**: Optional deduction from EBITDA before coverage calculations.
    """
)
st.markdown("### Outputs")
st.markdown(
    """
- **Key Servicing Numbers (Blended)**: Year-1 vs Avg. Years 2+ totals and monthly equivalents, plus **Debt as % of EBITDA**.
- **Monthly by Loan**: Bank and Seller monthly figures shown **separately** (Year-1 and Avg. Y2+), plus totals.
- **Loan Snapshots**: Principals, structures, rates, terms.
- **Print / Export**: HTML Print View and (optionally) PDF download.
- **Risk Hints**: Flags thin/negative coverage or bank capacity capping.
    """
)

st.markdown("### Definitions")
st.markdown(
    """
- **EBITDA** — Earnings Before Interest, Taxes, Depreciation & Amortization.  
- **FFE** — Furniture, Fixtures & Equipment (collateral).  
- **P+I** — Principal & Interest (amortizing).  
- **IO** — Interest-Only (principal due later or at maturity).  
- **Alleviator** — Seller “Tax Burden Alleviator”: a one-off extra principal payment (you choose the month).  
- **Y1 / Y2+** — Year-1 and average of Years 2 and beyond.  
    """
)
