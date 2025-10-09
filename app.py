# app_debt_servicing.py
import math
from io import BytesIO
import base64
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# PDF tools (pure Python)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(
    page_title="Debt Servicing Calculator — Bank + Seller Note (Capacity-aware)",
    page_icon="📈",
    layout="wide",
)

# ========= THEME + RESET =========
DEFAULTS = {
    "sale_price": 5_000_000.0,
    "ebitda": 1_500_000.0,
    "op_salary": 250_000.0,
    "equity_roll_pct": 0.00,
    "deposit_pct": 0.00,
    "split_seller_pct": 0.20,
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
    "ffe_adv_rate": 0.70,
    "cap_bank_to_capacity": True,
    "use_operator_salary": False,
    "theme": "Light",
}

def _money_fmt_str(x: float) -> str:
    return f"{x:,.2f}"

def reset_defaults():
    st.session_state["sale_price"] = _money_fmt_str(DEFAULTS["sale_price"])
    st.session_state["ebitda"] = _money_fmt_str(DEFAULTS["ebitda"])
    st.session_state["op_salary"] = _money_fmt_str(DEFAULTS["op_salary"])
    st.session_state["allev_amt"] = _money_fmt_str(DEFAULTS["allev_amt"])
    st.session_state["ffe_val"] = _money_fmt_str(DEFAULTS["ffe_val"])
    st.session_state["equity_roll_pct"] = f"{DEFAULTS['equity_roll_pct']*100:.1f}%"
    st.session_state["deposit_pct"] = f"{DEFAULTS['deposit_pct']*100:.1f}%"
    st.session_state["ffe_adv_rate"] = f"{DEFAULTS['ffe_adv_rate']*100:.1f}%"
    st.session_state["split_seller_slider"] = int(DEFAULTS["split_seller_pct"] * 100)
    st.session_state["bank_structure_sel"] = DEFAULTS["bank_structure"]
    st.session_state["bank_rate_num"] = DEFAULTS["bank_rate"]
    st.session_state["bank_term_num"] = DEFAULTS["bank_term"]
    st.session_state["seller_structure_sel"] = DEFAULTS["seller_structure"]
    st.session_state["seller_rate_num"] = DEFAULTS["seller_rate"]
    st.session_state["seller_term_num"] = DEFAULTS["seller_term"]
    st.session_state["allev_month_num"] = DEFAULTS["allev_month"]
    st.session_state["unsecured_multiple_num"] = DEFAULTS["unsecured_multiple"]
    st.session_state["cap_bank_checkbox"] = DEFAULTS["cap_bank_to_capacity"]
    st.session_state["use_op_salary_chk"] = DEFAULTS["use_operator_salary"]
    st.session_state["theme_choice"] = DEFAULTS["theme"]

if "initialized" not in st.session_state:
    reset_defaults()
    st.session_state["initialized"] = True

# Header: theme + reset
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
        .stApp, .block-container { background-color: #0f172a !important; color: #e2e8f0 !important; }
        .stMetric, .stMarkdown, .stDataFrame { color: #e2e8f0 !important; }
        .stButton>button, .stDownloadButton>button { background: #1f2937; color: #e2e8f0; border: 1px solid #374151; }
        .stRadio>div[role='radiogroup'] label { color: #e2e8f0 !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

# ========= Display helpers =========
def fmt_money(x: float) -> str:
    try:
        return f"${float(x or 0):,.0f}"
    except:
        return "$0"

def fmt_money2(x: float) -> str:
    try:
        return f"${float(x or 0):,.2f}"
    except:
        return "$0.00"

def fmt_pct(x: float) -> str:
    try:
        return f"{float(x):.1%}"
    except:
        return "—"

# ========= Inputs (money with commas; percent text) =========
def _parse_money_str(s: str, default: float = 0.0) -> float:
    if s is None: return float(default)
    try:
        s = s.replace(",", "").strip()
        if s == "": return float(default)
        return float(s)
    except:
        return float(default)

def money_input(label: str, default: float, key: str, help: str | None = None) -> float:
    val_str = st.text_input(label, value=st.session_state.get(key, _money_fmt_str(default)), key=key, help=help)
    return _parse_money_str(val_str, default)

def _parse_percent_str(s: str, default: float = 0.0) -> float:
    if s is None: return float(default)
    s = s.strip().replace(",", "")
    if s.endswith("%"):
        s = s[:-1].strip()
        try: return float(s)/100.0
        except: return float(default)
    try:
        v = float(s)
        return v/100.0 if v > 1 else v
    except:
        return float(default)

def percent_input(label: str, default_fraction: float, key: str, help: str | None = None) -> float:
    default_str = st.session_state.get(key, f"{default_fraction*100:.1f}%")
    s = st.text_input(label, value=default_str, key=key, help=help)
    return _parse_percent_str(s, default_fraction)

# ========= Finance helpers =========
def pmt(rate_per_period: float, n_periods: int, present_value: float) -> float:
    if n_periods <= 0: return 0.0
    if rate_per_period == 0: return present_value / n_periods
    return (rate_per_period * present_value) / (1 - (1 + rate_per_period) ** (-n_periods))

def build_amortization_schedule(
    principal: float,
    annual_rate: float,
    term_years: int,
    structure: str,
    periods_per_year: int = 12,
    loan_label: str = "Loan",
    io_months: int = 0,
    extra_principal_map: dict | None = None,
):
    cols = ["Loan","Period","Payment","Interest","Principal","Ending Balance","Year","Month","Quarter","Cum Interest","Cum Principal"]
    if principal <= 0 or term_years <= 0:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=["Loan","Year","Payments","Interest","Principal","Ending Balance"])

    r = (annual_rate / 100.0) / periods_per_year
    n = int(term_years * periods_per_year)
    extra_principal_map = extra_principal_map or {}

    rows = []
    balance = principal
    amort_payment_after_io = None
    full_term_amort_payment = pmt(r, n, principal) if r != 0 else (principal / n)

    for t in range(1, n + 1):
        in_io_phase = (structure == "Interest-Only (Full Term)") or (structure == "IO 12m then Amortizing" and t <= io_months)
        interest = balance * r

        if in_io_phase:
            payment = interest
            principal_component = 0.0
        else:
            if structure == "Amortizing (P+I)":
                payment = full_term_amort_payment
            elif structure == "IO 12m then Amortizing":
                if amort_payment_after_io is None:
                    remaining = n - io_months
                    amort_payment_after_io = pmt(r, remaining, balance) if r != 0 else balance / max(remaining, 1)
                payment = amort_payment_after_io
            else:
                payment = pmt(r, n, balance) if r != 0 else balance / max(n, 1)
            principal_component = payment - interest

        extra = float(extra_principal_map.get(t, 0.0))
        if extra > 0:
            principal_component += extra
            payment += extra

        if t == n and principal_component > 0 and principal_component < balance + 1e-6:
            principal_component = balance
            payment = interest + principal_component

        balance = max(balance - principal_component, 0.0)
        rows.append((loan_label, t, payment, interest, principal_component, balance))

    df = pd.DataFrame(rows, columns=["Loan","Period","Payment","Interest","Principal","Ending Balance"])
    df["Year"] = ((df["Period"] - 1) // periods_per_year) + 1
    df["Month"] = ((df["Period"] - 1) % periods_per_year) + 1
    df["Quarter"] = ((df["Month"] - 1) // 3) + 1
    df["Cum Interest"] = df["Interest"].cumsum()
    df["Cum Principal"] = df["Principal"].cumsum()

    yearly = df.groupby(["Loan","Year"], as_index=False).agg(
        Payments=("Payment","sum"),
        Interest=("Interest","sum"),
        Principal=("Principal","sum")
    )
    yearly["Ending Balance"] = df.groupby(["Year"])["Ending Balance"].last().values
    return df, yearly

def pad_and_sum_monthly(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    if df_a.empty and df_b.empty: return pd.DataFrame()
    frames = []
    for df in [df_a, df_b]:
        if df.empty: continue
        frames.append(df[["Period","Payment","Interest","Principal","Ending Balance","Year","Month","Quarter"]].copy())
    max_period = 0
    for df in frames: max_period = max(max_period, int(df["Period"].max()))
    agg = pd.DataFrame({"Period": range(1, max_period + 1)})
    for col in ["Payment","Interest","Principal","Ending Balance"]: agg[col] = 0.0
    for df in frames:
        agg = agg.merge(df[["Period","Payment","Interest","Principal","Ending Balance","Year","Month","Quarter"]],
                        on="Period", how="left", suffixes=("","_x"))
        for col in ["Payment","Interest","Principal","Ending Balance"]:
            agg[col] = agg[col].fillna(0) + agg[f"{col}_x"].fillna(0)
            agg.drop(columns=[f"{col}_x"], inplace=True)
        for cal in ["Year","Month","Quarter"]:
            agg[cal] = agg[cal].fillna(df[cal])
    agg["Cum Interest"] = agg["Interest"].cumsum()
    agg["Cum Principal"] = agg["Principal"].cumsum()
    return agg

def to_quarterly(df_monthly: pd.DataFrame) -> pd.DataFrame:
    if df_monthly.empty: return df_monthly
    q = df_monthly.groupby(["Year","Quarter"], as_index=False).agg(
        Payments=("Payment","sum"), Interest=("Interest","sum"), Principal=("Principal","sum"))
    end_bal = df_monthly.groupby(["Year","Quarter"])["Ending Balance"].last().reset_index(name="Ending Balance")
    q = pd.merge(q, end_bal, on=["Year","Quarter"], how="left")
    q["Label"] = q.apply(lambda r: f"Y{int(r['Year'])} Q{int(r['Quarter'])}", axis=1)
    return q

def to_annual(df_monthly: pd.DataFrame) -> pd.DataFrame:
    if df_monthly.empty: return df_monthly
    y = df_monthly.groupby("Year", as_index=False).agg(
        Payments=("Payment","sum"), Interest=("Interest","sum"), Principal=("Principal","sum"))
    y["Ending Balance"] = df_monthly.groupby("Year")["Ending Balance"].last().values
    y["Label"] = y["Year"].apply(lambda x: f"Year {int(x)}")
    return y

def build_view_table(view: str, monthly_df: pd.DataFrame):
    if monthly_df is None or monthly_df.empty: return monthly_df, None, None, None
    if view == "Monthly":
        m = monthly_df.copy()
        m["Label"] = m.apply(lambda r: f"Y{int(r['Year'])} M{int(r['Month'])}", axis=1)
        cols = ["Period","Year","Month","Label","Payment","Interest","Principal","Ending Balance","Cum Interest","Cum Principal"]
        return m[cols], "Label", "Interest", "Principal"
    elif view == "Quarterly":
        q = to_quarterly(monthly_df)
        return q, "Label", "Interest", "Principal"
    else:
        a = to_annual(monthly_df)
        return a, "Label", "Interest", "Principal"

def make_chart(df: pd.DataFrame, label_col: str, interest_col: str, principal_col: str, title: str):
    if df is None or df.empty or label_col is None: return None
    x = df[label_col].astype(str).tolist()
    i = df[interest_col].values
    p = df[principal_col].values
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, i, label="Interest")
    ax.plot(x, p, label="Principal")
    ax.set_title(title)
    ax.set_xlabel("Period")
    ax.set_ylabel("Amount")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig

# ========= Sidebar Inputs =========
with st.sidebar:
    st.header("Deal Inputs")

    sale_price = money_input("Sale Price", DEFAULTS["sale_price"], key="sale_price")
    ebitda_input = money_input("EBITDA (annual, before any operator salary)", DEFAULTS["ebitda"], key="ebitda")

    use_operator_salary = st.checkbox("Subtract an Operator Salary before servicing?", value=DEFAULTS["use_operator_salary"], key="use_op_salary_chk")
    operator_salary = 0.0
    if use_operator_salary:
        operator_salary = money_input("Operator Salary (annual)", DEFAULTS["op_salary"], key="op_salary")

    st.markdown("---")
    st.subheader("Equity / Deposit")
    equity_roll_pct = percent_input("Equity Roll", DEFAULTS["equity_roll_pct"], key="equity_roll_pct")
    deposit_pct = percent_input("Deposit / Down Payment", DEFAULTS["deposit_pct"], key="deposit_pct")

    st.markdown("---")
    st.subheader("Debt Stack Split")
    split_seller_pct = st.slider("Seller Note (% of financed stack)", 0.0, 100.0, int(DEFAULTS["split_seller_pct"]*100), 1, key="split_seller_slider") / 100.0
    st.caption("Bank share = 1 − seller share. Applied to the financed amount after equity roll and deposit.")

    st.markdown("---")
    st.subheader("Bank Loan")
    bank_structure = st.selectbox(
        "Structure",
        ["Amortizing (P+I)", "Interest-Only (Full Term)", "IO 12m then Amortizing"],
        index=["Amortizing (P+I)", "Interest-Only (Full Term)", "IO 12m then Amortizing"].index(DEFAULTS["bank_structure"]),
        key="bank_structure_sel"
    )
    bank_rate = st.number_input("Interest Rate (annual %)", min_value=0.0, value=float(DEFAULTS["bank_rate"]), step=0.25, format="%.2f", key="bank_rate_num")
    bank_term = st.number_input("Term (years)", min_value=1, value=int(DEFAULTS["bank_term"]), step=1, key="bank_term_num")

    st.subheader("Seller Note")
    seller_structure = st.selectbox(
        "Structure ",
        ["Amortizing (P+I)", "Interest-Only (Full Term)"],
        index=["Amortizing (P+I)", "Interest-Only (Full Term)"].index(DEFAULTS["seller_structure"]),
        key="seller_structure_sel"
    )
    seller_rate = st.number_input("Interest Rate (annual %)", min_value=0.0, value=float(DEFAULTS["seller_rate"]), step=0.25, format="%.2f", key="seller_rate_num")
    seller_term = st.number_input("Term (years)", min_value=1, value=int(DEFAULTS["seller_term"]), step=1, key="seller_term_num")

    st.markdown("##### Seller: Tax Burden Alleviator")
    alleviator_amount = money_input("Tax Burden Alleviator (extra principal, A$)", DEFAULTS["allev_amt"], key="allev_amt")
    alleviator_month = st.number_input(
        "Month number for Alleviator (1–term months)",
        min_value=1, max_value=max(1, int(seller_term) * 12), value=min(int(DEFAULTS["allev_month"]), int(seller_term) * 12),
        step=1, key="allev_month_num"
    )

    st.markdown("---")
    st.subheader("Bank Capacity (Guide)")
    unsecured_multiple = st.number_input("Unsecured finance vs EBITDA (× multiple)", min_value=0.0, value=float(DEFAULTS["unsecured_multiple"]), step=0.1, format="%.2f", key="unsecured_multiple_num")
    ffe_value = money_input("FFE / Equipment Value (A$)", DEFAULTS["ffe_val"], key="ffe_val")
    ffe_advance_rate = percent_input("Advance rate on FFE", DEFAULTS["ffe_adv_rate"], key="ffe_adv_rate")
    cap_bank_to_capacity = st.checkbox("Cap bank loan to capacity & reallocate excess to Seller Note", value=DEFAULTS["cap_bank_to_capacity"], key="cap_bank_checkbox")

# ========= Core Calculations =========
equity_roll_value = sale_price * equity_roll_pct
deposit_value = sale_price * deposit_pct
finance_needed = max(sale_price - equity_roll_value - deposit_value, 0.0)

bank_principal_raw = finance_needed * (1 - split_seller_pct)
seller_principal_raw = finance_needed - bank_principal_raw

unsecured_capacity = ebitda_input * unsecured_multiple
secured_capacity = ffe_value * ffe_advance_rate
bank_capacity_total = unsecured_capacity + secured_capacity

if cap_bank_to_capacity and bank_principal_raw > bank_capacity_total:
    bank_principal = bank_capacity_total
    seller_principal = finance_needed - bank_principal
else:
    bank_principal = bank_principal_raw
    seller_principal = seller_principal_raw

seller_extra_map = {}
if alleviator_amount > 0 and 1 <= alleviator_month <= int(seller_term) * 12:
    seller_extra_map[alleviator_month] = alleviator_amount

bank_io_months = 12 if bank_structure == "IO 12m then Amortizing" else 0
bank_m_df, bank_y_df = build_amortization_schedule(
    principal=bank_principal, annual_rate=bank_rate, term_years=int(bank_term),
    structure=bank_structure, loan_label="Bank", io_months=bank_io_months
)
seller_m_df, seller_y_df = build_amortization_schedule(
    principal=seller_principal, annual_rate=seller_rate, term_years=int(seller_term),
    structure=seller_structure, loan_label="Seller", extra_principal_map=seller_extra_map
)

combined_m_df = pad_and_sum_monthly(bank_m_df, seller_m_df)

def year1_and_avg_later(df_monthly: pd.DataFrame):
    if df_monthly.empty: return 0.0, 0.0
    annual = to_annual(df_monthly)
    y1 = float(annual.loc[annual["Year"]==1, "Payments"].sum()) if (annual["Year"]==1).any() else 0.0
    later = annual.loc[annual["Year"]>=2, "Payments"]
    avg_later = float(later.mean()) if not later.empty else 0.0
    return y1, avg_later

repay_year1_total, repay_avg_later_total = year1_and_avg_later(combined_m_df)

def per_loan_year1_and_avg_later(df_monthly: pd.DataFrame):
    if df_monthly.empty: return 0.0, 0.0
    annual = to_annual(df_monthly)
    y1 = float(annual.loc[annual["Year"]==1, "Payments"].sum()) if (annual["Year"]==1).any() else 0.0
    later = annual.loc[annual["Year"]>=2, "Payments"]
    avg_later = float(later.mean()) if not later.empty else 0.0
    return y1, avg_later

bank_y1, bank_later = per_loan_year1_and_avg_later(bank_m_df)
seller_y1, seller_later = per_loan_year1_and_avg_later(seller_m_df)

ebitda_adjusted = max(ebitda_input - (operator_salary if use_operator_salary else 0.0), 0.0)
profit_after_debt_yr1 = ebitda_adjusted - repay_year1_total
profit_after_debt_later = ebitda_adjusted - repay_avg_later_total
buffer_ratio_yr1 = (repay_year1_total / ebitda_adjusted) if ebitda_adjusted > 0 else float("inf")
buffer_ratio_later = (repay_avg_later_total / ebitda_adjusted) if ebitda_adjusted > 0 else float("inf")

# ========= Header / KPIs =========
st.title("📈 Debt Servicing Calculator — Bank + Seller Note (Capacity-aware)")
st.caption("Comma-friendly money inputs on the left. Percent fields accept '10%' or '0.10'. Shows IO-first year, seller alleviator, bank capacity capping, and per-loan monthly costs.")

top_cols = st.columns([1,1,1,1,1,1])
with top_cols[0]: st.metric("Sale Price", fmt_money(sale_price))
with top_cols[1]: st.metric("Equity Roll", f"{fmt_money(equity_roll_value)} ({fmt_pct(equity_roll_pct)})")
with top_cols[2]: st.metric("Deposit", f"{fmt_money(deposit_value)} ({fmt_pct(deposit_pct)})")
with top_cols[3]: st.metric("Financed Amount", fmt_money(finance_needed))
with top_cols[4]: st.metric("Bank Principal", fmt_money(bank_principal))
with top_cols[5]: st.metric("Seller Principal", fmt_money(seller_principal))

st.markdown("### Bank Capacity Guide")
cap1, cap2, cap3, cap4 = st.columns(4)
with cap1: st.metric("Unsecured Capacity", fmt_money(unsecured_capacity), help="EBITDA × unsecured multiple")
with cap2: st.metric("FFE Capacity", fmt_money(secured_capacity), help="FFE Value × advance rate")
with cap3: st.metric("Total Bank Capacity", fmt_money(bank_capacity_total))
with cap4:
    gap = bank_principal_raw - bank_capacity_total
    st.metric("Bank Over Capacity?", fmt_money(gap) if gap > 0 else "None")

st.markdown("---")

center_cols = st.columns([1,2,1])
with center_cols[1]:
    st.markdown(
        """
        <div style="text-align:center;">
            <h2 style="margin-bottom:0.5rem;">Key Servicing Numbers (Blended)</h2>
            <p style="color:#6b7280;margin-top:0;">Year-1 vs Avg. Years 2+ (IO year lowers Y1 if selected)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    kpi = st.container(border=True)
    with kpi:
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Year 1 Repayments", fmt_money(repay_year1_total))
            st.metric("Monthly (Year 1)", fmt_money(repay_year1_total/12 if repay_year1_total else 0.0))
        with k2:
            st.metric("Avg. Year 2+ Repayments", fmt_money(repay_avg_later_total))
            st.metric("Monthly (Y2+ avg.)", fmt_money(repay_avg_later_total/12 if repay_avg_later_total else 0.0))
        with k3:
            st.metric("Debt as % EBITDA (Y1)", "∞" if buffer_ratio_yr1 == float("inf") else fmt_pct(buffer_ratio_yr1))
            st.metric("Debt as % EBITDA (Y2+)", "∞" if buffer_ratio_later == float("inf") else fmt_pct(buffer_ratio_later))
            st.caption("Lower is safer. Consider WC, capex, tax, contingencies.")

# ========= Monthly costs by loan =========
st.markdown("### Monthly Repayments by Loan")
byloan1, byloan2 = st.columns(2)
with byloan1:
    card = st.container(border=True)
    with card:
        st.subheader("Year 1 — Monthly")
        if bank_y1 > 0: st.metric("Bank (Y1 Monthly)", fmt_money(bank_y1 / 12))
        if seller_y1 > 0: st.metric("Seller (Y1 Monthly)", fmt_money(seller_y1 / 12))
        st.metric("Total (Y1 Monthly)", fmt_money(repay_year1_total / 12 if repay_year1_total else 0.0))
with byloan2:
    card = st.container(border=True)
    with card:
        st.subheader("Avg. Years 2+ — Monthly")
        if bank_later > 0: st.metric("Bank (Y2+ Monthly)", fmt_money(bank_later / 12))
        if seller_later > 0: st.metric("Seller (Y2+ Monthly)", fmt_money(seller_later / 12))
        st.metric("Total (Y2+ Monthly)", fmt_money(repay_avg_later_total / 12 if repay_avg_later_total else 0.0))

st.caption(
    "🛈 Notes: Year-1 may be lower if the bank loan uses an interest-only (IO) first year; "
    "Year-1 may be higher if the Seller ‘Tax Burden Alleviator’ lump-sum occurs in Year-1. "
    "Avg. Years 2+ reflects steady-state after IO and may change again as one loan matures earlier than the other."
)

# ========= Loan Snapshots =========
st.markdown("### Loan Snapshots")
snap1, snap2 = st.columns(2)
with snap1:
    card = st.container(border=True)
    with card:
        st.subheader("Bank Loan")
        st.write(f"Structure: **{bank_structure}** · Rate: **{bank_rate:.2f}%** · Term: **{int(bank_term)} yrs**")
        st.write(f"Principal: **{fmt_money(bank_principal)}**")
        if bank_structure == "IO 12m then Amortizing":
            st.caption("First 12 months interest-only, then fixed P+I.")
        if not bank_m_df.empty:
            y1_pay = to_annual(bank_m_df).loc[lambda d: d['Year']==1,'Payments'].sum()
            st.write(f"Year 1 Payments: **{fmt_money(y1_pay)}**")
with snap2:
    card = st.container(border=True)
    with card:
        st.subheader("Seller Note")
        st.write(f"Structure: **{seller_structure}** · Rate: **{seller_rate:.2f}%** · Term: **{int(seller_term)} yrs**")
        st.write(f"Principal: **{fmt_money(seller_principal)}**")
        if alleviator_amount > 0:
            st.write(f"Alleviator: **{fmt_money(alleviator_amount)}** in **Month {int(alleviator_month)}**")
        if not seller_m_df.empty:
            y1_pay = to_annual(seller_m_df).loc[lambda d: d['Year']==1,'Payments'].sum()
            st.write(f"Year 1 Payments: **{fmt_money(y1_pay)}**")

# ========= Amortization View (Combined) =========
st.markdown("### Amortization View (Combined)")
view = st.radio("Choose view", ["Monthly", "Quarterly", "Annual"], horizontal=True)
combined_table_df, label_col, interest_col, principal_col = (None, None, None, None)
if combined_m_df is None or combined_m_df.empty:
    st.info("Enter valid loan values to generate a combined amortization schedule.")
else:
    combined_table_df, label_col, interest_col, principal_col = build_view_table(view, combined_m_df)
    if view == "Monthly":
        show_cols = ["Period","Year","Month","Label","Payment","Interest","Principal","Ending Balance","Cum Interest","Cum Principal"]
    elif view == "Quarterly":
        show_cols = ["Year","Quarter","Label","Payments","Interest","Principal","Ending Balance"]
    else:
        show_cols = ["Year","Label","Payments","Interest","Principal","Ending Balance"]

    st.dataframe(
        combined_table_df[show_cols].style.format({
            "Payment": "{:,.2f}", "Payments": "{:,.2f}", "Interest": "{:,.2f}",
            "Principal": "{:,.2f}", "Ending Balance": "{:,.2f}",
            "Cum Interest": "{:,.2f}", "Cum Principal": "{:,.2f}",
        }),
        use_container_width=True, height=380
    )

# ========= Chart =========
st.markdown("### Principal vs Interest Over Time (Combined)")
fig = make_chart(
    df=combined_table_df, label_col=label_col, interest_col=interest_col, principal_col=principal_col,
    title=f"{view} Principal vs Interest"
)
if fig is None:
    st.info("Chart will appear once the amortization schedule is available.")
else:
    st.pyplot(fig, use_container_width=True)
    img_buf = BytesIO()
    fig.savefig(img_buf, format="png", dpi=160, bbox_inches="tight")
    img_buf.seek(0)
    st.download_button("⬇️ Download Chart (PNG)", data=img_buf, file_name=f"principal_vs_interest_{view.lower()}.png", mime="image/png")

# ========= Export: Excel =========
st.markdown("### Export")
summary = {
    "Sale Price": sale_price,
    "Equity Roll %": equity_roll_pct, "Equity Roll Value": equity_roll_value,
    "Deposit %": deposit_pct, "Deposit Value": deposit_value,
    "Financed Amount": finance_needed,
    "Bank Principal (final)": bank_principal,
    "Seller Principal (final)": seller_principal,
    "Bank Principal (raw split)": bank_principal_raw,
    "Seller Principal (raw split)": seller_principal_raw,
    "Bank Capacity - Unsecured": unsecured_capacity,
    "Bank Capacity - FFE": secured_capacity,
    "Bank Capacity - Total": bank_capacity_total,
    "Cap to Capacity Applied?": cap_bank_to_capacity,
    "Bank Rate %": bank_rate, "Bank Term (yrs)": int(bank_term), "Bank Structure": bank_structure,
    "Seller Rate %": seller_rate, "Seller Term (yrs)": int(seller_term), "Seller Structure": seller_structure,
    "Seller Alleviator Amount": alleviator_amount,
    "Seller Alleviator Month": int(alleviator_month) if alleviator_amount > 0 else None,
    "EBITDA (input)": ebitda_input,
    "Operator Salary Deducted?": use_operator_salary,
    "Operator Salary (annual)": operator_salary if use_operator_salary else 0.0,
    "EBITDA Used for Servicing": ebitda_adjusted,
    "Year 1 Repayments (Bank)": bank_y1,
    "Year 1 Repayments (Seller)": seller_y1,
    "Year 1 Repayments (Total)": repay_year1_total,
    "Avg. Y2+ Repayments (Bank)": bank_later,
    "Avg. Y2+ Repayments (Seller)": seller_later,
    "Avg. Y2+ Repayments (Total)": repay_avg_later_total,
    "Debt as % EBITDA (Y1)": buffer_ratio_yr1 if ebitda_adjusted > 0 else None,
    "Debt as % EBITDA (Y2+)": buffer_ratio_later if ebitda_adjusted > 0 else None,
}
summary_df = pd.DataFrame([summary])

quarterly_df = to_quarterly(combined_m_df) if not combined_m_df.empty else pd.DataFrame()
annual_df = to_annual(combined_m_df) if not combined_m_df.empty else pd.DataFrame()

excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    summary_df.to_excel(writer, index=False, sheet_name="Summary (Blended+Capacity)")
    if not bank_m_df.empty:
        bank_m_df.to_excel(writer, index=False, sheet_name="Bank (Monthly)")
        to_quarterly(bank_m_df).to_excel(writer, index=False, sheet_name="Bank (Quarterly)")
        to_annual(bank_m_df).to_excel(writer, index=False, sheet_name="Bank (Annual)")
    if not seller_m_df.empty:
        seller_m_df.to_excel(writer, index=False, sheet_name="Seller (Monthly)")
        to_quarterly(seller_m_df).to_excel(writer, index=False, sheet_name="Seller (Quarterly)")
        to_annual(seller_m_df).to_excel(writer, index=False, sheet_name="Seller (Annual)")
    if not combined_m_df.empty:
        combined_m_df.to_excel(writer, index=False, sheet_name="Combined (Monthly)")
        if not quarterly_df.empty:
            quarterly_df.to_excel(writer, index=False, sheet_name="Combined (Quarterly)")
        if not annual_df.empty:
            annual_df.to_excel(writer, index=False, sheet_name="Combined (Annual)")
excel_buffer.seek(0)

st.download_button(
    "⬇️ Download Excel (Stack + Capacity)",
    data=excel_buffer,
    file_name="debt_servicing_stack_capacity.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ========= Export: PDF (Dashboard Summary via ReportLab) =========
def build_dashboard_pdf(summary_dict: dict) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16*mm, rightMargin=16*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="h1", parent=styles["Heading1"], fontSize=16, spaceAfter=8))
    styles.add(ParagraphStyle(name="h2", parent=styles["Heading2"], fontSize=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="small", parent=styles["Normal"], fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="body", parent=styles["Normal"], fontSize=10, leading=14))

    elements = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    elements.append(Paragraph("Debt Servicing Calculator — Dashboard Summary", styles["h1"]))
    elements.append(Paragraph(f"Generated: {now}", styles["small"]))
    elements.append(Spacer(1, 6))

    meta_rows = [
        ["Sale Price", fmt_money(summary_dict["Sale Price"]), "Financed Amount", fmt_money(summary_dict["Financed Amount"])],
        ["Equity Roll", f"{fmt_money(summary_dict['Equity Roll Value'])} ({fmt_pct(summary_dict['Equity Roll %'])})",
         "Deposit", f"{fmt_money(summary_dict['Deposit Value'])} ({fmt_pct(summary_dict['Deposit %'])})"],
        ["Bank Principal (final)", fmt_money(summary_dict["Bank Principal (final)"]),
         "Seller Principal (final)", fmt_money(summary_dict["Seller Principal (final)"])],
        ["EBITDA (input)", fmt_money(summary_dict["EBITDA (input)"]),
         "EBITDA Used for Servicing", fmt_money(summary_dict["EBITDA Used for Servicing"])],
    ]
    t = Table(meta_rows, colWidths=[45*mm, 45*mm, 45*mm, 45*mm])
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (1,0), (-1,-1), "RIGHT"),
    ]))
    elements.append(t); elements.append(Spacer(1, 6))

    cov_rows = [
        ["Year 1 Repayments (Total)", fmt_money(summary_dict["Year 1 Repayments (Total)"]),
         "Monthly (Y1)", fmt_money(summary_dict["Year 1 Repayments (Total)"]/12 if summary_dict["Year 1 Repayments (Total)"] else 0.0)],
        ["Avg. Y2+ Repayments (Total)", fmt_money(summary_dict["Avg. Y2+ Repayments (Total)"]),
         "Monthly (Y2+ avg.)", fmt_money(summary_dict["Avg. Y2+ Repayments (Total)"]/12 if summary_dict["Avg. Y2+ Repayments (Total)"] else 0.0)],
        ["Debt as % EBITDA (Y1)", fmt_pct(summary_dict["Debt as % EBITDA (Y1)"]),
         "Debt as % EBITDA (Y2+)", fmt_pct(summary_dict["Debt as % EBITDA (Y2+)"])],
    ]
    t2 = Table(cov_rows, colWidths=[60*mm, 30*mm, 60*mm, 30*mm])
    t2.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ALIGN", (1,0), (-1,-1), "RIGHT"),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
    ]))
    elements.append(Paragraph("Coverage Summary", styles["h2"]))
    elements.append(t2); elements.append(Spacer(1, 6))

    elements.append(Paragraph("Monthly Repayments by Loan", styles["h2"]))
    perloan_rows = [
        ["", "Bank", "Seller", "Total"],
        ["Year 1 (Monthly)",
         fmt_money2(summary_dict["Year 1 Repayments (Bank)"]/12 if summary_dict["Year 1 Repayments (Bank)"] else 0.0),
         fmt_money2(summary_dict["Year 1 Repayments (Seller)"]/12 if summary_dict["Year 1 Repayments (Seller)"] else 0.0),
         fmt_money2(summary_dict["Year 1 Repayments (Total)"]/12 if summary_dict["Year 1 Repayments (Total)"] else 0.0)],
        ["Avg. Y2+ (Monthly)",
         fmt_money2(summary_dict["Avg. Y2+ Repayments (Bank)"]/12 if summary_dict["Avg. Y2+ Repayments (Bank)"] else 0.0),
         fmt_money2(summary_dict["Avg. Y2+ Repayments (Seller)"]/12 if summary_dict["Avg. Y2+ Repayments (Seller)"] else 0.0),
         fmt_money2(summary_dict["Avg. Y2+ Repayments (Total)"]/12 if summary_dict["Avg. Y2+ Repayments (Total)"] else 0.0)],
    ]
    t3 = Table(perloan_rows, colWidths=[40*mm, 40*mm, 40*mm, 40*mm])
    t3.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
    ]))
    elements.append(t3); elements.append(Spacer(1, 6))

    elements.append(Paragraph("Loan Snapshots", styles["h2"]))
    snap_txt = (
        f"<b>Bank</b>: {summary_dict['Bank Structure']}, {summary_dict['Bank Rate %']:.2f}% p.a., "
        f"{summary_dict['Bank Term (yrs)']} yrs, Principal {fmt_money(summary_dict['Bank Principal (final)'])}<br/>"
        f"<b>Seller</b>: {summary_dict['Seller Structure']}, {summary_dict['Seller Rate %']:.2f}% p.a., "
        f"{summary_dict['Seller Term (yrs)']} yrs, Principal {fmt_money(summary_dict['Seller Principal (final)'])}"
    )
    elements.append(Paragraph(snap_txt, getSampleStyleSheet()["Normal"]))
    if summary_dict.get("Seller Alleviator Amount", 0) and summary_dict.get("Seller Alleviator Month"):
        elements.append(Paragraph(
            f"Seller Alleviator: {fmt_money(summary_dict['Seller Alleviator Amount'])} in Month {summary_dict['Seller Alleviator Month']}",
            getSampleStyleSheet()["Normal"])
        )
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("How to Use — SOP", styles["h2"]))
    sop_inputs = (
        "- Sale Price / EBITDA / FFE value: whole dollars (commas ok).<br/>"
        "- Equity Roll / Deposit / FFE advance: enter as % (e.g., 10% or 0.10).<br/>"
        "- Debt Stack Split: % of financed amount as Seller Note (bank = balance).<br/>"
        "- Bank Structure: Amortizing / IO Full Term / IO 12m then Amortizing.<br/>"
        "- Seller Alleviator: one-off extra principal in a chosen month.<br/>"
        "- Bank Capacity: Unsecured (EBITDA×multiple) + Secured (FFE×advance). Cap bank loan if desired.<br/>"
        "- Operator Salary: optional deduction from EBITDA before coverage."
    )
    elements.append(Paragraph(sop_inputs, getSampleStyleSheet()["Normal"]))
    elements.append(Spacer(1, 4))
    sop_outputs = (
        "- Key Servicing Numbers: Y1 vs Avg. Y2+ (and monthly), Debt as % of EBITDA.<br/>"
        "- Monthly by Loan: Bank & Seller shown separately and totalled.<br/>"
        "- Loan Snapshots: principals, terms, structures, rates.<br/>"
        "- Amortization Views & Excel export; Risk Hints for coverage & capacity."
    )
    elements.append(Paragraph(sop_outputs, getSampleStyleSheet()["Normal"]))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("Definitions", styles["h2"]))
    defs = (
        "- EBITDA — Earnings Before Interest, Taxes, Depreciation & Amortization.<br/>"
        "- FFE — Furniture, Fixtures & Equipment (collateral).<br/>"
        "- P+I — Principal & Interest (amortizing).<br/>"
        "- IO — Interest-Only (principal later or at maturity).<br/>"
        "- Alleviator — one-off extra principal (seller).<br/>"
        "- Y1 / Y2+ — Year-1 and average of Years 2 and beyond."
    )
    elements.append(Paragraph(defs, getSampleStyleSheet()["Normal"]))

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()

pdf_bytes = build_dashboard_pdf(summary)
st.download_button("🖨️ Download PDF (Dashboard Summary)", data=pdf_bytes, file_name="debt_servicing_dashboard.pdf", mime="application/pdf")

# ========= NEW: Print-styled HTML view (opens in new tab) =========
def build_print_html(summary_dict: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    style = """
    <style>
      @media print {
        .no-print { display:none; }
      }
      body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin:24px; color:#111; }
      h1 { font-size:22px; margin:0 0 6px 0; }
      h2 { font-size:16px; margin:16px 0 6px 0; }
      table { width:100%; border-collapse:collapse; margin:8px 0 16px 0; }
      th, td { border:1px solid #ddd; padding:6px 8px; font-size:12px; }
      th { background:#f6f6f6; text-align:left; }
      .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
      .muted { color:#555; font-size:12px; }
      .btn { display:inline-block; padding:8px 12px; border:1px solid #bbb; border-radius:6px; text-decoration:none; color:#111; }
      .headerline { margin-bottom:12px; }
    </style>
    """
    def m(v): return fmt_money(v)
    def p(v): return fmt_pct(v)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">{style}</head>
    <body>
      <div class="headerline">
        <h1>Debt Servicing Calculator — Dashboard Summary</h1>
        <div class="muted">Generated: {now}</div>
        <div class="no-print" style="margin-top:8px;">
          <a class="btn" href="#" onclick="window.print()">Print / Save as PDF</a>
        </div>
      </div>

      <h2>Deal Snapshot</h2>
      <table>
        <tr><th>Sale Price</th><td>{m(summary_dict['Sale Price'])}</td><th>Financed Amount</th><td>{m(summary_dict['Financed Amount'])}</td></tr>
        <tr><th>Equity Roll</th><td>{m(summary_dict['Equity Roll Value'])} ({p(summary_dict['Equity Roll %'])})</td>
            <th>Deposit</th><td>{m(summary_dict['Deposit Value'])} ({p(summary_dict['Deposit %'])})</td></tr>
        <tr><th>Bank Principal (final)</th><td>{m(summary_dict['Bank Principal (final)'])}</td>
            <th>Seller Principal (final)</th><td>{m(summary_dict['Seller Principal (final)'])}</td></tr>
        <tr><th>EBITDA (input)</th><td>{m(summary_dict['EBITDA (input)'])}</td>
            <th>EBITDA Used for Servicing</th><td>{m(summary_dict['EBITDA Used for Servicing'])}</td></tr>
      </table>

      <h2>Coverage Summary</h2>
      <table>
        <tr><th>Year 1 Repayments (Total)</th><td>{m(summary_dict['Year 1 Repayments (Total)'])}</td>
            <th>Monthly (Y1)</th><td>{m((summary_dict['Year 1 Repayments (Total)'] or 0)/12)}</td></tr>
        <tr><th>Avg. Y2+ Repayments (Total)</th><td>{m(summary_dict['Avg. Y2+ Repayments (Total)'])}</td>
            <th>Monthly (Y2+ avg.)</th><td>{m((summary_dict['Avg. Y2+ Repayments (Total)'] or 0)/12)}</td></tr>
        <tr><th>Debt as % EBITDA (Y1)</th><td>{p(summary_dict['Debt as % EBITDA (Y1)'])}</td>
            <th>Debt as % EBITDA (Y2+)</th><td>{p(summary_dict['Debt as % EBITDA (Y2+)'])}</td></tr>
      </table>

      <h2>Monthly Repayments by Loan</h2>
      <table>
        <tr><th></th><th>Bank</th><th>Seller</th><th>Total</th></tr>
        <tr><th>Year 1 (Monthly)</th>
            <td>{fmt_money2((summary_dict['Year 1 Repayments (Bank)'] or 0)/12)}</td>
            <td>{fmt_money2((summary_dict['Year 1 Repayments (Seller)'] or 0)/12)}</td>
            <td>{fmt_money2((summary_dict['Year 1 Repayments (Total)'] or 0)/12)}</td></tr>
        <tr><th>Avg. Y2+ (Monthly)</th>
            <td>{fmt_money2((summary_dict['Avg. Y2+ Repayments (Bank)'] or 0)/12)}</td>
            <td>{fmt_money2((summary_dict['Avg. Y2+ Repayments (Seller)'] or 0)/12)}</td>
            <td>{fmt_money2((summary_dict['Avg. Y2+ Repayments (Total)'] or 0)/12)}</td></tr>
      </table>

      <h2>Loan Snapshots</h2>
      <table>
        <tr><th>Bank</th><td>{summary_dict['Bank Structure']}, {summary_dict['Bank Rate %']:.2f}% p.a., {summary_dict['Bank Term (yrs)']} yrs, Principal {m(summary_dict['Bank Principal (final)'])}</td></tr>
        <tr><th>Seller</th><td>{summary_dict['Seller Structure']}, {summary_dict['Seller Rate %']:.2f}% p.a., {summary_dict['Seller Term (yrs)']} yrs, Principal {m(summary_dict['Seller Principal (final)'])}</td></tr>
      </table>
      {"<div class='muted'>Seller Alleviator: " + m(summary_dict['Seller Alleviator Amount']) + " in Month " + str(summary_dict['Seller Alleviator Month']) + "</div>" if summary_dict.get('Seller Alleviator Amount', 0) and summary_dict.get('Seller Alleviator Month') else ""}

      <h2>How to Use — SOP</h2>
      <div class="muted">
        - Sale Price / EBITDA / FFE value: whole dollars (commas ok).<br/>
        - Equity Roll / Deposit / FFE advance: enter as % (e.g., 10% or 0.10).<br/>
        - Debt Stack Split: % of financed amount as Seller Note (bank = balance).<br/>
        - Bank Structure: Amortizing / IO Full Term / IO 12m then Amortizing.<br/>
        - Seller Alleviator: one-off extra principal in a chosen month.<br/>
        - Bank Capacity: Unsecured (EBITDA×multiple) + Secured (FFE×advance).<br/>
        - Operator Salary: optional deduction from EBITDA before coverage.
      </div>

      <h2>Definitions</h2>
      <div class="muted">
        - EBITDA — Earnings Before Interest, Taxes, Depreciation & Amortization.<br/>
        - FFE — Furniture, Fixtures & Equipment (collateral).<br/>
        - P+I — Principal & Interest (amortizing).<br/>
        - IO — Interest-Only (principal later or at maturity).<br/>
        - Alleviator — one-off extra principal (seller).<br/>
        - Y1 / Y2+ — Year-1 and average of Years 2 and beyond.
      </div>
    </body></html>
    """
    return html

# Build HTML and expose as data URL to open in new tab
html_str = build_print_html(summary)
html_b64 = base64.b64encode(html_str.encode("utf-8")).decode("ascii")
data_url = f"data:text/html;base64,{html_b64}"
st.markdown(f'<a href="{data_url}" target="_blank" class="st-emotion-cache-0">🖨️ Open Print View (HTML)</a>', unsafe_allow_html=True)

# ========= Risk Hints =========
st.markdown("---")
warn = []
worst_repay = max(repay_year1_total, repay_avg_later_total)
if ebitda_adjusted == 0:
    warn.append("Adjusted EBITDA is zero; debt service is not covered.")
elif worst_repay > ebitda_adjusted:
    warn.append("Repayments exceed adjusted EBITDA in at least one phase (negative coverage).")
elif worst_repay > 0.7 * ebitda_adjusted:
    warn.append("Debt service consumes >~70% of adjusted EBITDA in at least one phase (thin buffer).")
if cap_bank_to_capacity and bank_principal_raw > bank_capacity_total:
    warn.append("Bank principal capped by capacity; excess reallocated to Seller Note.")
if warn:
    st.warning(" • ".join(warn))
else:
    st.info("Coverage looks reasonable based on inputs. Layer in working capital, capex, taxes, and contingencies.")

# ========= How to Use — SOP (full on page) =========
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
- **Loan Snapshots**: Per-loan principal, structure, rate, term, and Year-1 totals.
- **Amortization View**: Switch between **Monthly**, **Quarterly**, or **Annual** summaries; export chart and Excel workbook; **Print View / PDF**.
- **Risk Hints**: Flags thin/negative coverage or bank capacity capping.
    """
)

# ========= Definitions =========
st.markdown("### Definitions")
st.markdown(
    """
- **EBITDA** — Earnings Before Interest, Taxes, Depreciation & Amortization.
- **FFE** — Furniture, Fixtures & Equipment (used as collateral for secured lending).
- **P+I** — Principal & Interest (standard amortizing payments).
- **IO** — Interest-Only (pay interest only; principal due later or at maturity).
- **Alleviator** — Seller “Tax Burden Alleviator”: a one-off extra principal payment (you choose the month).
- **Y1 / Y2+** — Year-1 and Years 2 and beyond (average of all years ≥ 2).
    """
)
