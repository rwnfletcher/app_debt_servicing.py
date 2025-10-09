# app_debt_servicing.py
import math
from io import BytesIO
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Debt Servicing Calculator — Bank + Seller Note (Capacity-aware)",
    page_icon="📈",
    layout="wide",
)

# ========= Display helpers =========
def fmt_money(x: float) -> str:
    try:
        return f"${float(x or 0):,.0f}"
    except:
        return "$0"

def fmt_pct(x: float) -> str:
    try:
        return f"{float(x):.1%}"
    except:
        return "—"

# ========= Input helpers (money with commas; percent text) =========
def _parse_money_str(s: str, default: float = 0.0) -> float:
    if s is None:
        return float(default)
    try:
        s = s.replace(",", "").strip()
        if s == "":
            return float(default)
        return float(s)
    except:
        return float(default)

def money_input(label: str, default: float, key: str, help: str | None = None) -> float:
    """Text input that accepts commas. Returns float."""
    val_str = st.text_input(label, value=f"{default:,.2f}", key=key, help=help)
    return _parse_money_str(val_str, default)

def _parse_percent_str(s: str, default: float = 0.0) -> float:
    """
    Accepts '10%', '10', or '0.10' and returns a fraction: 0.10
    """
    if s is None:
        return float(default)
    s = s.strip().replace(",", "")
    if s.endswith("%"):
        s = s[:-1].strip()
        try:
            return float(s) / 100.0
        except:
            return float(default)
    try:
        v = float(s)
        return v / 100.0 if v > 1 else v
    except:
        return float(default)

def percent_input(label: str, default_fraction: float, key: str, help: str | None = None) -> float:
    """Text input that accepts '%', returns fraction (e.g., 0.10)."""
    default_str = f"{default_fraction*100:.1f}%"
    s = st.text_input(label, value=default_str, key=key, help=help)
    return _parse_percent_str(s, default_fraction)

# ========= Finance helpers =========
def pmt(rate_per_period: float, n_periods: int, present_value: float) -> float:
    """Standard PMT for amortizing loan (end-of-period)."""
    if n_periods <= 0:
        return 0.0
    if rate_per_period == 0:
        return present_value / n_periods
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
    """
    Build monthly + yearly amortization schedule for a single loan.

    structure:
      - "Amortizing (P+I)"
      - "Interest-Only (Full Term)"
      - "IO 12m then Amortizing" (use io_months=12)
    extra_principal_map: {month_index: extra_principal_amount}
    """
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
        in_io_phase = (
            structure == "Interest-Only (Full Term)"
            or (structure == "IO 12m then Amortizing" and t <= io_months)
        )

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

        # Extra principal (e.g., seller alleviator)
        extra = float(extra_principal_map.get(t, 0.0))
        if extra > 0:
            principal_component += extra
            payment += extra

        # Final rounding snap
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
    """Align two monthly schedules by Period and sum numeric columns."""
    if df_a.empty and df_b.empty:
        return pd.DataFrame()
    frames = []
    for df in [df_a, df_b]:
        if df.empty: continue
        frames.append(df[["Period","Payment","Interest","Principal","Ending Balance","Year","Month","Quarter"]].copy())

    max_period = 0
    for df in frames:
        max_period = max(max_period, int(df["Period"].max()))

    agg = pd.DataFrame({"Period": range(1, max_period + 1)})
    for col in ["Payment","Interest","Principal","Ending Balance"]:
        agg[col] = 0.0

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
        Payments=("Payment","sum"),
        Interest=("Interest","sum"),
        Principal=("Principal","sum"),
    )
    end_bal = df_monthly.groupby(["Year","Quarter"])["Ending Balance"].last().reset_index(name="Ending Balance")
    q = pd.merge(q, end_bal, on=["Year","Quarter"], how="left")
    q["Label"] = q.apply(lambda r: f"Y{int(r['Year'])} Q{int(r['Quarter'])}", axis=1)
    return q

def to_annual(df_monthly: pd.DataFrame) -> pd.DataFrame:
    if df_monthly.empty: return df_monthly
    y = df_monthly.groupby("Year", as_index=False).agg(
        Payments=("Payment","sum"), Interest=("Interest","sum"), Principal=("Principal","sum")
    )
    y["Ending Balance"] = df_monthly.groupby("Year")["Ending Balance"].last().values
    y["Label"] = y["Year"].apply(lambda x: f"Year {int(x)}")
    return y

def build_view_table(view: str, monthly_df: pd.DataFrame):
    if monthly_df is None or monthly_df.empty:
        return monthly_df, None, None, None
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

    # Money inputs with commas (EBITDA included)
    sale_price = money_input("Sale Price", 5_000_000.0, key="sale_price")
    ebitda_input = money_input("EBITDA (annual, before any operator salary)", 1_500_000.0, key="ebitda")

    use_operator_salary = st.checkbox("Subtract an Operator Salary before servicing?", value=False)
    operator_salary = 0.0
    if use_operator_salary:
        operator_salary = money_input("Operator Salary (annual)", 250_000.0, key="op_salary")

    st.markdown("---")
    st.subheader("Equity / Deposit")
    equity_roll_pct = percent_input("Equity Roll", 0.00, key="equity_roll_pct")
    deposit_pct = percent_input("Deposit / Down Payment", 0.00, key="deposit_pct")

    st.markdown("---")
    st.subheader("Debt Stack Split")
    split_seller_pct = st.slider("Seller Note (% of financed stack)", 0.0, 100.0, 20.0, 1.0) / 100.0
    st.caption("Bank share = 1 − seller share. Applied to the financed amount after equity roll and deposit.")

    st.markdown("---")
    st.subheader("Bank Loan")
    bank_structure = st.selectbox(
        "Structure",
        ["Amortizing (P+I)", "Interest-Only (Full Term)", "IO 12m then Amortizing"]
    )
    bank_rate = st.number_input("Interest Rate (annual %)", min_value=0.0, value=6.0, step=0.25, format="%.2f")
    bank_term = st.number_input("Term (years)", min_value=1, value=7, step=1)

    st.subheader("Seller Note")
    seller_structure = st.selectbox("Structure ", ["Amortizing (P+I)", "Interest-Only (Full Term)"])
    seller_rate = st.number_input("Interest Rate (annual %)", min_value=0.0, value=8.0, step=0.25, format="%.2f")
    seller_term = st.number_input("Term (years)", min_value=1, value=5, step=1)

    st.markdown("##### Seller: Tax Burden Alleviator")
    alleviator_amount = money_input("Tax Burden Alleviator (extra principal, A$)", 0.0, key="allev_amt")
    alleviator_month = st.number_input(
        "Month number for Alleviator (1–term months)",
        min_value=1, max_value=max(1, seller_term * 12), value=min(6, seller_term * 12), step=1
    )

    st.markdown("---")
    st.subheader("Bank Capacity (Guide)")
    unsecured_multiple = st.number_input("Unsecured finance vs EBITDA (× multiple)", min_value=0.0, value=2.2, step=0.1, format="%.2f")
    ffe_value = money_input("FFE / Equipment Value (A$)", 0.0, key="ffe_val")
    ffe_advance_rate = percent_input("Advance rate on FFE", 0.70, key="ffe_adv_rate")
    cap_bank_to_capacity = st.checkbox("Cap bank loan to capacity & reallocate excess to Seller Note", value=True)

# ========= Core Calculations =========
equity_roll_value = sale_price * equity_roll_pct
deposit_value = sale_price * deposit_pct
finance_needed = max(sale_price - equity_roll_value - deposit_value, 0.0)

# Raw split of financed amount
bank_principal_raw = finance_needed * (1 - split_seller_pct)
seller_principal_raw = finance_needed - bank_principal_raw

# Bank capacity
unsecured_capacity = ebitda_input * unsecured_multiple
secured_capacity = ffe_value * ffe_advance_rate
bank_capacity_total = unsecured_capacity + secured_capacity

# Cap bank to capacity if selected (overflow to seller)
if cap_bank_to_capacity and bank_principal_raw > bank_capacity_total:
    bank_principal = bank_capacity_total
    seller_principal = finance_needed - bank_principal
else:
    bank_principal = bank_principal_raw
    seller_principal = seller_principal_raw

# Seller extra principal map (Tax Burden Alleviator)
seller_extra_map = {}
if alleviator_amount > 0 and 1 <= alleviator_month <= seller_term * 12:
    seller_extra_map[alleviator_month] = alleviator_amount

# Build schedules
bank_io_months = 12 if bank_structure == "IO 12m then Amortizing" else 0
bank_m_df, bank_y_df = build_amortization_schedule(
    principal=bank_principal, annual_rate=bank_rate, term_years=bank_term,
    structure=bank_structure, loan_label="Bank", io_months=bank_io_months
)
seller_m_df, seller_y_df = build_amortization_schedule(
    principal=seller_principal, annual_rate=seller_rate, term_years=seller_term,
    structure=seller_structure, loan_label="Seller", extra_principal_map=seller_extra_map
)

# Combined monthly schedule
combined_m_df = pad_and_sum_monthly(bank_m_df, seller_m_df)

# ---- Annual payments for Year 1 and Avg Years 2+ (combined) ----
def year1_and_avg_later(df_monthly: pd.DataFrame):
    if df_monthly.empty: return 0.0, 0.0
    annual = to_annual(df_monthly)
    y1 = float(annual.loc[annual["Year"]==1, "Payments"].sum()) if (annual["Year"]==1).any() else 0.0
    later = annual.loc[annual["Year"]>=2, "Payments"]
    avg_later = float(later.mean()) if not later.empty else 0.0
    return y1, avg_later

repay_year1_total, repay_avg_later_total = year1_and_avg_later(combined_m_df)

# ---- Per-loan Year 1 and Avg Years 2+ (for monthly display by loan) ----
def per_loan_year1_and_avg_later(df_monthly: pd.DataFrame):
    if df_monthly.empty:
        return 0.0, 0.0
    annual = to_annual(df_monthly)
    y1 = float(annual.loc[annual["Year"]==1, "Payments"].sum()) if (annual["Year"]==1).any() else 0.0
    later = annual.loc[annual["Year"]>=2, "Payments"]
    avg_later = float(later.mean()) if not later.empty else 0.0
    return y1, avg_later

bank_y1, bank_later = per_loan_year1_and_avg_later(bank_m_df)
seller_y1, seller_later = per_loan_year1_and_avg_later(seller_m_df)

# EBITDA adjust
ebitda_adjusted = max(ebitda_input - (operator_salary if use_operator_salary else 0.0), 0.0)

# Profit after debt service — both phases (combined)
profit_after_debt_yr1 = ebitda_adjusted - repay_year1_total
profit_after_debt_later = ebitda_adjusted - repay_avg_later_total

# Buffers (combined)
buffer_ratio_yr1 = (repay_year1_total / ebitda_adjusted) if ebitda_adjusted > 0 else float("inf")
buffer_ratio_later = (repay_avg_later_total / ebitda_adjusted) if ebitda_adjusted > 0 else float("inf")

# ========= Header / KPIs =========
st.title("📈 Debt Servicing Calculator — Bank + Seller Note (Capacity-aware)")
st.caption("Comma-friendly money inputs on the left. Percent fields accept '10%' or '0.10'. Shows IO-first year, seller alleviator, bank capacity capping, and per-loan monthly costs.")

top_cols = st.columns([1,1,1,1,1,1])
with top_cols[0]:
    st.metric("Sale Price", fmt_money(sale_price))
with top_cols[1]:
    st.metric("Equity Roll", f"{fmt_money(equity_roll_value)} ({fmt_pct(equity_roll_pct)})")
with top_cols[2]:
    st.metric("Deposit", f"{fmt_money(deposit_value)} ({fmt_pct(deposit_pct)})")
with top_cols[3]:
    st.metric("Financed Amount", fmt_money(finance_needed))
with top_cols[4]:
    st.metric("Bank Principal", fmt_money(bank_principal))
with top_cols[5]:
    st.metric("Seller Principal", fmt_money(seller_principal))

st.markdown("### Bank Capacity Guide")
cap1, cap2, cap3, cap4 = st.columns(4)
with cap1:
    st.metric("Unsecured Capacity", fmt_money(unsecured_capacity), help="EBITDA × unsecured multiple")
with cap2:
    st.metric("FFE Capacity", fmt_money(secured_capacity), help="FFE Value × advance rate")
with cap3:
    st.metric("Total Bank Capacity", fmt_money(bank_capacity_total))
with cap4:
    gap = bank_principal_raw - bank_capacity_total
    gap_display = fmt_money(gap) if gap > 0 else "None"
    st.metric("Bank Over Capacity?", gap_display)

st.markdown("---")

# ========= Blended coverage KPIs =========
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
            b1 = "∞" if buffer_ratio_yr1 == float("inf") else fmt_pct(buffer_ratio_yr1)
            b2 = "∞" if buffer_ratio_later == float("inf") else fmt_pct(buffer_ratio_later)
            st.metric("Debt as % EBITDA (Y1)", b1)
            st.metric("Debt as % EBITDA (Y2+)", b2)
            st.caption("Lower is safer. Consider WC, capex, tax, contingencies.")

# ========= Monthly costs by loan (separate + total) =========
st.markdown("### Monthly Repayments by Loan")
byloan1, byloan2 = st.columns(2)

with byloan1:
    card = st.container(border=True)
    with card:
        st.subheader("Year 1 — Monthly")
        if bank_y1 > 0:
            st.metric("Bank (Y1 Monthly)", fmt_money(bank_y1 / 12))
        if seller_y1 > 0:
            st.metric("Seller (Y1 Monthly)", fmt_money(seller_y1 / 12))
        st.metric("Total (Y1 Monthly)", fmt_money(repay_year1_total / 12 if repay_year1_total else 0.0))

with byloan2:
    card = st.container(border=True)
    with card:
        st.subheader("Avg. Years 2+ — Monthly")
        if bank_later > 0:
            st.metric("Bank (Y2+ Monthly)", fmt_money(bank_later / 12))
        if seller_later > 0:
            st.metric("Seller (Y2+ Monthly)", fmt_money(seller_later / 12))
        st.metric("Total (Y2+ Monthly)", fmt_money(repay_avg_later_total / 12 if repay_avg_later_total else 0.0))

# Explanatory note under per-loan monthly section
st.caption(
    "🛈 Notes: Year-1 may be lower if the bank loan uses an interest-only (IO) first year; "
    "Year-1 may be higher if the Seller 'Tax Burden Alleviator' lump-sum occurs in Year-1. "
    "Avg. Years 2+ reflects steady-state after IO and may change again as one loan matures earlier than the other."
)

# ========= Loan Snapshots =========
st.markdown("### Loan Snapshots")
snap1, snap2 = st.columns(2)
with snap1:
    card = st.container(border=True)
    with card:
        st.subheader("Bank Loan")
        st.write(f"Structure: **{bank_structure}** · Rate: **{bank_rate:.2f}%** · Term: **{bank_term} yrs**")
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
        st.write(f"Structure: **{seller_structure}** · Rate: **{seller_rate:.2f}%** · Term: **{seller_term} yrs**")
        st.write(f"Principal: **{fmt_money(seller_principal)}**")
        if alleviator_amount > 0:
            st.write(f"Alleviator: **{fmt_money(alleviator_amount)}** in **Month {alleviator_month}**")
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
            "Payment": "{:,.2f}",
            "Payments": "{:,.2f}",
            "Interest": "{:,.2f}",
            "Principal": "{:,.2f}",
            "Ending Balance": "{:,.2f}",
            "Cum Interest": "{:,.2f}",
            "Cum Principal": "{:,.2f}",
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

# ========= Export =========
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
    "Bank Rate %": bank_rate, "Bank Term (yrs)": bank_term, "Bank Structure": bank_structure,
    "Seller Rate %": seller_rate, "Seller Term (yrs)": seller_term, "Seller Structure": seller_structure,
    "Seller Alleviator Amount": alleviator_amount,
    "Seller Alleviator Month": alleviator_month if alleviator_amount > 0 else None,
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

# Precise export views
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

# ========= How to Use — SOP =========
st.markdown("---")
st.markdown("## How to Use — SOP")
st.markdown("### Inputs")
st.markdown(
    """
- **Sale Price / EBITDA / FFE value**: Type whole dollars with commas (e.g., `5,000,000`).
- **Equity Roll / Deposit / FFE advance**: Type as `%` or decimal (e.g., `10%` or `0.10`).
- **Debt Stack Split**: Choose what % of the financed amount is a **Seller Note** (bank gets the rest).
- **Bank Structure**:
  - *Amortizing (P+I)*: fixed payments across the term.
  - *Interest-Only (Full Term)*: interest only until maturity, balloon at end.
  - *IO 12m then Amortizing*: first 12 months interest-only, then fixed P+I.
- **Seller Alleviator**: Optional one-off extra principal in a specified month (default Month 6).
- **Bank Capacity**: Unsecured capacity = EBITDA × multiple; secured = FFE × advance rate. Optionally cap the bank loan to this capacity (overflow goes to the Seller Note).
- **Operator Salary**: Optional; deducted from EBITDA *before* debt coverage is assessed.
    """
)
st.markdown("### Outputs")
st.markdown(
    """
- **Key Servicing Numbers (Blended)**: Year-1 vs Avg. Years 2+ totals and monthly equivalents. Coverage shown as **Debt as % of EBITDA**.
- **Monthly by Loan**: Shows **Bank** and **Seller** monthly figures separately (Year-1 and Avg. Y2+), plus totals.
- **Loan Snapshots**: Per-loan principal, structure, rate, term, and Year-1 total.
- **Amortization View**: Switch between **Monthly**, **Quarterly**, or **Annual** summaries; download the chart and Excel workbook.
- **Risk Hints**: Flags thin or negative coverage, or bank capacity capping.
    """
)
