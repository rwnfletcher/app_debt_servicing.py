# app_debt_servicing.py
import math
from io import BytesIO
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Debt Servicing Calculator", page_icon="📈", layout="wide")

# ========= Helpers =========
def pmt(rate_per_period: float, n_periods: int, present_value: float) -> float:
    """Standard PMT for amortizing loan (end-of-period)."""
    if n_periods <= 0:
        return 0.0
    if rate_per_period == 0:
        return present_value / n_periods
    return (rate_per_period * present_value) / (1 - (1 + rate_per_period) ** (-n_periods))

def fmt_money(x): return f"${x:,.0f}"
def fmt_pct(x): return f"{x:.1%}"

def build_amortization_schedule(
    principal: float,
    annual_rate: float,
    term_years: int,
    structure: str,
    periods_per_year: int = 12,
    loan_label: str = "Loan",
):
    """
    Returns monthly and yearly amortization DataFrames for a single loan.
    structure: "Amortizing (P+I)" or "Interest-Only"
    """
    cols = ["Loan","Period","Payment","Interest","Principal","Ending Balance","Year","Month","Quarter","Cum Interest","Cum Principal"]
    if principal <= 0 or term_years <= 0:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=["Loan","Year","Payments","Interest","Principal","Ending Balance"])

    r = (annual_rate / 100.0) / periods_per_year
    n = int(term_years * periods_per_year)

    rows = []
    balance = principal

    if structure == "Amortizing (P+I)":
        payment = pmt(r, n, principal)
        for t in range(1, n + 1):
            interest = balance * r
            principal_component = payment - interest
            if t == n:  # snap rounding
                principal_component = balance
                payment = interest + principal_component
            balance = max(balance - principal_component, 0.0)
            rows.append((loan_label, t, payment, interest, principal_component, balance))
    else:
        # Interest-only: periodic interest, balloon principal at maturity
        interest_payment = balance * r
        for t in range(1, n):
            payment = interest_payment
            rows.append((loan_label, t, payment, interest_payment, 0.0, balance))
        final_interest = balance * r
        payment = final_interest + balance
        rows.append((loan_label, n, payment, final_interest, balance, 0.0))
        balance = 0.0

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
    """Align two monthly schedules by Period and sum numeric columns, keeping Ending Balance as sum of balances."""
    if df_a.empty and df_b.empty:
        return pd.DataFrame()
    frames = []
    for df in [df_a, df_b]:
        if df.empty:
            continue
        # Minimal set for combine
        frames.append(df[["Period","Payment","Interest","Principal","Ending Balance","Year","Month","Quarter"]].copy())
    # Build master index of periods
    max_period = 0
    for df in frames:
        max_period = max(max_period, df["Period"].max())
    base = pd.DataFrame({"Period": range(1, max_period + 1)})
    # Sum over aligned periods
    agg = base.copy()
    for col in ["Payment","Interest","Principal","Ending Balance"]:
        agg[col] = 0.0
    # Year/Month/Quarter from the longest DF where available
    # Fill progressively
    for df in frames:
        agg = agg.merge(df[["Period","Payment","Interest","Principal","Ending Balance","Year","Month","Quarter"]],
                        on="Period", how="left", suffixes=("","_x"))
        # Add numerics
        for col in ["Payment","Interest","Principal","Ending Balance"]:
            agg[col] = agg[col].fillna(0) + agg[f"{col}_x"].fillna(0)
            agg.drop(columns=[f"{col}_x"], inplace=True)
        # Fill calendar fields if missing
        for cal in ["Year","Month","Quarter"]:
            agg[cal] = agg[cal].fillna(df[cal])
    # Cum totals
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

def to_annual(df_monthly_or_yearly: pd.DataFrame, from_monthly: bool = True) -> pd.DataFrame:
    if df_monthly_or_yearly.empty: return df_monthly_or_yearly
    if from_monthly:
        y = df_monthly_or_yearly.groupby("Year", as_index=False).agg(
            Payments=("Payment","sum"), Interest=("Interest","sum"), Principal=("Principal","sum")
        )
        y["Ending Balance"] = df_monthly_or_yearly.groupby("Year")["Ending Balance"].last().values
    else:
        y = df_monthly_or_yearly.copy()
    y["Label"] = y["Year"].apply(lambda x: f"Year {int(x)}")
    return y

def build_view_table(view: str, monthly_df: pd.DataFrame):
    """Return (table_df, label_col_name, interest_col, principal_col). For combined schedule."""
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
        a = to_annual(monthly_df, from_monthly=True)
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
    sale_price = st.number_input("Sale Price", min_value=0.0, value=5_000_000.0, step=50_000.0, format="%.2f")

    equity_roll_pct = st.slider("Equity Roll (% of Sale Price)", 0.0, 90.0, 0.0, 1.0) / 100.0
    st.caption("Equity roll reduces the cash consideration and financing need.")

    deposit_pct = st.slider("Deposit / Down Payment (% of Sale Price)", 0.0, 90.0, 0.0, 1.0) / 100.0
    st.caption("Cash deposit paid at close (separate from equity roll).")

    ebitda_input = st.number_input("EBITDA (annual, before any operator salary)", min_value=0.0, value=1_500_000.0, step=50_000.0, format="%.2f")

    use_operator_salary = st.checkbox("Subtract an Operator Salary before servicing?", value=False)
    operator_salary = 0.0
    if use_operator_salary:
        operator_salary = st.number_input("Operator Salary (annual)", min_value=0.0, value=250_000.0, step=10_000.0, format="%.2f")

    st.markdown("---")
    st.subheader("Debt Stack Split")
    split_seller_pct = st.slider("Seller Note (% of financed stack)", 0.0, 100.0, 20.0, 1.0) / 100.0
    st.caption("Bank share = 1 - seller share. Applied to the financed amount after equity roll and deposit.")

    st.markdown("---")
    st.subheader("Bank Loan")
    bank_structure = st.selectbox("Bank Structure", ["Amortizing (P+I)", "Interest-Only"])
    bank_rate = st.number_input("Bank Interest Rate (annual %)", min_value=0.0, value=6.0, step=0.25, format="%.2f")
    bank_term = st.number_input("Bank Term (years)", min_value=1, value=7, step=1)

    st.subheader("Seller Note")
    seller_structure = st.selectbox("Seller Structure", ["Amortizing (P+I)", "Interest-Only"])
    seller_rate = st.number_input("Seller Interest Rate (annual %)", min_value=0.0, value=8.0, step=0.25, format="%.2f")
    seller_term = st.number_input("Seller Term (years)", min_value=1, value=5, step=1)

# ========= Core Calculations =========
equity_roll_value = sale_price * equity_roll_pct
deposit_value = sale_price * deposit_pct

# Amount to finance after equity roll and deposit
finance_needed = max(sale_price - equity_roll_value - deposit_value, 0.0)

seller_principal = finance_needed * split_seller_pct
bank_principal = finance_needed - seller_principal

# Build per-loan monthly schedules
bank_m_df, bank_y_df = build_amortization_schedule(
    principal=bank_principal, annual_rate=bank_rate, term_years=bank_term, structure=bank_structure, loan_label="Bank"
)
seller_m_df, seller_y_df = build_amortization_schedule(
    principal=seller_principal, annual_rate=seller_rate, term_years=seller_term, structure=seller_structure, loan_label="Seller"
)

# Combine monthly for blended KPIs
combined_m_df = pad_and_sum_monthly(bank_m_df, seller_m_df)

# Compute repayments
annual_payment_total = combined_m_df.groupby("Year")["Payment"].sum().iloc[0] if not combined_m_df.empty else 0.0
# More robust: compute average annual equivalent from monthly totals
if not combined_m_df.empty:
    annual_payment_total = combined_m_df["Payment"].sum() / (combined_m_df["Year"].max())  # average per year over life
monthly_equiv_payment = annual_payment_total / 12.0

# EBITDA adjust (operator salary before servicing)
ebitda_adjusted = max(ebitda_input - (operator_salary if use_operator_salary else 0.0), 0.0)

# Profit after debt service using annual_equivalent
profit_after_debt_annual = ebitda_adjusted - annual_payment_total
profit_after_debt_monthly = profit_after_debt_annual / 12.0

# Buffer / Margin of Safety
buffer_ratio = (annual_payment_total / ebitda_adjusted) if ebitda_adjusted > 0 else float("inf")

# ========= Header / KPIs =========
st.title("📈 Debt Servicing Calculator — Bank + Seller Note")
st.caption("Equity roll and deposit reduce financing; split remainder between bank and seller note. EBITDA can optionally be reduced by an operator salary before coverage calculations.")

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

st.markdown("---")

center_cols = st.columns([1,2,1])
with center_cols[1]:
    st.markdown(
        """
        <div style="text-align:center;">
            <h2 style="margin-bottom:0.5rem;">Key Servicing Numbers (Blended)</h2>
            <p style="color:#6b7280;margin-top:0;">Repayments, profit left, and safety buffer across both loans</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    kpi = st.container(border=True)
    with kpi:
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Annual Repayments (equiv.)", fmt_money(annual_payment_total))
            st.metric("Monthly Repayments (equiv.)", fmt_money(monthly_equiv_payment))
        with k2:
            st.metric("Profit Left (Annual)", fmt_money(profit_after_debt_annual))
            st.metric("Profit Left (Monthly)", fmt_money(profit_after_debt_monthly))
        with k3:
            buffer_display = "∞" if buffer_ratio == float("inf") else fmt_pct(buffer_ratio)
            st.metric("Debt as % of EBITDA", buffer_display)
            st.caption("Lower is safer. Consider WC, capex, tax, contingencies.")

# ========= Per-Loan Snapshots =========
st.markdown("### Loan Snapshots")
snap1, snap2 = st.columns(2)
with snap1:
    card = st.container(border=True)
    with card:
        st.subheader("Bank Loan")
        st.write(f"Structure: **{bank_structure}** · Rate: **{bank_rate:.2f}%** · Term: **{bank_term} yrs**")
        st.write(f"Principal: **{fmt_money(bank_principal)}**")
        if not bank_m_df.empty:
            st.write(f"First-Year Payments: **{fmt_money(bank_m_df[bank_m_df['Year']==1]['Payment'].sum())}**")
with snap2:
    card = st.container(border=True)
    with card:
        st.subheader("Seller Note")
        st.write(f"Structure: **{seller_structure}** · Rate: **{seller_rate:.2f}%** · Term: **{seller_term} yrs**")
        st.write(f"Principal: **{fmt_money(seller_principal)}**")
        if not seller_m_df.empty:
            st.write(f"First-Year Payments: **{fmt_money(seller_m_df[seller_m_df['Year']==1]['Payment'].sum())}**")

# ========= View Toggle + Table (Combined) =========
st.markdown("### Amortization View (Combined)")
view = st.radio("Choose view", ["Monthly", "Quarterly", "Annual"], horizontal=True)

if combined_m_df is None or combined_m_df.empty:
    st.info("Enter valid loan values to generate a combined amortization schedule.")
    table_df, label_col, interest_col, principal_col = None, None, None, None
else:
    table_df, label_col, interest_col, principal_col = build_view_table(view, combined_m_df)
    # Show table
    if view == "Monthly":
        show_cols = ["Period","Year","Month","Label","Payment","Interest","Principal","Ending Balance","Cum Interest","Cum Principal"]
    elif view == "Quarterly":
        show_cols = ["Year","Quarter","Label","Payments","Interest","Principal","Ending Balance"]
    else:
        show_cols = ["Year","Label","Payments","Interest","Principal","Ending Balance"]

    st.dataframe(
        table_df[show_cols].style.format({
            "Payment":"{:,.2f}",
            "Payments":"{:,.2f}",
            "Interest":"{:,.2f}",
            "Principal":"{:,.2f}",
            "Ending Balance":"{:,.2f}",
            "Cum Interest":"{:,.2f}",
            "Cum Principal":"{:,.2f}",
        }),
        use_container_width=True,
        height=380
    )

# ========= Chart =========
st.markdown("### Principal vs Interest Over Time (Combined)")
fig = make_chart(
    df=table_df, label_col=label_col, interest_col=interest_col, principal_col=principal_col,
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

# Summary row
summary = {
    "Sale Price": sale_price,
    "Equity Roll %": equity_roll_pct, "Equity Roll Value": equity_roll_value,
    "Deposit %": deposit_pct, "Deposit Value": deposit_value,
    "Financed Amount": finance_needed,
    "Bank Principal": bank_principal,
    "Seller Principal": seller_principal,
    "Bank Rate %": bank_rate, "Bank Term (yrs)": bank_term, "Bank Structure": bank_structure,
    "Seller Rate %": seller_rate, "Seller Term (yrs)": seller_term, "Seller Structure": seller_structure,
    "EBITDA (input)": ebitda_input,
    "Operator Salary Deducted?": use_operator_salary,
    "Operator Salary (annual)": operator_salary if use_operator_salary else 0.0,
    "EBITDA Used for Servicing": ebitda_adjusted,
    "Annual Repayments (equiv.)": annual_payment_total,
    "Monthly Repayments (equiv.)": monthly_equiv_payment,
    "Profit Left (Annual)": profit_after_debt_annual,
    "Profit Left (Monthly)": profit_after_debt_monthly,
    "Debt as % of EBITDA": (annual_payment_total / ebitda_adjusted) if ebitda_adjusted > 0 else None,
}
summary_df = pd.DataFrame([summary])

# Build alternate views for export
quarterly_df = to_quarterly(combined_m_df) if not combined_m_df.empty else pd.DataFrame()
annual_df = to_annual(combined_m_df, from_monthly=True) if not combined_m_df.empty else pd.DataFrame()

excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    summary_df.to_excel(writer, index=False, sheet_name="Summary (Blended)")
    # Per-loan sheets
    if not bank_m_df.empty:
        bank_m_df.to_excel(writer, index=False, sheet_name="Bank (Monthly)")
        to_quarterly(bank_m_df).to_excel(writer, index=False, sheet_name="Bank (Quarterly)")
        to_annual(bank_m_df, from_monthly=True).to_excel(writer, index=False, sheet_name="Bank (Annual)")
    if not seller_m_df.empty:
        seller_m_df.to_excel(writer, index=False, sheet_name="Seller (Monthly)")
        to_quarterly(seller_m_df).to_excel(writer, index=False, sheet_name="Seller (Quarterly)")
        to_annual(seller_m_df, from_monthly=True).to_excel(writer, index=False, sheet_name="Seller (Annual)")
    # Combined sheets
    if not combined_m_df.empty:
        combined_m_df.to_excel(writer, index=False, sheet_name="Combined (Monthly)")
        if not quarterly_df.empty:
            quarterly_df.to_excel(writer, index=False, sheet_name="Combined (Quarterly)")
        if not annual_df.empty:
            annual_df.to_excel(writer, index=False, sheet_name="Combined (Annual)")
excel_buffer.seek(0)

st.download_button(
    "⬇️ Download Excel (Stack: Bank + Seller)",
    data=excel_buffer,
    file_name="debt_servicing_stack.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# CSV quick export of current view
if table_df is not None and not table_df.empty:
    st.download_button(
        f"Download {view} View CSV (Combined)",
        data=table_df.to_csv(index=False),
        file_name=f"amortization_combined_{view.lower()}.csv",
        mime="text/csv",
    )

# ========= Risk Hints =========
st.markdown("---")
warn = []
if ebitda_adjusted == 0:
    warn.append("Adjusted EBITDA is zero; debt service is not covered.")
elif annual_payment_total > ebitda_adjusted:
    warn.append("Annual repayments exceed adjusted EBITDA (negative coverage).")
elif annual_payment_total > 0.7 * ebitda_adjusted:
    warn.append("Debt service consumes more than ~70% of adjusted EBITDA (thin buffer).")
if warn:
    st.warning(" • ".join(warn))
else:
    st.info("Coverage looks reasonable based on inputs. Layer in working capital, capex, taxes, and contingencies.")
