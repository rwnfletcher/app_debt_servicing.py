# app_debt_servicing.py
import math
import base64
from io import BytesIO
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# --- Optional ReportLab import ---
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

st.set_page_config(page_title="Debt Servicing Calculator — Bank + Seller Note (Capacity-aware)", page_icon="📈", layout="wide")

# ========= Default Config =========
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

def reset_defaults():
    st.session_state.clear()

# ========= Theme + Reset =========
top_controls = st.columns([1, 1, 6])
with top_controls[0]:
    theme_choice = st.radio("Theme", ["Light", "Dark"], key="theme_choice", horizontal=True)
with top_controls[1]:
    if st.button("Reset to defaults"):
        reset_defaults()
        st.rerun()

if theme_choice == "Dark":
    st.markdown("""
    <style>
    .stApp, .block-container { background:#0f172a !important; color:#e2e8f0 !important; }
    .stButton>button, .stDownloadButton>button { background:#1f2937; color:#e2e8f0; border:1px solid #374151; }
    </style>
    """, unsafe_allow_html=True)

# ========= Format Helpers =========
def fmt_money(x): return f"${float(x or 0):,.0f}"
def fmt_money2(x): return f"${float(x or 0):,.2f}"
def fmt_pct(x): return f"{float(x):.1%}" if x not in (None, float('inf')) else "—"

# ========= Input Helpers =========
def _parse_money(s, default=0.0):
    try: return float(str(s).replace(",","").strip() or default)
    except: return default

def _parse_pct(s, default=0.0):
    s = str(s).replace(",","").strip()
    if s.endswith("%"): s = s[:-1]
    try:
        v = float(s)
        return v/100 if v>1 else v
    except:
        return default

def money_input(label, default, key):
    val_str = st.text_input(label, value=f"{default:,.2f}", key=key)
    return _parse_money(val_str, default)

def percent_input(label, default, key):
    val_str = st.text_input(label, value=f"{default*100:.1f}%", key=key)
    return _parse_pct(val_str, default)

# ========= Financial Logic =========
def pmt(r,n,pv):
    if n<=0: return 0
    if r==0: return pv/n
    return (r*pv)/(1-(1+r)**(-n))

def build_amort(principal, rate, years, struct, label, io_months=0, extra=None):
    extra=extra or {}
    r=rate/100/12; n=int(years*12)
    rows=[]; bal=principal; pay_after=None
    full=pmt(r,n,principal)
    for t in range(1,n+1):
        io=(struct=="Interest-Only (Full Term)") or (struct=="IO 12m then Amortizing" and t<=io_months)
        interest=bal*r
        if io: pay=interest; pr=0
        else:
            if struct=="Amortizing (P+I)": pay=full
            elif struct=="IO 12m then Amortizing":
                if pay_after is None: pay_after=pmt(r,n-io_months,bal)
                pay=pay_after
            else: pay=pmt(r,n,bal)
            pr=pay-interest
        add=extra.get(t,0)
        pr+=add; pay+=add
        if t==n: pr=bal; pay=interest+bal
        bal=max(bal-pr,0)
        rows.append((label,t,pay,interest,pr,bal))
    df=pd.DataFrame(rows,columns=["Loan","Period","Payment","Interest","Principal","Ending"])
    df["Year"]=((df["Period"]-1)//12)+1
    return df

def to_annual(df):
    if df.empty: return df
    y=df.groupby("Year",as_index=False).agg(Payments=("Payment","sum"))
    return y

def perloan_years(df):
    if df.empty: return (0,0)
    y=to_annual(df)
    y1=float(y.loc[y["Year"]==1,"Payments"].sum())
    rest=y.loc[y["Year"]>=2,"Payments"]
    avg=float(rest.mean()) if not rest.empty else 0
    return y1,avg

# ========= Sidebar Inputs =========
with st.sidebar:
    st.header("Deal Inputs")
    sale_price = money_input("Sale Price", DEFAULTS["sale_price"], "sale_price")
    ebitda_input = money_input("EBITDA (annual, before any operator salary)", DEFAULTS["ebitda"], "ebitda")
    use_op = st.checkbox("Subtract Operator Salary?", DEFAULTS["use_operator_salary"], key="use_op")
    op_sal = money_input("Operator Salary (annual)", DEFAULTS["op_salary"], "op_salary") if use_op else 0

    st.markdown("---")
    st.subheader("Equity / Deposit")
    eq = percent_input("Equity Roll", DEFAULTS["equity_roll_pct"], "equity_roll_pct")
    dep = percent_input("Deposit / Down Payment", DEFAULTS["deposit_pct"], "deposit_pct")

    st.markdown("---")
    st.subheader("Debt Stack Split")
    split_seller_pct = st.slider(
        "Seller Note (% of financed stack)", 0, 100, int(DEFAULTS["split_seller_pct"]*100), 1, key="split_seller_slider"
    )/100.0
    st.caption("Bank share = 1 − seller share. Applied after equity roll and deposit.")

    st.markdown("---")
    st.subheader("Bank Loan")
    bank_struct = st.selectbox("Structure",["Amortizing (P+I)","Interest-Only (Full Term)","IO 12m then Amortizing"],index=2)
    bank_rate = st.number_input("Interest Rate (annual %)",0.0,20.0,DEFAULTS["bank_rate"],0.25)
    bank_term = st.number_input("Term (years)",1,30,DEFAULTS["bank_term"],1)

    st.subheader("Seller Note")
    sell_struct = st.selectbox("Structure",["Amortizing (P+I)","Interest-Only (Full Term)"],index=0)
    sell_rate = st.number_input("Interest Rate (annual %)",0.0,20.0,DEFAULTS["seller_rate"],0.25)
    sell_term = st.number_input("Term (years)",1,30,DEFAULTS["seller_term"],1)

    st.markdown("##### Seller: Tax Burden Alleviator")
    allev_amt = money_input("Alleviator (extra principal, A$)", DEFAULTS["allev_amt"], "allev_amt")
    allev_month = st.number_input("Month number for Alleviator (1–term months)",1,int(sell_term*12),DEFAULTS["allev_month"],1)

    st.markdown("---")
    st.subheader("Bank Capacity (Guide)")
    unsec_mult = st.number_input("Unsecured finance vs EBITDA (× multiple)",0.0,10.0,DEFAULTS["unsecured_multiple"],0.1)
    ffe_val = money_input("FFE / Equipment Value (A$)", DEFAULTS["ffe_val"], "ffe_val")
    ffe_adv = percent_input("Advance rate on FFE", DEFAULTS["ffe_adv_rate"], "ffe_adv_rate")
    cap_bank = st.checkbox("Cap bank loan to capacity & reallocate excess", DEFAULTS["cap_bank_to_capacity"])

# ========= Core Calculations =========
eq_val=sale_price*eq; dep_val=sale_price*dep
fin_needed=max(sale_price-eq_val-dep_val,0)
bank_raw=fin_needed*(1-split_seller_pct)
sell_raw=fin_needed-bank_raw
unsec=ebitda_input*unsec_mult; sec=ffe_val*ffe_adv; cap_total=unsec+sec
if cap_bank and bank_raw>cap_total:
    bank_prin=cap_total; sell_prin=fin_needed-bank_prin
else: bank_prin=bank_raw; sell_prin=sell_raw
extra={allev_month:allev_amt} if allev_amt>0 else {}
bank_df=build_amort(bank_prin,bank_rate,bank_term,bank_struct,"Bank",12 if bank_struct=="IO 12m then Amortizing" else 0)
sell_df=build_amort(sell_prin,sell_rate,sell_term,sell_struct,"Seller",0,extra)
comb=pd.concat([bank_df,sell_df]).sort_values("Period")
comb["PaymentSum"]=comb.groupby("Period")["Payment"].transform("sum")

bank_y1,bank_y2=perloan_years(bank_df); sell_y1,sell_y2=perloan_years(sell_df)
comb_y=to_annual(comb)
y1=float(comb_y.loc[comb_y["Year"]==1,"Payments"].sum())
later=float(comb_y.loc[comb_y["Year"]>=2,"Payments"].mean()) if not comb_y.empty else 0
ebitda_adj=max(ebitda_input-(op_sal if use_op else 0),0)
buf_y1=y1/ebitda_adj if ebitda_adj else float("inf")
buf_y2=later/ebitda_adj if ebitda_adj else float("inf")

# ========= Dashboard =========
st.title("📊 Debt Servicing Calculator — Bank + Seller Note")
st.metric("Sale Price",fmt_money(sale_price))
st.metric("Financed Amount",fmt_money(fin_needed))
st.metric("Bank Principal",fmt_money(bank_prin))
st.metric("Seller Principal",fmt_money(sell_prin))

st.markdown("### Coverage Summary")
cols=st.columns(3)
cols[0].metric("Year 1 Repayments",fmt_money(y1))
cols[1].metric("Avg. Y2+ Repayments",fmt_money(later))
cols[2].metric("Debt as % EBITDA",f"{buf_y1*100:.1f}% (Y1) / {buf_y2*100:.1f}% (Y2+)")

st.markdown("### Monthly Repayments by Loan")
cols2=st.columns(3)
cols2[0].metric("Bank (Y1 Monthly)",fmt_money(bank_y1/12))
cols2[1].metric("Seller (Y1 Monthly)",fmt_money(sell_y1/12))
cols2[2].metric("Total (Y1 Monthly)",fmt_money(y1/12))
cols2b=st.columns(3)
cols2b[0].metric("Bank (Y2+ Monthly)",fmt_money(bank_y2/12))
cols2b[1].metric("Seller (Y2+ Monthly)",fmt_money(sell_y2/12))
cols2b[2].metric("Total (Y2+ Monthly)",fmt_money(later/12))
st.caption("🛈 Year-1 may differ due to IO period or Seller Alleviator payment.")

# ========= Print / Export =========
def build_print_html(summary):
    now=datetime.now().strftime("%Y-%m-%d %H:%M")
    def m(v):return fmt_money(v)
    def p(v):return fmt_pct(v)
    html=f"""
    <html><head><meta charset='utf-8'><title>Debt Servicing Summary</title>
    <style>
    body{{font-family:Arial;margin:24px;color:#111}}
    table{{width:100%;border-collapse:collapse;margin:8px 0}}
    th,td{{border:1px solid #ccc;padding:6px 8px;font-size:12px}}
    th{{background:#f6f6f6}}
    </style></head><body>
    <h2>Debt Servicing Calculator — Summary</h2>
    <p>Generated {now}</p>
    <a style='border:1px solid #ccc;padding:6px 10px;border-radius:5px;' onclick='window.print()'>Print / Save as PDF</a>
    <h3>Snapshot</h3>
    <table><tr><th>Sale Price</th><td>{m(summary['Sale Price'])}</td><th>Financed</th><td>{m(summary['Financed Amount'])}</td></tr>
    <tr><th>Bank Principal</th><td>{m(summary['Bank Principal (final)'])}</td><th>Seller Principal</th><td>{m(summary['Seller Principal (final)'])}</td></tr>
    <tr><th>EBITDA</th><td>{m(summary['EBITDA (input)'])}</td><th>EBITDA Used</th><td>{m(summary['EBITDA Used for Servicing'])}</td></tr></table>
    <h3>Coverage</h3>
    <table><tr><th>Year 1 Total</th><td>{m(summary['Year 1 Repayments (Total)'])}</td><th>Avg. Y2+ Total</th><td>{m(summary['Avg. Y2+ Repayments (Total)'])}</td></tr>
    <tr><th>Debt as % EBITDA (Y1)</th><td>{p(summary['Debt as % EBITDA (Y1)'])}</td><th>Debt as % EBITDA (Y2+)</th><td>{p(summary['Debt as % EBITDA (Y2+)'])}</td></tr></table>
    <h3>Definitions</h3>
    <ul><li><b>EBITDA</b>: Earnings Before Interest, Taxes, Depreciation & Amortization.</li>
    <li><b>FFE</b>: Furniture, Fixtures & Equipment.</li><li><b>P+I</b>: Principal & Interest.</li>
    <li><b>IO</b>: Interest-Only.</li><li><b>Alleviator</b>: One-off Seller principal payment.</li></ul>
    </body></html>"""
    return html

summary={
 "Sale Price":sale_price,"Financed Amount":fin_needed,"Bank Principal (final)":bank_prin,"Seller Principal (final)":sell_prin,
 "EBITDA (input)":ebitda_input,"EBITDA Used for Servicing":ebitda_adj,
 "Year 1 Repayments (Total)":y1,"Avg. Y2+ Repayments (Total)":later,
 "Debt as % EBITDA (Y1)":buf_y1,"Debt as % EBITDA (Y2+)":buf_y2,
}
html=build_print_html(summary)
html64=base64.b64encode(html.encode()).decode()
url=f"data:text/html;base64,{html64}"
st.markdown(f"<a href='{url}' target='_blank'>🖨️ Open Print View (HTML)</a>",unsafe_allow_html=True)

if REPORTLAB_AVAILABLE:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    buf=BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4)
    styles=getSampleStyleSheet()
    doc.build([Paragraph("Debt Servicing Summary",styles['Heading1']),
               Paragraph(f"Sale Price: {fmt_money(sale_price)}<br/>Financed: {fmt_money(fin_needed)}",styles['Normal']),
               Paragraph(f"EBITDA: {fmt_money(ebitda_input)}<br/>EBITDA Used: {fmt_money(ebitda_adj)}",styles['Normal'])])
    buf.seek(0)
    st.download_button("📄 Download PDF",data=buf.getvalue(),file_name="debt_servicing.pdf",mime="application/pdf")
else:
    st.info("Add `reportlab` to requirements.txt to enable PDF export. The HTML Print View still works.")
