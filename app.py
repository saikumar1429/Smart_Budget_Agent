import streamlit as st
import pandas as pd
from groq import Groq

st.set_page_config(
    page_title="Smart AI Budget Agent ",
    layout="wide"
)

def analyze_transactions(df):
    income = df[df["amount"] > 0]["amount"].sum()

    expenses_df = df[df["amount"] < 0].copy()
    expenses_df["amount"] = expenses_df["amount"].abs()

    fixed_categories = ["rent", "utilities", "emi"]
    fixed = expenses_df[expenses_df["category"].str.lower().isin(fixed_categories)]["amount"].sum()

    variable = expenses_df[~expenses_df["category"].str.lower().isin(fixed_categories)]
    variable_summary = variable.groupby("category")["amount"].sum()

    return income, fixed, variable_summary


def generate_budget_plan(income, fixed, variable_summary):
    savings_goal = round(income * 0.15, 2)

    return {
        "income": income,
        "savings": savings_goal,
        "fixed": fixed,
        "variable": variable_summary
    }


def ai_budget_recommendation(budget_data, api_key):
    try:
        client = Groq(api_key=api_key.strip())

        user_prompt = f"""
Monthly Income: {budget_data['income']}
Savings Goal (15%): {budget_data['savings']}
Fixed Expenses: {budget_data['fixed']}

Variable Expenses:
{budget_data['variable'].to_string()}

Tasks:
1. Recommend the best budget plan
2. Identify overspending categories
3. Give 3 clear saving tips
"""

        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "You are a professional financial advisor."},
                {"role": "user", "content": user_prompt.strip()}
            ],
            temperature=0.3,
            max_tokens=500
        )

        return completion.choices[0].message.content, None

    except Exception as e:
        return None, str(e)

st.title("💰 Smart AI Budget Agent ")
st.write(
    "Upload a CSV file with your financial transactions. "
    "The AI will analyze your expenses and recommend a budget plan."
)

api_key = st.text_input("Enter Groq API Key", type="password")

uploaded_file = st.file_uploader(
    "Upload Expense CSV File",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower() for c in df.columns]

    required_cols = {"amount", "category"}
    if not required_cols.issubset(df.columns):
        st.error("CSV must contain at least 'amount' and 'category' columns.")
    else:
        st.success("CSV loaded successfully")
        st.dataframe(df, use_container_width=True)

        income, fixed, variable_summary = analyze_transactions(df)
        budget = generate_budget_plan(income, fixed, variable_summary)

        st.header("📌 Budget Overview")

        st.markdown(f"""
- **Monthly Income:** ₹{income:,.2f}
- **Savings Goal (15%):** ₹{budget['savings']:,.2f}
- **Fixed Expenses:** ₹{fixed:,.2f}
- **Variable Expenses:** ₹{variable_summary.sum():,.2f}
""")

        st.subheader("📊 Variable Expense Breakdown")
        for cat, amt in variable_summary.items():
            st.write(f"- **{cat}**: ₹{amt:,.2f}")

        if st.button("Generate AI Budget Recommendation"):
            if not api_key:
                st.warning("Please enter your Groq API key.")
            else:
                with st.spinner("Analyzing your budget with AI..."):
                    ai_response, error = ai_budget_recommendation(budget, api_key)

                if error:
                    st.error("AI service unavailable")
                    st.info(error)
                else:
                    st.markdown("### 🧠 AI Budget Recommendation")
                    st.success(ai_response)

st.caption(
    "This application uses Groq-hosted LLaMA models to provide intelligent budget planning recommendations."
)
