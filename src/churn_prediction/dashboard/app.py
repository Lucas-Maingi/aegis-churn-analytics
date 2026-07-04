import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "test_api_key_1234")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# Set page config
st.set_page_config(
    page_title="Churn Risk Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Risk Tier Pills */
    .risk-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.05em;
        text-align: center;
    }
    .risk-HIGH {
        background-color: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.5);
    }
    .risk-MEDIUM {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.5);
    }
    .risk-LOW {
        background-color: rgba(34, 197, 94, 0.2);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.5);
    }
    
    /* Driver Card */
    .driver-card {
        background: rgba(51, 65, 85, 0.5);
        border-left: 4px solid #38bdf8;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

def render_gauge(probability, tier):
    color_map = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability", 'font': {'color': '#f8fafc', 'size': 20}},
        number = {'font': {'color': '#f8fafc', 'size': 40}, 'suffix': "%"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color_map.get(tier, "white")},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 40], 'color': "rgba(34, 197, 94, 0.1)"},
                {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.1)"},
                {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.1)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': probability
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def single_user_profiler():
    st.sidebar.markdown("### Single User Profiler")
    
    with st.sidebar.form("single_user_form"):
        customerID = st.text_input("Customer ID (optional)", "CUST-001")
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
            Partner = st.selectbox("Partner", ["No", "Yes"])
            Dependents = st.selectbox("Dependents", ["No", "Yes"])
        with col2:
            tenure = st.number_input("Tenure (months)", min_value=0, value=12)
            PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
            MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            
        st.markdown("**Services**")
        col3, col4 = st.columns(2)
        with col3:
            OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        with col4:
            TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            
        st.markdown("**Billing & Contracts**")
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        col5, col6 = st.columns(2)
        with col5:
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
        with col6:
            TotalCharges = st.number_input("Total Charges", min_value=0.0, value=600.0)
            
        submitted = st.form_submit_button("Predict Risk", type="primary", use_container_width=True)
        
        if submitted:
            return {
                "customerID": customerID,
                "gender": gender,
                "SeniorCitizen": SeniorCitizen,
                "Partner": Partner,
                "Dependents": Dependents,
                "tenure": tenure,
                "PhoneService": PhoneService,
                "MultipleLines": MultipleLines,
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection,
                "TechSupport": TechSupport,
                "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies,
                "Contract": Contract,
                "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod,
                "MonthlyCharges": float(MonthlyCharges),
                "TotalCharges": float(TotalCharges)
            }
    return None

def main():
    st.title("🔮 Churn Risk Intelligence")
    st.markdown("Predict customer churn probabilities and uncover behavioral risk drivers in real-time.")
    
    # Render Sidebar Profiler
    payload = single_user_profiler()
    
    # Layout
    tab1, tab2 = st.tabs(["Single User Profile", "Bulk File Analytics"])
    
    with tab1:
        if payload:
            with st.spinner("Analyzing risk profile..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload, headers=HEADERS)
                    if response.status_code == 200:
                        data = response.json()
                        prob = data['churn_probability']
                        tier = data['risk_tier']
                        explanations = data['explanations']
                        
                        col_g, col_e = st.columns([1, 1])
                        
                        with col_g:
                            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                            st.plotly_chart(render_gauge(prob, tier), use_container_width=True)
                            st.markdown(f'<div style="text-align: center;"><span class="risk-pill risk-{tier}">RISK TIER: {tier}</span></div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        with col_e:
                            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                            st.markdown("### Top Behavioral Drivers")
                            for exp in explanations:
                                icon = "⬆️" if exp['direction'] == 'increases' else "⬇️"
                                color = "#ef4444" if exp['direction'] == 'increases' else "#22c55e"
                                st.markdown(f"""
                                <div class="driver-card">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                        <strong>{exp['feature_name']}</strong>
                                        <span style="color: {color};">{icon} {abs(exp['shap_value']):.3f} SHAP</span>
                                    </div>
                                    <div style="font-size: 0.9em; opacity: 0.8; margin-bottom: 4px;">Current Value: {exp['feature_value']}</div>
                                    <div style="font-size: 0.95em;">{exp['plain_english']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
        else:
            st.info("👈 Enter customer details in the sidebar and click 'Predict Risk' to generate a profile.")
            
    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Bulk Prediction")
        uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} records.")
                
                # Replace NaNs with empty string or default appropriate values to avoid JSON issues
                df = df.fillna("")
                
                if st.button("Score Customers", type="primary"):
                    with st.spinner("Scoring batch..."):
                        # Convert to JSON payload
                        records = df.to_dict(orient="records")
                        # Handle TotalCharges that might be empty space
                        for r in records:
                            if isinstance(r.get('TotalCharges'), str) and r.get('TotalCharges').strip() == "":
                                r['TotalCharges'] = " "
                                
                        batch_payload = {"customers": records}
                        
                        response = requests.post(f"{API_URL}/predict/batch", json=batch_payload, headers=HEADERS)
                        
                        if response.status_code == 200:
                            results = response.json()['predictions']
                            res_df = pd.DataFrame(results)
                            
                            # Merge back IDs or use indices
                            
                            # Key Metrics
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Scored Customers", len(res_df))
                            col2.metric("Avg Churn Probability", f"{res_df['churn_probability'].mean():.1f}%")
                            col3.metric("High Risk Customers", len(res_df[res_df['risk_tier'] == 'HIGH']))
                            
                            # Charts
                            c1, c2 = st.columns(2)
                            with c1:
                                tier_counts = res_df['risk_tier'].value_counts().reset_index()
                                tier_counts.columns = ['Tier', 'Count']
                                fig_pie = px.pie(tier_counts, values='Count', names='Tier', title="Risk Tier Distribution",
                                                color='Tier', color_discrete_map={'HIGH':'#ef4444', 'MEDIUM':'#f59e0b', 'LOW':'#22c55e'})
                                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                                st.plotly_chart(fig_pie, use_container_width=True)
                                
                            with c2:
                                fig_hist = px.histogram(res_df, x='churn_probability', nbins=20, title="Probability Distribution",
                                                       color_discrete_sequence=['#38bdf8'])
                                fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                                st.plotly_chart(fig_hist, use_container_width=True)
                                
                            # Data Table
                            st.markdown("### Scored Data")
                            display_cols = ['customerID', 'churn_probability', 'risk_tier']
                            st.dataframe(res_df[display_cols].sort_values('churn_probability', ascending=False), use_container_width=True)
                            
                            # Download
                            csv = res_df[display_cols].to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Scored Data",
                                data=csv,
                                file_name='scored_customers.csv',
                                mime='text/csv',
                            )
                        else:
                            st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Error processing file: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
