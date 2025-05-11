import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from io import BytesIO
import openpyxl
from scipy.integrate import simpson

# Decline curve functions
def exponential(t, qi, D):
    return qi * np.exp(-D * t)

def harmonic(t, qi, D):
    return qi / (1 + D * t)

def hyperbolic(t, qi, D, b):
    return qi / (1 + b * D * t) ** (1 / b)

# App layout
st.title("📉 Decline Curve Analysis (DCA) App with Forecast & EUR")
st.markdown("""
Upload an Excel file with columns **'time'** and **'rate'** to analyze production decline, forecast future production, and estimate EUR.
""")

# Upload Excel
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    if 'time' not in df.columns or 'rate' not in df.columns:
        st.error("The Excel file must contain 'time' and 'rate' columns.")
    else:
        # Extract time and rate
        t_data = df['time'].values
        q_data = df['rate'].values

        # Curve fitting
        popt_exp, _ = curve_fit(exponential, t_data, q_data, p0=[q_data[0], 0.01])
        qi_exp, D_exp = popt_exp
        q_exp = exponential(t_data, qi_exp, D_exp)
        rmse_exp = np.sqrt(mean_squared_error(q_data, q_exp))

        popt_harm, _ = curve_fit(harmonic, t_data, q_data, p0=[q_data[0], 0.01])
        qi_harm, D_harm = popt_harm
        q_harm = harmonic(t_data, qi_harm, D_harm)
        rmse_harm = np.sqrt(mean_squared_error(q_data, q_harm))

        popt_hyp, _ = curve_fit(hyperbolic, t_data, q_data, p0=[q_data[0], 0.01, 0.5],
                                bounds=([0, 0, 0], [np.inf, 1, 2]))
        qi_hyp, D_hyp, b_hyp = popt_hyp
        q_hyp = hyperbolic(t_data, qi_hyp, D_hyp, b_hyp)
        rmse_hyp = np.sqrt(mean_squared_error(q_data, q_hyp))

        # Select best-fit model
        rmses = {'Exponential': rmse_exp, 'Harmonic': rmse_harm, 'Hyperbolic': rmse_hyp}
        best_model = min(rmses, key=rmses.get)

        st.subheader("📊 Model Fitting Results")
        st.markdown(f"""
        - **Exponential**: RMSE = `{rmse_exp:.2f}`, D = `{D_exp:.4f}`
        - **Harmonic**: RMSE = `{rmse_harm:.2f}`, D = `{D_harm:.4f}`
        - **Hyperbolic**: RMSE = `{rmse_hyp:.2f}`, D = `{D_hyp:.4f}`, b = `{b_hyp:.4f}`
        """)
        st.success(f"✅ Best Fit: **{best_model}**")

        # Forecast input
        st.subheader("📅 Forecast Parameters")
        forecast_days = st.number_input("Enter forecast duration (in days):", min_value=1, max_value=10000, value=365*5)
        t_forecast = np.linspace(0, forecast_days, 100)

        # Predict using best model
        if best_model == 'Exponential':
            q_forecast = exponential(t_forecast, qi_exp, D_exp)
        elif best_model == 'Harmonic':
            q_forecast = harmonic(t_forecast, qi_harm, D_harm)
        elif best_model == 'Hyperbolic':
            q_forecast = hyperbolic(t_forecast, qi_hyp, D_hyp, b_hyp)

        # Calculate EUR
        eur = simpson(q_forecast, t_forecast)

        st.subheader("📈 Forecast & EUR")
        st.markdown(f"""
        - **Forecast Duration**: `{forecast_days}` days  
        - **Estimated Ultimate Recovery (EUR)**: `{eur:.2f}` STB
        """)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(t_data, q_data, label='Actual Data', color='black')
        ax.plot(t_forecast, exponential(t_forecast, qi_exp, D_exp), label='Exponential Fit', linestyle='--')
        ax.plot(t_forecast, harmonic(t_forecast, qi_harm, D_harm), label='Harmonic Fit', linestyle='--')
        ax.plot(t_forecast, hyperbolic(t_forecast, qi_hyp, D_hyp, b_hyp), label='Hyperbolic Fit', linestyle='--')
        ax.plot(t_forecast, q_forecast, label=f'{best_model} Forecast', color='red', linewidth=2)

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Production Rate (STB/day)")
        ax.set_title("Decline Curve Forecast")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Export forecast and EUR to Excel
        export_df = pd.DataFrame({
            'time (days)': t_forecast,
            'forecast rate': q_forecast
        })
        export_df.loc[0, 'EUR (STB)'] = eur  # Add EUR in the sheet

        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name="Forecast")

        st.download_button(
            label="📥 Download Forecast & EUR",
            data=towrite.getvalue(),
            file_name="dca_forecast_eur.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
