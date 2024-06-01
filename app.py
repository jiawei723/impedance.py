import streamlit as st
import json
import pandas as pd
import numpy as np
from numpy import log10, absolute, angle
import matplotlib.pyplot as plt
from impedance.models.circuits import Randles, CustomCircuit
from impedance import preprocessing
from impedance.visualization import plot_nyquist


def app_gui():

    st.set_page_config(page_title="impedance.py", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title("impedance.py")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])


    if uploaded_file is not None:
        # Load data from the example EIS data
        frequencies, Z = preprocessing.readCSV(uploaded_file)

        # keep only the impedance data in the first quandrant
        frequencies, Z = preprocessing.ignoreBelowX(frequencies, Z)
    else:
        # st.write("Please upload a CSV impedance file to continue.")
        Z = None

    if Z is not None:
        # choose circuit model type
        model_type_options = ["Randles", "Custom"]
        model_type = st.selectbox("Model Type", model_type_options)

        if model_type == "Randles":
            # create a Randles circuit
            randles_options = st.selectbox('Randles circuit option', ['constant phase element (CPE)', 'ideal capacitor'])
            initial_guess_str = st.text_input("Initial Guess", value=".01,.005,.001,200,.1,.9")
            initial_guess = [float(x) for x in initial_guess_str.split(',')]
            if randles_options == 'constant phase element (CPE)':
                model_label = 'Randles w/ CPE'
                circuit = Randles(initial_guess=initial_guess, CPE=True)
            else:
                model_label = 'Randles'
                circuit = Randles(initial_guess=initial_guess)
            

        elif model_type == "Custom":
            # create a custom circuit
            circuit_options = st.text_input("Custom Circuit", value="R_0-p(R_1,C_1)-p(R_2,C_2)-Wo_1")
            model_label = f'Custom Circuit - {circuit_options}'
            # Create a constant value input
            const_check = st.checkbox("Use constant values")
            if const_check:
                const_input = st.text_area("Enter constant values", value='{"R_0": 0.02, "Wo_1_1": 200}')
                # Convert the JSON string to a dictionary
                try:
                    const_input = json.loads(const_input)
                except json.JSONDecodeError:
                    st.error("Invalid Input format. Please enter a valid dictionary.")
                
                initial_guess_str = st.text_input("Initial Guess", value="None,.005,.1,.005,.1,.001,None")
                initial_guess = [float(x) if x != 'None' else None for x in initial_guess_str.split(',')]
            else:
                const_input = None
                initial_guess_str = st.text_input("Initial Guess", value=".01,.005,.1,.005,.1,.001,200")
                initial_guess = [float(x) for x in initial_guess_str.split(',')]

            circuit = CustomCircuit(initial_guess=initial_guess, circuit=circuit_options, constants=const_input)
        
        st.write("Check the circuit model below")
        st.text(str(circuit))
            
        # Store state variables in st.session_state
        if "model_fitted" not in st.session_state:
            st.session_state.model_fitted = False
        if "circuit" not in st.session_state:
            st.session_state.circuit = None
        if "f_pred" not in st.session_state:
            st.session_state.f_pred = None
        if "circuit_fit" not in st.session_state:
            st.session_state.circuit_fit = None

        # fit the model
        if st.button("Fit Model"):
            circuit.fit(frequencies, Z)
            st.write("Check the Fit circuit model below")
            st.text(str(circuit))

            # plot the data and the model
            f_pred = np.logspace(5,-2)
            circuit_fit = circuit.predict(f_pred)
            fig, ax = plt.subplots(figsize=(4,4))
            plot_nyquist(Z, ax=ax)
            plot_nyquist(circuit_fit, fmt='-', ax=ax)
            ax.legend(['Data', f'{model_label}'], loc='upper center', bbox_to_anchor=(0.5, 1.4))
            st.pyplot(fig)
            fig.savefig('EIS.png')

            if st.button("Export Model"):
                export_filename = st.text_input("Enter filename for model export", value="template_model.json")
                
                # Convert the circuit object to a JSON string
                model_string = circuit.circuit
                model_name = circuit.name
                initial_guess = circuit.initial_guess
                if circuit._is_fit():
                    parameters_ = list(circuit.parameters_)
                    model_conf_ = list(circuit.conf_)
                    data_dict = {"Name": model_name,
                                "Circuit String": model_string,
                                "Initial Guess": initial_guess,
                                "Constants": circuit.constants,
                                "Fit": True,
                                "Parameters": parameters_,
                                "Confidence": model_conf_,
                                }
                else:
                    data_dict = {"Name": model_name,
                                "Circuit String": model_string,
                                "Initial Guess": initial_guess,
                                "Constants": circuit.constants,
                                "Fit": False}
                json_data = json.dumps(data_dict)
                
                # Create the download button
                st.download_button(
                    label="Download Model",
                    data=json_data,
                    file_name=export_filename,
                    mime="application/json"
                )
                # st.success("Model exported successfully.")
            if st.button("Export Figure"):
                export_filename = st.text_input("Enter filename for figure export", value="figure.png")
                with open('EIS.png', "rb") as f1:
                    st.download_button(label=f"Download Figure", data=f1.read(), file_name=export_filename, mime="image/png")
                # st.success(f"Figure exported to {export_filename}")
            if st.button("Export Fit EIS Data"):
                export_filename = st.text_input("Enter filename for fit EIS data export", value="fit_data.csv")
                df = pd.DataFrame(np.column_stack((f_pred, circuit_fit.real, circuit_fit.imag)), columns=["freq", "Z_re", "Z_im"])
                csv = df.to_csv().encode('utf-8')
                st.download_button(label="Download Fit EIS Data", data=csv, file_name=export_filename, mime="text/csv")
                # st.success(f"Fit data exported to {export_filename}")
    else:
        st.write("Please upload a valid CSV impedance file to continue.")    


def main():
    app_gui()

if __name__ == "__main__":
    main()