import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# [theme]
primaryColor="#adb9bf"
backgroundColor="#9dc5c0"
secondaryBackgroundColor="#d2b9b9"
textColor="#1e0202"
font="monospace"

# Load the dataset
df = pd.read_csv("water_potability.csv")

# Preprocessing steps
# Fill missing values with mean
df.fillna(df.mean(), inplace=True)

# Separate features and target variable
X = df.drop('Potability', axis=1)
y = df['Potability']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load the trained model
model_svm = joblib.load('model_svm.pkl')

# Maximum and minimum allowed values for each feature
param_ranges = {
    'pH': {'min': 0.0, 'max': 14.0},
    'Hardness': {'min': 0.0, 'max': 500.0},
    'Solids': {'min': 0.0, 'max': 5000.0},
    'Chloramines': {'min': 0.0, 'max': 20.0},
    'Sulfate': {'min': 0.0, 'max': 500.0},
    'Conductivity': {'min': 0.0, 'max': 5000.0},
    'Organic Carbon': {'min': 0.0, 'max': 50.0},
    'Trihalomethanes': {'min': 0.0, 'max': 150.0},
    'Turbidity': {'min': 0.0, 'max': 10.0}
}

# Water Potability Prediction Page
def water_potability_prediction():
    st.title('Water Potability Prediction')
    st.write('Enter the water quality parameters to predict potability.')
    
    # Create input columns for user input
    col1, col2, col3 = st.columns(3)
    with col1:
        ph = st.text_input('pH', placeholder='14.0', value='0')
        
    with col2:
        hardness = st.text_input('Hardness', placeholder='500.0', value='0')
       
    with col3:
        solids = st.text_input('Solids', placeholder='5000.0', value='0')
        

    col4, col5, col6 = st.columns(3)
    with col4:
        chloramines = st.text_input('Chloramines', placeholder='20.0', value='0')
        
    with col5:
        sulfate = st.text_input('Sulfate', placeholder='500.0', value='0')
        
    with col6:
        conductivity = st.text_input('Conductivity', placeholder='5000.0', value='0')
       

    col7, col8, col9 = st.columns(3)
    with col7:
        organic_carbon = st.text_input('Organic Carbon', placeholder='50.0', value='0')
        
    with col8:
        trihalomethanes = st.text_input('Trihalomethanes', placeholder='150.0', value='0')
    with col9:
        turbidity = st.text_input('Turbidity', placeholder='10.0', value='0')
        

    # Predict button
    if st.button('Predict'):
        # Convert input values to float
        input_values = {
            'pH': {'value': float(ph) if ph else None, 'max': param_ranges['pH']['max']},
            'Hardness': {'value': float(hardness) if hardness else None, 'max': param_ranges['Hardness']['max']},
            'Solids': {'value': float(solids) if solids else None, 'max': param_ranges['Solids']['max']},
            'Chloramines': {'value': float(chloramines) if chloramines else None, 'max': param_ranges['Chloramines']['max']},
            'Sulfate': {'value': float(sulfate) if sulfate else None, 'max': param_ranges['Sulfate']['max']},
            'Conductivity': {'value': float(conductivity) if conductivity else None, 'max': param_ranges['Conductivity']['max']},
            'Organic Carbon': {'value': float(organic_carbon) if organic_carbon else None, 'max': param_ranges['Organic Carbon']['max']},
            'Trihalomethanes': {'value': float(trihalomethanes) if trihalomethanes else None, 'max': param_ranges['Trihalomethanes']['max']},
            'Turbidity': {'value': float(turbidity) if turbidity else None, 'max': param_ranges['Turbidity']['max']}
        }

        # Check if any input value exceeds the maximum allowed value
        contaminated = False
        contaminated_params = []
        for feature, value_dict in input_values.items():
            value = value_dict['value']
            if value is not None and value > value_dict['max']:
                contaminated = True
                contaminated_params.append(feature)
        
        if contaminated:
            st.write('The water is contaminated and it is harmful to humans.')
            st.subheader(':blue[Conclusion/Solution:]')
            for param in contaminated_params:
                 
                st.write(f"The {param} should be in the range of 0 to {param_ranges[param]['max']} to use for the damestic usage")
        else:
            # Create input data as a numpy array
            input_data = np.array([[input_values['pH']['value'], input_values['Hardness']['value'], input_values['Solids']['value'], input_values['Chloramines']['value'],
                                    input_values['Sulfate']['value'], input_values['Conductivity']['value'], input_values['Organic Carbon']['value'],
                                    input_values['Trihalomethanes']['value'], input_values['Turbidity']['value']]])

            # Preprocess input data
            preprocessed_data = scaler.transform(input_data)

            # Make predictions using the loaded model
            prediction_svm = model_svm.predict(preprocessed_data)

            # Display prediction result
            if prediction_svm[0] == 1:
                st.write('The water is potable. That Means that the water is not contaminated and it can be use for the doemstic usage')
            else:
                st.write('The water is not potable.')



# About Page
def Home_page():
    st.title("ABOUT STATEMENT")
    # st.title(''':green[Water quality Assessment on contaminated water and its impacts on Human Health]''')
    st.header( ':blue[Water quality Assessment on contaminated water and its impacts on Human Health]',divider= 'rainbow')
    st.write("An evaluation assessment of contaminated water and its impacts on human health involves a systematic analysis of various aspects related to water contamination and its effects on individuals and communities. ")
    st.subheader('Water Quality Analysis:')  
    st.write("This involves testing water samples to identify the presence and concentration of contaminants. It includes assessing parameters such as microbial content (bacteria, viruses, parasites), chemical composition (heavy metals, pesticides, toxins), and physical characteristics (turbidity, odor, color).")
    st.subheader('Exposure Assessment: ')
    st.write('Evaluating how people come into contact with contaminated water. This includes assessing different exposure pathways such as ingestion (drinking, cooking, food preparation), inhalation (steam, aerosols), and dermal contact (bathing, swimming). Factors such as frequency, duration, and intensity of exposure are considered.')

    st.subheader('Health Risk Assessment:')
    st.write('Determining the potential health risks associated with exposure to contaminated water. This involves considering toxicity data for identified contaminants, epidemiological studies linking exposure to adverse health outcomes, and factors influencing susceptibility (age, health status, genetics). Quantitative risk assessment may be conducted to estimate the likelihood and severity of health effects.')
    st.subheader('Health Impact Analysis:')
    st.write('Examining the actual health effects observed in populations exposed to contaminated water. This includes reviewing medical records, conducting epidemiological studies, and monitoring health indicators such as morbidity (incidence of illness) and mortality (death rates). Long-term health impacts, such as chronic diseases or developmental issues, may also be assessed.')
    st.subheader('Social and Economic Implications: ')
    st.write('Assessing the broader social and economic consequences of water contamination. This includes evaluating impacts on livelihoods, productivity, healthcare costs, and community well-being. Consideration is given to both immediate effects and long-term implications for sustainable development.')

# Details Page
def details_page():
    st.title("Description ")
    st.subheader("pH:")
    st.write("pH: pH measures the acidity or alkalinity of water. The acceptable pH range for drinking water is generally between 6.5 and 8.5. Extreme pH levels can affect the taste of water and potentially indicate the presence of corrosive or scaling substances, which can impact plumbing and affect the effectiveness of water treatment processes.")
    st.markdown('<a style="color: blue;" href="https://en.m.wikipedia.org/wiki/pH">Read More</a>', unsafe_allow_html=True)

    st.subheader("Hardness:")
    st.write("Hardness: Hardness refers to the concentration of minerals, primarily calcium and magnesium, dissolved in water. While hard water is not usually a health concern, it can cause scaling in pipes and appliances, decrease soap effectiveness, and lead to skin and hair irritation. Acceptable hardness levels vary but are typically below 100-150 mg/L as calcium carbonate.")
    st.markdown('<a style="color: blue;" href="https://en.m.wikipedia.org/wiki/Hardness">Read More</a>', unsafe_allow_html=True)

    st.subheader("Chloramines:")
    st.write("Chloramines: Chloramines are disinfection byproducts formed when chlorine reacts with organic matter in water. They are used as a secondary disinfectant in some water treatment systems. While chloramines themselves are not highly toxic, excessive levels can cause skin irritation and respiratory issues in sensitive individuals. The maximum residual disinfectant level for chloramines in drinking water is typically around 4.0 mg/L.")
    st.markdown('<a style="color: blue;" href="https://en.m.wikipedia.org/wiki/Chloramine">Read More</a>', unsafe_allow_html=True)

    st.subheader("TDS (Total Dissolved Solids):")
    st.write("Total Dissolved Solids (TDS): TDS measures the total concentration of dissolved inorganic substances in water. Acceptable levels vary depending on the source and treatment of water but generally fall below 500 mg/L. Elevated TDS levels can affect the taste and appearance of water, and very high levels may indicate the presence of harmful contaminants.")
    st.markdown('<a style="color: blue;" href="https://en.wikipedia.org/wiki/Total_dissolved_solids">Read More</a>', unsafe_allow_html=True)

    st.subheader("Sulfates:")
    st.write("Sulfates: Sulfates are naturally occurring compounds found in soil and rocks. While they are not typically harmful at low levels, high sulfate concentrations (>250 mg/L) can cause gastrointestinal issues, particularly in infants and individuals with certain medical conditions.")
    st.markdown('<a style="color: blue;" href="https://en.m.wikipedia.org/wiki/Sulfate">Read More</a>', unsafe_allow_html=True)

    st.subheader("conductivity:")
    st.write("Conductivity: Conductivity measures the ability of water to conduct electrical current, which is influenced by dissolved ions. It is often used as an indicator of overall water quality. Acceptable conductivity levels for drinking water vary but generally fall below 800 μS/cm. Elevated conductivity can indicate the presence of dissolved salts or pollutants.")
    st.markdown('<a style="color: blue;" href="https://en.m.wikipedia.org/wiki/Conductivity">Read More</a>', unsafe_allow_html=True)

    st.subheader("organic carbon:")
    st.write("Organic Carbon: Organic carbon compounds can originate from natural sources or anthropogenic activities. High levels of organic carbon can contribute to taste and odor issues in water and serve as precursors to disinfection byproducts. Acceptable levels vary, but organic carbon concentrations are typically kept below 2-4 mg/L in treated drinking water.")
    st.markdown('<a style="color: blue;" href="https://en.m.wikipedia.org/wiki/Organic_carbon">Read More</a>', unsafe_allow_html=True)
    st.subheader("Trihalomethanes (THMs): ")
    st.write("Trihalomethanes (THMs): THMs are disinfection byproducts formed when chlorine reacts with organic matter in water. Long-term exposure to elevated THM levels (>80 μg/L) has been associated with increased cancer risk and adverse reproductive outcomes. Regulations typically limit THM levels in drinking water to below 80 μg/L.")
    st.markdown('<a style="color: blue;" href="https://en.m.wikipedia.org/wiki/Trihalomethanes">Read More</a>', unsafe_allow_html=True)

    st.subheader("Turbidity:")
    # st.link_button(":red[Turbidity]", "https://en.m.wikipedia.org/wiki/Turbidity")

    st.write("Turbidity: Turbidity measures the cloudiness or clarity of water caused by suspended particles. While not directly harmful, high turbidity can indicate the presence of pathogens, organic matter, or other contaminants. Acceptable turbidity levels for drinking water are typically below 1 NTU (Nephelometric Turbidity Units).")
    st.markdown('<a style="color: blue;" href="https://en.m.wikipedia.org/wiki/Turbidity">Read More</a>', unsafe_allow_html=True)

# Contact Us Page

def contact_page():
    st.title("Contact Us")
    st.subheader("Mentor Name")
    st.write("Dr. V. Anusuya Devi CSE")
    st.write("Email: v.anusuyadevi@klu.ac.in")
    st.subheader("Team Members")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("GODUGU VARAPARSAD")
        st.write("Reg no: 99210041857")
        st.write("Email: 99210041857@klu.ac.in")
    with col2:
        st.write("U.HARSHA VARDHAN")
        st.write('Reg no: 99210041788')
        st.write("Email: 99210041788@klu.ac.in")
    with col3:
        st.write('CH.V.V Sai Kumar Reddy')
        st.write('Reg no: 9921004847')
        st.write('Email: 9921004847@klu.ac.in')
    col4,col5,col6 = st.columns(3)
    with col4:
        st.write('A.V.MANOJ REDDY REDDY')
        st.write('Reg no: 9921004821')
        st.write('Email: 9921004821@klu.ac.in')
    with col5:
        st.write('P.VARUN TEJ')
        st.write('Reg no: 9921004592')
        st.write('Email: 9921004592@klu.ac.in')
    
def main():
    st.sidebar.title("Water Quality Analysis")
    st.sidebar.markdown("* * *")
    selection = st.sidebar.radio(" ", ["Home", "Water Potability Prediction", "Details", "Contact Us"])

    if selection == "Water Potability Prediction":
        water_potability_prediction()

    elif selection == "Home":
        Home_page()

    elif selection == "Details":
        details_page()

    elif selection == "Contact Us":
        contact_page()
    st.sidebar.markdown("---")
    st.sidebar.write("© 2024 created by varaprasad and team. All rights reserved.")

if __name__ == '__main__':
    main()
