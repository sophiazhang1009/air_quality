import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import numpy as np
from PIL import Image

st.title("Air Quality Trend and Forecasting System")
st.write("Forecasting the future statistics and health impacts of air pollutants using machine learning.")

df = pd.read_csv("data/air_quality.csv")

st.subheader("Select Parameters")

enough_data = []

pollutant = st.selectbox("Select a pollutant", ["PM2.5 (μg/m3)", "PM10 (μg/m3)", "NO2 (μg/m3)"])

for city in df["City or Locality"].unique():
    city_data = df[df["City or Locality"] == city]
    yearly_data = city_data.groupby("Version of the database")[pollutant].mean().dropna()

    if len(yearly_data) >= 2:
        enough_data.append(city)

city = st.selectbox("Select a city", enough_data)

city_data = df[df["City or Locality"] == city]

yearly_data = city_data.groupby("Version of the database")[pollutant].mean().reset_index()

st.subheader(f"Historical {pollutant} Levels")

plotly_chart = st.plotly_chart(px.line(yearly_data, x="Version of the database", y=pollutant, title=f"{pollutant} Levels in {city}"))
model = LinearRegression()
city_data = city_data.reset_index(drop=True)

yearly_data = yearly_data.dropna(subset=[pollutant])

X = yearly_data["Version of the database"].values.reshape(-1, 1)
y = yearly_data[pollutant].values

st.write(f"Linear reggression was used to allow observation of long-term trends from air pollutant concentrations over time. This model shows a clear visual representation of the general increase and decrease in air pollutant levels in {city}.")

years_ahead = st.slider(
    "Years into the future", 0, 24)

model.fit(X,y)

future_year = yearly_data["Version of the database"].max() + years_ahead
predicted_value = model.predict([[future_year+4]])[0]
predicted_value = max(0, predicted_value)

if years_ahead == 0:
    st.write(f"The predicted {pollutant} level in {city} for the remainder of 2026 is: {predicted_value:.2f}.")

else:
    st.write(f"The predicted {pollutant} level in {city} in {future_year+4} is: {predicted_value:.2f}")

st.subheader("Health Impact")
st.write("How does air quality affect the human respiratory systems and cardiovascular health?")

prediction_df = pd.DataFrame({
    "Version of the database": [future_year+4],
    pollutant: [predicted_value]
})

#almost done, add some more and then finish the app tomorrow so there's time to figure out how to deploy it and stuff!!

if pollutant == "PM2.5 (μg/m3)":
    if predicted_value == 5:
        st.write(f"**The predicted PM2.5 level is 5 μg/m³ in {city}, which indicates stable air quality.**")

    elif predicted_value < 5:
        st.write(f"**The predicted PM2.5 level is below 5 μg/m³ in {city}, which is considered healthy air quality.**")

    elif predicted_value >= 15:
        st.write(f"**The predicted PM2.5 level is above 15 μg/m³ in {city}, which indicates extremely poor air quality and a serious health risk to residents and citizens. It is considered extremely dangerous to live in such conditions, with many risks of cardiovascular and respiratory disease.**")
        with st.expander("What is PM2.5?"):
            st.write("Fine particulate matter is one of the most major consequences of air pollution and likely the most fatal. They are often the cause of damaged cells and tissues in the human body and a risk to lung cancer and other respiratory conditions. The size of the particles are the direct cause of extreme health consequences from fine particulate matter, as they easily travel through the lungs and potentially enter bloodstreams.")
        with st.expander("Health Effects of PM2.5"):
            st.write("According to the World Health Organization, this small size also allows many air pollutants, including fine particulate matter, to damage almost every organ in the body, consequently leading to increased risk of systemic inflammation or carcinogenicity. As a result, PM2.5 can lead to premature death, heart disease, irregular heartbeat, higher risk of asthma, decreased lung function, and increased symptoms of respiratory diseases or difficulty breathing. The consequences of PM2.5 are also a cause of eye, nose, and throat irritation, lung and respiratory conditions, and increased risk of chronic obstructive pulmonary disease, asthma, and cardiovascular diseases.")
            st.write("Not only does this hold significant consequence for the respiratory and cardiovascular systems, but PM2.5 exposure has also been linked to adverse pregnancy outcomes, including low birth weight, preterm birth, and developmental delays in children. PM2.5 is also a leading cause of ocular health decrease, with its small size allowing it to enter the ocular tissues, causing cellular damage and inflammation. This leads to increased risk of ocular surface damage and conditions such as conjunctivitis, which is the inflammation of the clear membrane covering the white part of the eye, and dry eye syndrome.")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image("data/air_image.jpg", caption="Respiratory System Affected by Air Pollution", width=300)
            with col2:
                st.write("This image illustrates the size of PM2.5 particles in comparison to the human hair, and the effect that these particles can have on the lungs and respiratory system when inhaled. As shown in the image, PM2.5 enters the interior airways and travels deep into the alveoli sacs, which enter bloodstreams in the body.")
        with st.expander("Long-term Exposure Effects"):
            st.write("Long-term exposure to PM2.5 has also been linked to the development of chronic respiratory diseases in children, reduced lung development, and increased risk of acute lower respiratory infections. Vulnerable populations, such as children, the elderly, and individuals with pre-existing health conditions, are particularly susceptible to the adverse health effects of PM2.5 exposure.")
    
    elif 15 > predicted_value >= 10:
        st.write(f"**The predicted PM2.5 level is above 10 μg/m³ in {city}, which indicates a potential health risk to residents and citizens.**")
        with st.expander("What is PM2.5?"):
            st.write("Fine particulate matter is one of the most major consequences of air pollution and likely the most fatal. They are often the cause of damaged cells and tissues in the human body and a risk to lung cancer and other respiratory conditions. The size of the particles are the direct cause of extreme health consequences from fine particulate matter, as they easily travel through the lungs and potentially enter bloodstreams.")
        with st.expander("Health Effects of PM2.5"):
            st.write("According to the World Health Organization, this small size also allows many air pollutants, including fine particulate matter, to damage almost every organ in the body, consequently leading to increased risk of systemic inflammation or carcinogenicity. As a result, PM2.5 can lead to premature death, heart disease, irregular heartbeat, higher risk of asthma, decreased lung function, and increased symptoms of respiratory diseases or difficulty breathing. The consequences of PM2.5 are also a cause of eye, nose, and throat irritation, lung and respiratory conditions, and increased risk of chronic obstructive pulmonary disease, asthma, and cardiovascular diseases.")
            st.write("Not only does this hold significant consequence for the respiratory and cardiovascular systems, but PM2.5 exposure has also been linked to adverse pregnancy outcomes, including low birth weight, preterm birth, and developmental delays in children. PM2.5 is also a leading cause of ocular health decrease, with its small size allowing it to enter the ocular tissues, causing cellular damage and inflammation. This leads to increased risk of ocular surface damage and conditions such as conjunctivitis, which is the inflammation of the clear membrane covering the white part of the eye, and dry eye syndrome.")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image("data/air_image.jpg", caption="Respiratory System Affected by Air Pollution", width=300)
            with col2:
                st.write("This image illustrates the size of PM2.5 particles in comparison to the human hair, and the effect that these particles can have on the lungs and respiratory system when inhaled. As shown in the image, PM2.5 enters the interior airways and travels deep into the alveoli sacs, which enter bloodstreams in the body.")
        with st.expander("Long-term Exposure Effects"):
            st.write("Long-term exposure to PM2.5 has also been linked to the development of chronic respiratory diseases in children, reduced lung development, and increased risk of acute lower respiratory infections. Vulnerable populations, such as children, the elderly, and individuals with pre-existing health conditions, are particularly susceptible to the adverse health effects of PM2.5 exposure.")

    elif 10 > predicted_value >= 5:
        st.write(f"**The predicted PM2.5 level is above 5 μg/m³ in {city}, which indicates a potential health risk to residents and citizens.**")
        with st.expander("What is PM2.5?"):
            st.write("Fine particulate matter is one of the most major consequences of air pollution and likely the most fatal. They are often the cause of damaged cells and tissues in the human body and a risk to lung cancer and other respiratory conditions. The size of the particles are the direct cause of extreme health consequences from fine particulate matter, as they easily travel through the lungs and potentially enter bloodstreams.")
        with st.expander("Health Effects of PM2.5"):
            st.write("According to the World Health Organization, this small size also allows many air pollutants, including fine particulate matter, to damage almost every organ in the body, consequently leading to increased risk of systemic inflammation or carcinogenicity. As a result, PM2.5 can lead to premature death, heart disease, irregular heartbeat, higher risk of asthma, decreased lung function, and increased symptoms of respiratory diseases or difficulty breathing. The consequences of PM2.5 are also a cause of eye, nose, and throat irritation, lung and respiratory conditions, and increased risk of chronic obstructive pulmonary disease, asthma, and cardiovascular diseases.")
            st.write("Not only does this hold significant consequence for the respiratory and cardiovascular systems, but PM2.5 exposure has also been linked to adverse pregnancy outcomes, including low birth weight, preterm birth, and developmental delays in children. PM2.5 is also a leading cause of ocular health decrease, with its small size allowing it to enter the ocular tissues, causing cellular damage and inflammation. This leads to increased risk of ocular surface damage and conditions such as conjunctivitis, which is the inflammation of the clear membrane covering the white part of the eye, and dry eye syndrome.")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image("data/air_image.jpg", caption="Respiratory System Affected by Particulate Matter", width=300)
            with col2:
                st.write("This image illustrates the size of PM2.5 particles in comparison to the human hair, and the effect that these particles can have on the lungs and respiratory system when inhaled. As shown in the image, PM2.5 enters the interior airways and travels deep into the alveoli sacs, which enter bloodstreams in the body.")
        with st.expander("Long-term Exposure Effects"):
            st.write("Long-term exposure to PM2.5 has also been linked to the development of chronic respiratory diseases in children, reduced lung development, and increased risk of acute lower respiratory infections. Vulnerable populations, such as children, the elderly, and individuals with pre-existing health conditions, are particularly susceptible to the adverse health effects of PM2.5 exposure.")


if pollutant == "PM10 (μg/m3)":
    if predicted_value == 15:
        st.write(f"**The predicted PM10 level is 20 μg/m³ in {city}, which is considered stable.**")
    elif predicted_value < 15:
        st.write(f"**The predicted PM10 level is below 20 μg/m³ in {city}, which is considered healthy air quality.**")
    elif predicted_value >= 15:
        st.write(f"**The predicted PM10 level is above 15 μg/m³ in {city}, which indicates poor air quality.**")
        with st.expander("What is PM10?"):
            st.write("Particulate Matter 10 (PM10) refers to inhalable particles with diameters that are generally 10 micrometers and smaller. These particles can include dust, pollen, mold, and other airborne substances. Due to their small size, PM10 particles can penetrate the respiratory system and reach the lungs, potentially causing various health issues. While generally considered less harmful than PM2.5, it is a significant health concern, especially for individuals with pre-existing respiratory conditions, children, and the elderly.")
        with st.expander("Health Effects of PM10"):
            st.write("Exposure to elevated levels of PM10 can lead to a range of health problems, particularly affecting the respiratory system. Short-term exposure may cause irritation of the eyes, nose, and throat, coughing, and shortness of breath. It can also exacerbate existing respiratory conditions such as asthma and bronchitis. Long-term exposure to high levels of PM10 has been associated with chronic respiratory diseases, reduced lung function, and increased risk of cardiovascular diseases. Vulnerable populations, including children, the elderly, and individuals with pre-existing health conditions, are particularly susceptible to the adverse effects of PM10 exposure.")
            st.write("In addition to respiratory issues, PM10 exposure has been linked to other health problems such as cardiovascular diseases, including heart attacks and strokes. The particles can enter the bloodstream through the lungs, leading to systemic inflammation and oxidative stress, which can contribute to the development of atherosclerosis (hardening of the arteries) and other cardiovascular conditions.")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image("data/air_image.jpg", caption="Respiratory System Affected by Particulate Matter", width=300)
            with col2:
                st.write("This image illustrates the size of PM10 particles in comparison to the human hair, and the effect that these particles can have on the lungs and respiratory system when inhaled. As shown in the image, PM10 enters the interior airways and can reach the lungs, potentially causing various health issues.")
        with st.expander("Long-term Exposure Effects"):
            st.write("Long-term exposure to PM10 and other particulate matter has been linked to the development of chronic respiratory diseases in children, reduced lung development, and increased risk of acute lower respiratory infections. Vulnerable populations, such as children, the elderly, and individuals with pre-existing health conditions, are particularly susceptible to the adverse health effects of PM10 exposure.")

if pollutant == "NO2 (μg/m3)":
    if predicted_value == 10:
        st.write(f"**The predicted NO2 level is 10 μg/m³ in {city}, which is considered stable.**")

    elif predicted_value < 10:
        st.write(f"**The predicted NO2 level is below 10 μg/m³ in {city}, which is considered healthy air quality.**")

    elif 40 >= predicted_value >= 10:
        st.write(f"**The predicted NO2 level is above 10 μg/m³ in {city}, which indicates air quality levels exceeding the annual mean rate of the World Health Organization (WHO) recommendations.**")
        st.write("However, in current conditions, many cities have exceeded these limits. This led to the limit being altered later to the higher risk of 40 μg/m³, making this prediction close to the current average of nitrogen dioxide rates, while the predicted rate still poses high health risks.")
        with st.expander("What is NO2?"):
            st.write("Nitrogen Dioxide (NO2) is a reddish-brown gas with a characteristic sharp, biting odor and is a prominent air pollutant. It is primarily produced from the combustion of fossil fuels, such as those used in vehicles, power plants, and industrial processes. NO2 is a member of the nitrogen oxides (NOx) family and plays a significant role in the formation of ground-level ozone and particulate matter, both of which have adverse effects on human health and the environment.")
        with st.expander("Health Effects of NO2"):
            st.write("Exposure to elevated levels of nitrogen dioxide (NO2) can lead to increased risk of respiratory diseases or symptoms, particularly in children, the elderly, and individuals with pre-existing respiratory conditions.")
            st.write("Exposure can lead to shortness of breath, coughing, wheezing, and increased susceptibility to respiratory infections. It can also lead to formation of smog (ground-level ozone) and acid rain, which can lead to further risk of decreased lung function and increase risk of respiratory illnesses.")
            st.write("Not only does NO2 have significant consequences for the human body, it also has detrimental effects on the environment. As a leading cause of acid rain, it affects nutrients in soils and plantlife, as well as lowering pH in bodies of water, which deteriorates aquatic food sources. This can potentially lead to further decline of population in aquatic animals and other wildlife, as well as lowering pH levels in aquatic environments.")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image("data/air_image_2.jpg", caption="Unaffected Airway versus Asthmatic Airway", width=300)
            with col2:
                st.write("As a strong respiratory irritant, nitrogen dioxide causes acute inflammation in the airways, making it a leading cause of asthmatic airways. This image illustrates the comparison between an unaffected airway and an asthmatic airway as a result of exposure to nitrogen dioxides. As shown in the image, asthmatic airways have increased risk of tightened smooth muscles, an inflamed, swollen wall, and extra mucus.")
        with st.expander("Long-term Exposure Effects"):
            st.write("Long-term exposure to elevated levels of nitrogen dioxide (NO2) has been associated with chronic respiratory diseases, reduced lung function, and increased mortality rates. Studies have revealed that NO2 long-term exposure can also correlate with the development or worsening of asthma, especially in children.")


    elif predicted_value >= 40:
        st.write(f"**The predicted NO2 level is above 40 μg/m³ in {city}, which indicates poor air quality and a serious health risk to residents and citizens. Prolonged exposure to such high levels of nitrogen dioxide can lead to significant respiratory issues and other health complications.**")
        with st.expander("What is NO2?"):
            st.write("Nitrogen Dioxide (NO2) is a reddish-brown gas with a characteristic sharp, biting odor and is a prominent air pollutant. It is primarily produced from the combustion of fossil fuels, such as those used in vehicles, power plants, and industrial processes. NO2 is a member of the nitrogen oxides (NOx) family and plays a significant role in the formation of ground-level ozone and particulate matter, both of which have adverse effects on human health and the environment.")
        with st.expander("Health Effects of NO2"):
            st.write("Exposure to elevated levels of nitrogen dioxide (NO2) can lead to increased risk of respiratory diseases or symptoms, particularly in children, the elderly, and individuals with pre-existing respiratory conditions.")
            st.write("Exposure can lead to shortness of breath, coughing, wheezing, and increased susceptibility to respiratory infections. It can also lead to formation of smog (ground-level ozone) and acid rain, which can lead to further risk of decreased lung function and increase risk of respiratory illnesses.")
            st.write("Not only does NO2 have significant consequences for the human body, it also has detrimental effects on the environment. As a leading cause of acid rain, it affects nutrients in soils and plantlife, as well as lowering pH in bodies of water, which deteriorates aquatic food sources. This can potentially lead to further decline of population in aquatic animals and other wildlife, as well as lowering pH levels in aquatic environments.")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image("data/air_image_2.jpg", caption="Unaffected Airway versus Asthmatic Airway", width=300)
            with col2:
                st.write("As a strong respiratory irritant, nitrogen dioxide causes acute inflammation in the airways, making it a leading cause of asthmatic airways. This image illustrates the comparison between an unaffected airway and an asthmatic airway as a result of exposure to nitrogen dioxides. As shown in the image, asthmatic airways have increased risk of tightened smooth muscles, an inflamed, swollen wall, and extra mucus.")
        with st.expander("Long-term Exposure Effects"):
            st.write("Long-term exposure to elevated levels of nitrogen dioxide (NO2) has been associated with chronic respiratory diseases, reduced lung function, and increased mortality rates. Studies have revealed that NO2 long-term exposure can also correlate with the development or worsening of asthma, especially in children.")

#YAY IT FINALLY WORKS :SOBOFHAPPINESS
st.caption("Developed by Sophia Zhang | Data Source: World Health Organization Global Air Quality Database")