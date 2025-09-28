import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load model, scaler, and columns
model = pickle.load(open(os.path.join("models", "mlp_model.sav"), "rb"))
scaler = pickle.load(open(os.path.join("models", "scaler.sav"), "rb"))
columns = pickle.load(open(os.path.join("models", "columns.sav"), "rb"))
if os.path.exists(model) and os.path.exists(scaler) and os.path.exists(columns):
    model = pickle.load(open(model, "rb"))
    scaler = pickle.load(open(scaler, "rb"))
    columns = pickle.load(open(columns, "rb"))
else:
    st.error("❌ One or more model files are missing. Please check your 'models/' folder.")
st.set_page_config(page_title="Carbon Footprint Predictor", page_icon="🌱")
st.title("🌍 Carbon Footprint Predictor")
st.write("Estimate your monthly carbon footprint based on lifestyle inputs.")

# User Inputs
gender = st.selectbox("Gender", ["male", "female"])
diet = st.selectbox("Diet Type", ["vegan", "vegetarian", "pescatarian", "omnivore"])
frequency_travel = st.selectbox("Frequency of Traveling by Air", ["never", "rarely", "frequently", "very frequently"])
energy_source = st.selectbox("Heating Energy Source", ["electricity", "coal", "wood", "natural gas"])
transport_mode = st.selectbox("Transport Mode", ["walk/bicycle", "public", "private"])
fuel_type = st.selectbox("Vehicle Fuel Type", ["petrol", "diesel", "lpg", "hybrid", "electric", "None"])
monthly_distance = st.number_input("Monthly Distance by Vehicle (km)", min_value=0)
waste_bag_size = st.selectbox("Waste Bag Size", ["small", "medium", "large", "extra large"])
waste_bag_count = st.number_input("Waste Bag Weekly Count", min_value=0)
tv_pc_hours = st.number_input("Daily TV/PC Usage (hours)", min_value=0)
clothes_monthly = st.number_input("New Clothes Bought Monthly", min_value=0)
internet_hours = st.number_input("Daily Internet Usage (hours)", min_value=0)
energy_efficiency = st.selectbox("Energy Efficient Appliances", ["Yes", "No", "Sometimes"])
recycling = st.multiselect("Recycling Materials", ["Paper", "Plastic", "Glass", "Metal"])
cooking_methods = st.multiselect("Cooking Methods Used", ["Stove", "Oven", "Microwave", "Grill", "Airfryer"])
shower_freq = st.selectbox("How Often Do You Shower?", ["daily", "twice a day", "more frequently", "less frequently"])
grocery_bill = st.number_input("Monthly Grocery Spending ($)", min_value=0)

# Prepare input dictionary
data = {
    "Gender": gender,
    "Diet Type": diet,
    "Frequency of Traveling by Air": frequency_travel,
    "Heating Energy Source": energy_source,
    "Transport Mode": transport_mode,
    "Vehicle Fuel Type": fuel_type,
    "Monthly Distance Km": monthly_distance,
    "Waste Bag Size": waste_bag_size,
    "Waste Bag Weekly Count": waste_bag_count,
    "How Long TV PC Daily Hour": tv_pc_hours,
    "How Many New Clothes Monthly": clothes_monthly,
    "How Long Internet Daily Hour": internet_hours,
    "Energy efficiency": energy_efficiency,
    "How Often Shower": shower_freq,
    "Monthly Grocery Bill": grocery_bill
}

# Add one-hot encoded fields for recycling and cooking
data.update({f"Recycling_{item}": 1 for item in recycling})
data.update({f"Cooking_With_{item}": 1 for item in cooking_methods})

# Predict
if st.button("Predict My Carbon Footprint"):
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = round(model.predict(scaler.transform(input_df))[0])
    st.subheader(f"🌡️ Estimated Monthly Carbon Emission: {prediction} kg CO₂")

    tree_count = round(prediction / 411.4)
    st.markdown(f"""You owe nature <b>{tree_count}</b> tree{'s' if tree_count > 1 else ''} monthly.""", unsafe_allow_html=True)
    

    st.markdown("### 🌿 Tips to Reduce Your Footprint")
    st.markdown("- Use public transport or walk more")
    st.markdown("- Switch to renewable energy sources")
    st.markdown("- Reduce meat consumption")
    st.markdown("- Recycle consistently and buy fewer new clothes")
