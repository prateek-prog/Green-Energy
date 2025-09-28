import streamlit as st
import pandas as pd
import numpy as np
from streamlit.components.v1 import html
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import base64
from functions import *

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Carbon Footprint Calculator", page_icon="./media/favicon.ico")

# --- Background & Styling ---
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

background = get_base64("./media/background_min.jpg")
icon2 = get_base64("./media/icon2.png")
icon3 = get_base64("./media/icon3.png")

with open("./style/style.css", "r") as style:
    css = f"""<style>{style.read().format(background=background, icon2=icon2, icon3=icon3)}</style>"""
    st.markdown(css, unsafe_allow_html=True)

def script():
    with open("./style/scripts.js", "r", encoding="utf-8") as scripts:
        open_script = f"""<script>{scripts.read()}</script>"""
        html(open_script, width=0, height=0)

# --- Layout ---
left, middle, right = st.columns([2, 3.5, 2])
main, comps, result = middle.tabs([" ", " ", " "])

with open("./style/main.md", "r", encoding="utf-8") as main_page:
    main.markdown(main_page.read())

_, but, _ = main.columns([1, 2, 1])
if but.button("Calculate Your Carbon Footprint!", type="primary"):
    click_element('tab-1')

tab1, tab2, tab3, tab4, tab5 = comps.tabs(["👴 Personal", "🚗 Travel", "🗑️ Waste", "⚡ Energy", "💸 Consumption"])
tab_result, _ = result.tabs([" ", " "])

# --- Input Component ---
def component():
    tab1col1, tab1col2 = tab1.columns(2)
    height = tab1col1.number_input("Height", 0, 251, value=None, placeholder="160", help="in cm")
    weight = tab1col2.number_input("Weight", 0, 250, value=None, placeholder="75", help="in kg")
    height = height if height else 1
    weight = weight if weight else 1
    bmi = weight / (height / 100) ** 2
    body_type = "underweight" if bmi < 18.5 else "normal" if bmi < 25 else "overweight" if bmi < 30 else "obese"

    sex = tab1.selectbox('Gender', ["female", "male"])
    diet = tab1.selectbox('Diet', ['omnivore', 'pescatarian', 'vegetarian', 'vegan'])
    social = tab1.selectbox('Social Activity', ['never', 'often', 'sometimes'])

    transport = tab2.selectbox('Transportation', ['public', 'private', 'walk/bicycle'])
    vehicle_type = tab2.selectbox('Vehicle Type', ['petrol', 'diesel', 'hybrid', 'lpg', 'electric']) if transport == "private" else "None"
    vehicle_km = 0 if transport == "walk/bicycle" else tab2.slider('Monthly vehicle distance (km)', 0, 5000, 0)

    air_travel = tab2.selectbox('Air travel last month', ['never', 'rarely', 'frequently', 'very frequently'])

    waste_bag = tab3.selectbox('Waste bag size', ['small', 'medium', 'large', 'extra large'])
    waste_count = tab3.slider('Waste bags per week', 0, 10, 0)
    recycle = tab3.multiselect('Recycled materials', ['Plastic', 'Paper', 'Metal', 'Glass'])

    heating_energy = tab4.selectbox('Heating source', ['natural gas', 'electricity', 'wood', 'coal'])
    for_cooking = tab4.multiselect('Cooking systems', ['microwave', 'oven', 'grill', 'airfryer', 'stove'])
    energy_efficiency = tab4.selectbox('Energy-efficient devices?', ['No', 'Yes', 'Sometimes'])
    daily_tv_pc = tab4.slider('Daily PC/TV hours', 0, 24, 0)
    internet_daily = tab4.slider('Daily internet hours', 0, 24, 0)

    shower = tab5.selectbox('Shower frequency', ['daily', 'twice a day', 'more frequently', 'less frequently'])
    grocery_bill = tab5.slider('Monthly grocery spending ($)', 0, 500, 0)
    clothes_monthly = tab5.slider('Monthly clothes bought', 0, 30, 0)

    data = {
        'Body Type': body_type,
        "Sex": sex,
        'Diet': diet,
        "How Often Shower": shower,
        "Heating Energy Source": heating_energy,
        "Transport": transport,
        "Social Activity": social,
        'Monthly Grocery Bill': grocery_bill,
        "Frequency of Traveling by Air": air_travel,
        "Vehicle Monthly Distance Km": vehicle_km,
        "Waste Bag Size": waste_bag,
        "Waste Bag Weekly Count": waste_count,
        "How Long TV PC Daily Hour": daily_tv_pc,
        "Vehicle Type": vehicle_type,
        "How Many New Clothes Monthly": clothes_monthly,
        "How Long Internet Daily Hour": internet_daily,
        "Energy efficiency": energy_efficiency
    }

    data.update({f"Cooking_with_{x}": 1 for x in for_cooking})
    data.update({f"Do You Recyle_{x}": 1 for x in recycle})

    return pd.DataFrame(data, index=[0])

# --- Data Preparation ---
df = component()
data = input_preprocessing(df)

# --- Load Model & Scaler ---
scale_path = "./models/scale.joblib"
model_path = "./models/model.joblib"

if os.path.exists(scale_path) and os.path.exists(model_path):
    ss = joblib.load(scale_path)
    model = joblib.load(model_path)
else:
    st.error("❌ Model or scaler file missing.")
    st.stop()

# --- Prediction ---
try:
    sample_df = pd.DataFrame(data, index=[0])
    prediction = round(np.exp(model.predict(ss.transform(sample_df))[0]))
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# --- Result Display ---
column1, column2 = tab1.columns(2)
_, resultbutton, _ = tab5.columns([1, 1, 1])
if resultbutton.button(" ", type="secondary"):
    tab_result.image(chart(model, ss, sample_df, prediction), use_column_width="auto")
    click_element('tab-2')

# --- Did You Know Popup ---
pop_button = """<button id="button-17" class="button-17" role="button"> ❔ Did You Know</button>"""
_, home, _ = comps.columns([1, 2, 1])
_, col2, _ = comps.columns([1, 10, 1])
col2.markdown(pop_button, unsafe_allow_html=True)

pop = """
<div id="popup" class="DidYouKnow_root">
<p class="DidYouKnow_title TextNew" style="font-size: 20px;"> ❔ Did you know</p>
<p id="popupText" class="DidYouKnow_content TextNew"><span>
Each year, human activities release over 40 billion metric tons of carbon dioxide into the atmosphere, contributing to climate change.
</span></p>
</div>
"""
col2.markdown(pop, unsafe_allow_html=True)

if home.button("🏡"):
    click_element('tab-0')

_, resultmid, _ = result.columns([1, 2, 1])
tree_count = round(prediction / 411.4)
tab_result.markdown(f"""You owe nature <b>{tree_count}</b> tree{'s' if tree_count > 1 else ''} monthly.<br>
{f"<a href='https://www.tema.org.tr/en/homepage' id='button-17' class='button-17' role='button'> 🌳 Proceed to offset 🌳</a>" if tree_count > 0 else ""}""", unsafe_allow_html=True)

if resultmid.button("  ", type="secondary"):
    click_element('tab-1')

# --- Footer ---
with open("./style/footer.html", "r", encoding="utf-8") as footer:
    footer_html = footer.read()
    st.markdown(footer_html, unsafe_allow_html=True)

script()
