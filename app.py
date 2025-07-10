import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and clean data
try:
    data = pd.read_csv("sales_data.csv", delimiter="\t")  # use ',' if it's comma-separated
    data.columns = data.columns.str.strip()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


st.markdown("""
# 🍽️ Hyderabad Food Sales Predictor
Use this **AI-powered tool** to estimate food sales and reduce waste in your kitchen.
""")


# Clean column names
data.columns = data.columns.str.strip()


# Stop if dataset is empty
if data.empty:
    st.error("❌ Your dataset is empty after cleaning. Please check your CSV file.")
    st.stop()

# Encode categorical columns
for col in ['Day', 'Weather', 'Holiday', 'FoodItem']:
    data[col] = data[col].astype('category')

data['DayCode'] = data['Day'].cat.codes
data['WeatherCode'] = data['Weather'].cat.codes
data['HolidayCode'] = data['Holiday'].cat.codes
data['FoodCode'] = data['FoodItem'].cat.codes

# Features and target
X = data[['DayCode', 'WeatherCode', 'HolidayCode', 'FoodCode']]
y = data['QuantitySold']

# Train the model
model = LinearRegression()
model.fit(X, y)

# --- Streamlit UI ---

st.subheader("📋 Enter Today's Details")

day = st.selectbox("📅 Select Day", data['Day'].cat.categories)
weather = st.selectbox("🌤️ Select Weather", data['Weather'].cat.categories)
holiday = st.selectbox("🏖️ Is it a Holiday?", data['Holiday'].cat.categories)
food = st.selectbox("🍛 Select Food Item", data['FoodItem'].cat.categories)

# Convert user input to codes
day_code = data['Day'].cat.categories.get_loc(day)
weather_code = data['Weather'].cat.categories.get_loc(weather)
holiday_code = data['Holiday'].cat.categories.get_loc(holiday)
food_code = data['FoodItem'].cat.categories.get_loc(food)
data.columns = data.columns.str.strip()

# Predict button
if st.button("🔮 Predict Sales"):
    prediction = model.predict([[day_code, weather_code, holiday_code, food_code]])[0]
    st.success(
        f"🤖 On **{day}**, with **{weather}** weather and "
        f"{'a holiday' if holiday == 'Yes' else 'a working day'}, "
        f"you can expect to sell approximately **{int(prediction)} units** of **{food}**."
    )
