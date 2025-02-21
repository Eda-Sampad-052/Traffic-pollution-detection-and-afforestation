import streamlit as st
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from io import BytesIO

# Plant suggestions with pollution reduction rates (in g/m²)
PLANT_SUGGESTIONS = {
    "Low": {"plants": "Lavender, Aloe Vera, Snake Plant", "reduction": 5},
    "Moderate": {"plants": "Spider Plant, Peace Lily, Bamboo Palm", "reduction": 10},
    "High": {"plants": "Areca Palm, Boston Fern, Rubber Plant", "reduction": 20},
    "Severe": {"plants": "Neem Tree, Peepal Tree, Banyan Tree", "reduction": 30}
}

EMISSION_FACTORS = {"CO2": 120, "NOx": 0.6, "PM2.5": 0.005}

# Function to classify pollution levels
def get_pollution_level(vehicle_count):
    if vehicle_count == 0:
        return "Low"
    elif vehicle_count <= 5:
        return "Moderate"
    elif vehicle_count <= 10:
        return "High"
    else:
        return "Severe"

# Function to calculate green area
def calculate_green_area(image, scale=0.1):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_open_green = np.array([35, 30, 30])
    upper_open_green = np.array([85, 200, 200])
    open_green_mask = cv2.inRange(hsv, lower_open_green, upper_open_green)

    contours, _ = cv2.findContours(open_green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area_m2 = sum(cv2.contourArea(cnt) * scale**2 for cnt in contours if cv2.contourArea(cnt) > 0.05)

    return total_area_m2

# Streamlit Interface
st.set_page_config(page_title='Vehicle Pollution Detection', layout='wide')

st.title('🌱 Vehicle Pollution Detection and Plant Suggestion 🌱')
uploaded_file = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    with st.spinner('Processing video... 💾'):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        car_cascade = cv2.CascadeClassifier('haarcascade_cars.xml')

        frame_count = 0
        total_green_area_m2 = 0
        total_emissions = {"CO2": 0, "NOx": 0, "PM2.5": 0}

        video_frames = []

        while cap.isOpened():
            ret, frames = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            cars = car_cascade.detectMultiScale(gray, 1.1, 3)
            vehicle_count = len(cars)

            for pollutant, factor in EMISSION_FACTORS.items():
                total_emissions[pollutant] += vehicle_count * factor

            frame_count += 1
            total_green_area_m2 = calculate_green_area(frames)

            for (x, y, w, h) in cars:
                cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)

            video_frames.append(frames)

        cap.release()

        if frame_count > 0:
            pollution_level = get_pollution_level(frame_count)
            suggested_plants = PLANT_SUGGESTIONS[pollution_level]["plants"]
            reduction_per_plant = PLANT_SUGGESTIONS[pollution_level]["reduction"]

            plant_distance_m = 3
            plants_count = int(total_green_area_m2 / (plant_distance_m ** 2))

            avg_emissions = {k: v / frame_count for k, v in total_emissions.items()}
            total_reduction = plants_count * reduction_per_plant

            st.success('✅ Video processed successfully!')

            st.subheader('📹 Video Analysis')
            video_placeholder = st.empty()

            for frame in video_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels='RGB')

            col1, col2 = st.columns(2)

            with col1:
                st.subheader('📊 Results')
                st.metric(label="Total Green Area", value=f"{total_green_area_m2:.2f} m²")
                st.metric(label="Plants Count", value=f"{plants_count} (3m apart)")
                st.metric(label="Total Pollution Reduction", value=f"{total_reduction} g/m²")

            with col2:
                st.subheader('🌿 Plant Suggestions')
                st.write(f"**Suggested Plants:** {suggested_plants}")
                st.write(f"**Pollution Level:** {pollution_level}")
                st.write(f"**Reduction per Plant:** {reduction_per_plant} g/m²")

            st.subheader('💨 Emission Averages')
            st.write(f"**Avg CO2:** {avg_emissions['CO2']:.2f} g/m²")
            st.write(f"**Avg NOx:** {avg_emissions['NOx']:.2f} g/m²")
            st.write(f"**Avg PM2.5:** {avg_emissions['PM2.5']:.2f} g/m²")

        else:
            st.error("❗ No frames detected in the video.")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption('🚗🌱 Developed by C.V.M. CHATURVEDI, E.SAMPAD and A.PRAMAY MOHAN using Streamlit, OpenCV, NumPy, and Matplotlib 🌱🚗')