import datetime as dt
import json
import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="DT",
    layout="wide",
)


MODEL_DIR = Path("best_models")
FEATURE_INFO_PATH = MODEL_DIR / "feature_info.json"
MAX_DISTANCE_KM = 50

WEATHER_MAP = {
    "Cloudy": "Cloudy",
    "Fog": "Fog",
    "Sandstorms": "Sandstorms",
    "Stormy": "Stormy",
    "Sunny": "Sunny",
    "Windy": "Windy",
}

TRAFFIC_MAP = {
    "High": "High ",
    "Jam": "Jam ",
    "Low": "Low ",
    "Medium": "Medium ",
}

VEHICLE_MAP = {
    "Bicycle": "bicycle ",
    "Motorcycle": "motorcycle ",
    "Scooter": "scooter ",
    "Van": "van",
}

AREA_MAP = {
    "Metropolitian": "Metropolitian ",
    "Other": "Other",
    "Semi-Urban": "Semi-Urban ",
    "Urban": "Urban ",
}

CATEGORY_MAP = {
    "Apparel": "Apparel",
    "Books": "Books",
    "Clothing": "Clothing",
    "Cosmetics": "Cosmetics",
    "Electronics": "Electronics",
    "Grocery": "Grocery",
    "Home": "Home",
    "Jewelry": "Jewelry",
    "Kitchen": "Kitchen",
    "Outdoors": "Outdoors",
    "Pet Supplies": "Pet Supplies",
    "Shoes": "Shoes",
    "Skincare": "Skincare",
    "Snacks": "Snacks",
    "Sports": "Sports",
    "Toys": "Toys",
}

SAMPLE_ORDERS = {
    "Balanced evening order": {
        "agent_age": 30,
        "agent_rating": 4.5,
        "prep_time": 12,
        "order_time": dt.time(hour=18, minute=30),
        "store_lat": 22.745049,
        "store_lon": 75.892471,
        "drop_lat": 22.765049,
        "drop_lon": 75.912471,
        "weather": "Sunny",
        "traffic": "Jam",
        "vehicle": "Motorcycle",
        "area": "Metropolitian",
        "category": "Clothing",
    },
    "Low-risk short route": {
        "agent_age": 34,
        "agent_rating": 4.8,
        "prep_time": 8,
        "order_time": dt.time(hour=11, minute=0),
        "store_lat": 12.914264,
        "store_lon": 77.678400,
        "drop_lat": 12.924264,
        "drop_lon": 77.688400,
        "weather": "Sunny",
        "traffic": "Low",
        "vehicle": "Motorcycle",
        "area": "Urban",
        "category": "Electronics",
    },
    "High-risk peak order": {
        "agent_age": 24,
        "agent_rating": 4.2,
        "prep_time": 18,
        "order_time": dt.time(hour=21, minute=15),
        "store_lat": 12.913041,
        "store_lon": 77.683237,
        "drop_lat": 13.043041,
        "drop_lon": 77.813237,
        "weather": "Stormy",
        "traffic": "Jam",
        "vehicle": "Scooter",
        "area": "Metropolitian",
        "category": "Grocery",
    },
}


@st.cache_resource
def load_artifacts():
    missing = [
        name
        for name in ("regressor.pkl", "classifier.pkl", "feature_info.json")
        if not (MODEL_DIR / name).exists()
    ]
    if missing:
        return None, None, None, missing

    regressor = joblib.load(MODEL_DIR / "regressor.pkl")
    classifier = joblib.load(MODEL_DIR / "classifier.pkl")
    feature_info = json.loads(FEATURE_INFO_PATH.read_text(encoding="utf-8"))
    return regressor, classifier, feature_info, []


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    return radius_km * 2 * math.asin(math.sqrt(a))


def build_feature_frame(
    feature_names,
    agent_age,
    agent_rating,
    distance_km,
    order_time,
    order_date,
    prep_time,
    weather,
    traffic,
    vehicle,
    area,
    category,
):
    raw = pd.DataFrame(
        [
            {
                "agent_age": int(agent_age),
                "agent_rating": float(agent_rating),
                "distance_km": float(distance_km),
                "order_hour": float(order_time.hour),
                "is_weekend": int(order_date.weekday() >= 5),
                "prep_time": float(prep_time),
                "weather": WEATHER_MAP[weather],
                "traffic": TRAFFIC_MAP[traffic],
                "vehicle": VEHICLE_MAP[vehicle],
                "area": AREA_MAP[area],
                "category": CATEGORY_MAP[category],
            }
        ]
    )

    encoded = pd.get_dummies(
        raw,
        columns=["weather", "traffic", "vehicle", "area", "category"],
        drop_first=True,
    )
    encoded = encoded.reindex(columns=feature_names, fill_value=0)

    for column in encoded.columns:
        if encoded[column].dtype == bool:
            encoded[column] = encoded[column].astype(int)

    return encoded


def risk_level(probability, predicted_minutes, threshold):
    if predicted_minutes > threshold or probability >= 0.65:
        return "High", "#b5472f"
    if probability >= 0.40:
        return "Watch", "#b7791f"
    return "Low", "#2f855a"


def render_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                linear-gradient(115deg, rgba(250, 247, 238, 0.82), rgba(232, 240, 246, 0.9)),
                radial-gradient(circle at 15% 10%, rgba(255, 206, 129, 0.35), transparent 28%),
                radial-gradient(circle at 85% 0%, rgba(64, 112, 156, 0.25), transparent 30%);
            color: #172033;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }

        .hero {
            background: linear-gradient(135deg, #12243a 0%, #21486b 62%, #c87535 140%);
            color: #fffaf2;
            border-radius: 28px;
            padding: 34px 36px;
            box-shadow: 0 24px 70px rgba(18, 36, 58, 0.20);
            margin-bottom: 1.25rem;
            position: relative;
            overflow: hidden;
        }

        .hero:after {
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            border-radius: 999px;
            right: -45px;
            top: -45px;
            background: rgba(255, 255, 255, 0.12);
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.74rem;
            opacity: 0.78;
            margin-bottom: 0.8rem;
        }

        .hero h1 {
            font-size: 2.65rem;
            line-height: 1;
            margin: 0 0 0.65rem 0;
            color: #ffffff;
        }

        .hero p {
            margin: 0;
            max-width: 760px;
            color: rgba(255, 250, 242, 0.86);
            font-size: 1.02rem;
        }

        .panel, div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(31, 54, 88, 0.09);
            border-radius: 24px;
            padding: 1.1rem;
            box-shadow: 0 16px 45px rgba(22, 35, 55, 0.09);
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 750;
            color: #15243b;
            margin-bottom: 0.5rem;
        }

        .helper {
            color: #647187;
            font-size: 0.93rem;
            margin-bottom: 1rem;
        }

        .metric-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(247,250,253,0.96));
            border: 1px solid rgba(32, 53, 84, 0.08);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            min-height: 120px;
        }

        .metric-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            color: #6a7486;
            margin-bottom: 0.45rem;
        }

        .metric-value {
            font-size: 2.05rem;
            font-weight: 760;
            color: #102038;
            margin: 0;
        }

        .metric-sub {
            color: #4e5d72;
            font-size: 0.9rem;
            margin: 0.25rem 0 0 0;
        }

        .result-card {
            border-radius: 24px;
            padding: 1.25rem;
            border: 1px solid rgba(31, 54, 88, 0.08);
            box-shadow: 0 16px 40px rgba(22, 35, 55, 0.08);
        }

        .result-title {
            font-size: 1.35rem;
            font-weight: 760;
            margin: 0;
        }

        .result-copy {
            color: #4c5a70;
            margin: 0.4rem 0 0 0;
        }

        .chip {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.72);
            font-weight: 700;
            font-size: 0.8rem;
            margin-bottom: 0.7rem;
        }

        .stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(135deg, #132a45, #2b5d85);
            color: white;
            border: 0;
            border-radius: 14px;
            font-weight: 700;
            padding: 0.78rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, subtitle):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <p class="metric-value">{value}</p>
            <p class="metric-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_context(feature_info):
    reg = feature_info["regression"]["metrics"]
    clf = feature_info["classification"]["metrics"]
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Regression R2", f"{reg['r2']:.2f}", f"MAE {reg['mae']:.2f} min")
    with col2:
        metric_card("Risk ROC-AUC", f"{clf['roc_auc']:.2f}", f"F1 {clf['f1_weighted']:.2f}")
    with col3:
        metric_card("Typical threshold", f"{feature_info['threshold']:.0f} min", "Median-based balanced target")


def render_inputs():
    st.markdown('<div class="section-title">Order Details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="helper">Choose a preset or enter custom values. Distance is calculated from the coordinates and checked against the training cleanup rule.</div>',
        unsafe_allow_html=True,
    )

    preset_name = st.selectbox("Quick scenario", list(SAMPLE_ORDERS.keys()))
    preset = SAMPLE_ORDERS[preset_name]

    col1, col2, col3 = st.columns(3)
    with col1:
        agent_age = st.number_input("Agent age", 18, 65, preset["agent_age"])
    with col2:
        agent_rating = st.slider("Agent rating", 1.0, 5.0, preset["agent_rating"], 0.1)
    with col3:
        prep_time = st.slider(
            "Order processing time (minutes)", 0, 60, preset["prep_time"], 1
        )

    col4, col5 = st.columns(2)
    with col4:
        order_date = st.date_input("Order date", value=dt.date.today())
    with col5:
        order_time = st.time_input("Order time", value=preset["order_time"])

    st.markdown('<div class="section-title">Route</div>', unsafe_allow_html=True)
    route_left, route_right = st.columns(2)
    with route_left:
        store_lat = st.number_input(
            "Store latitude", value=preset["store_lat"], format="%.6f", step=0.001
        )
        store_lon = st.number_input(
            "Store longitude", value=preset["store_lon"], format="%.6f", step=0.001
        )
    with route_right:
        drop_lat = st.number_input(
            "Drop latitude", value=preset["drop_lat"], format="%.6f", step=0.001
        )
        drop_lon = st.number_input(
            "Drop longitude", value=preset["drop_lon"], format="%.6f", step=0.001
        )

    distance_km = haversine_km(store_lat, store_lon, drop_lat, drop_lon)
    if distance_km > MAX_DISTANCE_KM:
        st.warning(
            f"Calculated distance is {distance_km:.2f} km. The model was trained after filtering routes above {MAX_DISTANCE_KM} km, so this prediction may be unreliable."
        )
    else:
        st.info(f"Calculated route distance: {distance_km:.2f} km")

    st.markdown('<div class="section-title">Conditions</div>', unsafe_allow_html=True)
    cond1, cond2, cond3 = st.columns(3)
    with cond1:
        weather = st.selectbox(
            "Weather", list(WEATHER_MAP.keys()), index=list(WEATHER_MAP).index(preset["weather"])
        )
        traffic = st.selectbox(
            "Traffic", list(TRAFFIC_MAP.keys()), index=list(TRAFFIC_MAP).index(preset["traffic"])
        )
    with cond2:
        vehicle = st.selectbox(
            "Vehicle", list(VEHICLE_MAP.keys()), index=list(VEHICLE_MAP).index(preset["vehicle"])
        )
        area = st.selectbox(
            "Area", list(AREA_MAP.keys()), index=list(AREA_MAP).index(preset["area"])
        )
    with cond3:
        category = st.selectbox(
            "Category", list(CATEGORY_MAP.keys()), index=list(CATEGORY_MAP).index(preset["category"])
        )

    return {
        "agent_age": agent_age,
        "agent_rating": agent_rating,
        "prep_time": prep_time,
        "order_date": order_date,
        "order_time": order_time,
        "distance_km": distance_km,
        "weather": weather,
        "traffic": traffic,
        "vehicle": vehicle,
        "area": area,
        "category": category,
    }


def render_result(pred_minutes, probability, predicted_class, threshold, order_datetime, inputs):
    level, color = risk_level(probability, pred_minutes, threshold)
    eta = order_datetime + dt.timedelta(minutes=float(pred_minutes))
    status = "Above typical" if predicted_class == 1 else "Typical range"
    card_bg = "#fff6ed" if level == "High" else "#fffaf0" if level == "Watch" else "#eef9f1"

    st.markdown(
        f"""
        <div class="result-card" style="background:{card_bg}; border-left: 7px solid {color};">
            <span class="chip" style="color:{color};">Risk level: {level}</span>
            <p class="result-title">{status} delivery duration</p>
            <p class="result-copy">
                The classifier estimates a {probability:.1%} chance that this order will exceed the
                model's typical-delivery threshold of {threshold:.0f} minutes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Estimated delivery", f"{pred_minutes:.0f} min", "Regression model output")
    with col2:
        metric_card("Above-typical probability", f"{probability:.1%}", "Classification risk signal")
    with col3:
        metric_card("Estimated ETA", eta.strftime("%H:%M"), "Order time plus predicted minutes")

    details = pd.DataFrame(
        [
            ("Predicted class", status),
            ("Order hour", int(inputs["order_time"].hour)),
            ("Weekend order", "Yes" if inputs["order_date"].weekday() >= 5 else "No"),
            ("Distance", f"{inputs['distance_km']:.2f} km"),
            ("Order processing time", f"{inputs['prep_time']:.0f} min"),
            ("Traffic / Weather", f"{inputs['traffic']} / {inputs['weather']}"),
        ],
        columns=["Field", "Value"],
    )
    st.dataframe(details, hide_index=True, use_container_width=True)


def main():
    render_css()
    regressor, classifier, feature_info, missing = load_artifacts()
    if missing:
        st.error(f"Missing required files: {', '.join(missing)}")
        st.stop()

    feature_names = feature_info["features"]["names"]
    threshold = feature_info["threshold"]

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">ML-powered ETA and risk scoring</div>
            <h1>Delivery Time Predictor</h1>
            <p>
                Estimate delivery time in minutes and flag orders likely to take longer than the
                dataset's typical delivery threshold. The app combines a regression ETA model with
                a separate above-typical risk classifier.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Model performance summary", expanded=False):
        render_model_context(feature_info)
        st.caption(
            "The risk label is statistical, not a promised-SLA delay label. Above typical means the predicted pattern is likely to exceed the median-based training threshold."
        )

    with st.form("prediction_form"):
        inputs = render_inputs()
        submitted = st.form_submit_button("Run prediction")

    if submitted:
        feature_frame = build_feature_frame(feature_names=feature_names, **inputs)
        pred_minutes = float(regressor.predict(feature_frame)[0])
        probabilities = classifier.predict_proba(feature_frame)[0]
        classes = list(classifier.classes_)
        above_typical_probability = float(probabilities[classes.index(1)])
        predicted_class = int(classifier.predict(feature_frame)[0])
        order_datetime = dt.datetime.combine(inputs["order_date"], inputs["order_time"])

        left, right = st.columns([1.25, 1], gap="large")
        with left:
            render_result(
                pred_minutes,
                above_typical_probability,
                predicted_class,
                threshold,
                order_datetime,
                inputs,
            )
        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Probability Breakdown</div>', unsafe_allow_html=True)
            probs_df = pd.DataFrame(
                {
                    "Outcome": ["Typical range", "Above typical"],
                    "Probability": [
                        float(probabilities[classes.index(0)]),
                        above_typical_probability,
                    ],
                }
            )
            st.bar_chart(probs_df.set_index("Outcome"), use_container_width=True)
            st.caption(
                "Use the regression output as the main ETA and this classifier as the supporting risk signal."
            )
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
