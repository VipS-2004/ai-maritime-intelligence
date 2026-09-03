import streamlit as st

from models.detector import load_model, run_detection
from analysis.maritime_analysis import generate_analysis
from ai.agent import generate_intelligence_report
from visualization.visualizer import generate_heatmap, zone_based_analysis


st.set_page_config(
    page_title="Satellite Maritime Intelligence",
    page_icon="Satellite",
    layout="wide"
)

st.title("Satellite Maritime Intelligence")
st.write("AI-powered maritime vessel detection and intelligence analysis.")


uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Satellite Image",
        width=650
    )

    if st.button("Analyze Image"):

        with st.spinner("Loading YOLO model..."):
            model = load_model("best.pt")

        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Detecting vessels..."):
            results = run_detection(model, "temp_image.jpg")

        orig_img = results[0].orig_img.copy()

        st.success("Detection completed.")

        st.subheader("Detected Vessels")

        detection_image = results[0].plot()

        st.image(
            detection_image,
            caption="YOLOv8 Vessel Detection",
            width=650
        )

        with st.spinner("Analyzing maritime activity..."):
            analysis = generate_analysis(results, model)

        st.subheader("Maritime Intelligence")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Ships",
                analysis["total_ships"]
            )

        with col2:
            st.metric(
                "Military Ships",
                analysis["military_ships"]
            )

        with col3:
            st.metric(
                "Risk Level",
                analysis["risk_level"]
            )

        st.subheader("Traffic Intelligence")

        col1, col2 = st.columns(2)

        with col1:
            st.write(
                "**Congestion:**",
                analysis["congestion"]["level"]
            )

        with col2:
            st.write(
                "**Density:**",
                analysis["congestion"]["density"]
            )

        st.subheader("Density Heatmap")

        heatmap = generate_heatmap(
            results,
            orig_img
        )

        st.image(
            heatmap,
            caption="Ship Density Heatmap",
            width=650
        )

        st.subheader("Zone-Based Traffic Analysis")

        zone_counts, hotspots, zone_overlay = zone_based_analysis(
            results,
            orig_img,
            grid_size=4
        )

        st.image(
            zone_overlay,
            caption="4x4 Zone Traffic Map",
            width=650
        )

        st.write("Ship count by zone:")

        st.dataframe(
            zone_counts,
            width="stretch"
        )

        if len(hotspots) > 0:
            st.write(
                "Hotspot zones:",
                hotspots.tolist()
            )
        else:
            st.write("No hotspots detected.")

        st.subheader("Vessel Classification")

        st.json(
            analysis["class_breakdown"]
        )

        st.subheader("Vessel Composition")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Military",
                analysis["military_ships"]
            )

        with col2:
            st.metric(
                "Civilian",
                analysis["civilian_ships"]
            )

        with col3:
            st.metric(
                "Unknown",
                analysis["unknown_ships"]
            )

        st.subheader("Clustering Analysis")

        if analysis["clustering"]["detected"]:
            st.warning(
                analysis["clustering"]["message"]
            )
        else:
            st.success(
                analysis["clustering"]["message"]
            )

        st.subheader("System Alert")

        if "WARNING" in analysis["alert"]:
            st.error(
                analysis["alert"]
            )
        else:
            st.info(
                analysis["alert"]
            )

        st.subheader("AI Intelligence Assessment")

        try:
            with st.spinner("Generating intelligence assessment..."):
                intelligence_report = generate_intelligence_report(
                    analysis
                )

            st.write(
                intelligence_report
            )

        except Exception:
            st.warning(
                "AI intelligence assessment is temporarily unavailable. "
                "The computer vision analysis is still available."
            )