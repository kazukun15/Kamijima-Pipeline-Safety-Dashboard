import streamlit as st
import geopandas as gpd
import folium
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import io
from reportlab.pdfgen import canvas
from shapely.geometry import Polygon

# ----- 1. GISデータの生成（疑似データ：愛媛県上島町付近） -----
def create_sample_gis_data():
    try:
        # 上島町付近の疑似ポリゴンを作成
        coords = [
            (132.495, 33.847),  # 左下
            (132.505, 33.847),  # 右下
            (132.505, 33.853),  # 右上
            (132.495, 33.853)   # 左上
        ]
        poly = Polygon(coords)
        # シンプルなGeoDataFrame作成。属性として risk (疑似値) を付与
        gdf = gpd.GeoDataFrame({'risk': [0.5]}, geometry=[poly], crs="EPSG:4326")
        return gdf
    except Exception as e:
        st.error(f"Error in creating sample GIS data: {e}")
        raise

def create_folium_map(gdf):
    try:
        if not gdf.empty:
            # GeoDataFrame の重心を計算してマップの中心に設定
            mean_lat = gdf.geometry.centroid.y.mean()
            mean_lon = gdf.geometry.centroid.x.mean()
        else:
            mean_lat, mean_lon = 33.85, 132.50  # デフォルト：上島町付近

        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=14)
        folium.GeoJson(gdf, name="Kamijima Area").add_to(m)
        folium.LayerControl().add_to(m)
        return m
    except Exception as e:
        st.error(f"Error in creating Folium map: {e}")
        raise

# ----- 2. 機械学習モデルによる予測（ダミーデータ） -----
def train_dummy_model():
    try:
        np.random.seed(42)
        # 仮の5個の特徴量を持つ学習データ
        X_train = np.random.rand(100, 5)
        # 仮のリスクスコア (0～1の範囲)
        y_train = np.random.rand(100)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Error in training dummy model: {e}")
        raise

def predict_risk(model, input_features):
    try:
        risk_score = model.predict(input_features)
        return risk_score[0]
    except Exception as e:
        st.error(f"Error in predicting risk: {e}")
        raise

# ----- 3. PDFレポート生成 -----
def generate_pdf_report(risk_score):
    try:
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer)
        # タイトル
        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, 800, "Kamijima Pipeline Safety Report")
        # 地域情報と予測結果の表示
        p.setFont("Helvetica", 12)
        p.drawString(100, 780, "Location: Kamijima, Ehime, Japan")
        p.drawString(100, 760, f"Predicted Risk Score: {risk_score:.2f}")
        p.drawString(100, 740, "Note: This is a test report using sample data.")
        p.showPage()
        p.save()
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error in generating PDF report: {e}")
        raise

# ----- 4. Streamlitアプリ本体 -----
def main():
    st.title("Kamijima Pipeline Safety Dashboard")
    
    # サイドバーでのパラメータ入力
    st.sidebar.header("Input Parameters (5 Features)")
    feature_values = []
    try:
        for i in range(5):
            val = st.sidebar.slider(f"Feature {i+1}", 0.0, 1.0, 0.5, 0.01)
            feature_values.append(val)
        # 1行×5列のNumPy配列に変換
        input_features = np.array(feature_values).reshape(1, -1)
    except Exception as e:
        st.error(f"Error in reading input parameters: {e}")
        return

    # GISデータの生成とFoliumマップ表示
    st.header("GIS Visualization (Kamijima Area)")
    try:
        gdf = create_sample_gis_data()
        folium_map = create_folium_map(gdf)
        # FoliumマップをStreamlitコンポーネントとして表示
        st.components.v1.html(folium_map._repr_html_(), height=500)
    except Exception as e:
        st.error("Failed to display GIS data.")
        return

    # 予測モデルの作成とリスク予測
    st.header("Risk Prediction")
    try:
        model = train_dummy_model()
        risk_score = predict_risk(model, input_features)
        st.write(f"Predicted Risk Score: **{risk_score:.2f}**")
    except Exception as e:
        st.error("Failed to predict risk.")
        return

    # PDFレポート生成の部分
    st.header("PDF Report Generation")
    try:
        if st.button("Generate PDF Report"):
            pdf_buffer = generate_pdf_report(risk_score)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="kamijima_pipeline_safety_report.pdf",
                mime="application/pdf"
            )
    except Exception as e:
        st.error("Failed to generate PDF report.")

if __name__ == "__main__":
    main()
