import streamlit as st
from joblib import load
import requests
import pandas as pd

loaded_model = load('Random_Forest.joblib')

Classes = ['Setosa', 'Versicolor', 'Virginica']


def index_maxValue(arr):
    maxValue = -1
    maxIdx = -1
    for i in range(0, len(arr)):
        if arr[i] > maxValue:
            maxValue = arr[i]
            maxIdx = i
    return maxIdx


def is_number(value):
    try:
        float(value)  # Chuyển đổi thành float, nếu thành công thì là số
        return True
    except ValueError:
        return False


def predict(data):
    result = loaded_model.predict_proba([data])
    return result[0]


def main():
    st.markdown("<h1 style='text-align: center;'>NHẬN DẠNG LOÀI HOA IRIS</h1>",
                unsafe_allow_html=True)
    st.write(
        "**Mô tả**: Hệ thống sử dụng giải thuật **Random Forest**, huấn luyện với tập dữ liệu Iris ([Xem chi tiết](https://archive.ics.uci.edu/dataset/53/iris)), dữ liệu được đánh giá với nghi thức **Hold-out** và được huấn luyện 1o lần được trung bình **Accuracy = 1.**")
    st.write("**Bộ tham số Training**: 'n_estimators=100, random_state=42'.")
    st.write("**Sinh viên**: Phan Minh Tài - B2113341.")
    st.subheader("Nhập 4 đặc trưng của hoa vào bên dưới (đo bằng cm):")
    feature1 = st.text_input(
        "Chiều dài đài hoa (Sepal Length):", placeholder="ví dụ: 5.1")
    feature2 = st.text_input(
        "Chiều rộng đài hoa (Sepal Width):", placeholder="ví dụ: 3.5")
    feature3 = st.text_input(
        "Chiều dài cánh hoa (Petal Length):", placeholder="ví dụ: 1.4")
    feature4 = st.text_input(
        "Chiều rộng cánh hoa (Petal Width):", placeholder="ví dụ: 0.2")

    predict_button = st.button("Nhận dạng")

    if predict_button:
        if not is_number(feature1) or feature1 == "":
            st.warning(
                "Vui lòng nhập chiều dài đài hoa (Sepal Length) là số (cm)"
            )
        elif not is_number(feature2) or feature2 == "":
            st.warning(
                "Vui lòng nhập Chiều rộng đài hoa (Sepal Width) là số (cm)"
            )
        elif not is_number(feature3) or feature3 == "":
            st.warning(
                "Vui lòng nhập Chiều dài cánh hoa (Petal Length) là số (cm)"
            )
        elif not is_number(feature4) or feature4 == "":
            st.warning(
                "Vui lòng nhập Chiều rộng cánh hoa (Petal Width) là số (cm)"
            )
        else:
            data = [float(feature1), float(feature2),
                    float(feature3), float(feature4)]
            result = predict(data)

            table = {
                "Loài": Classes,
                "Xác suất": result
            }
            df = pd.DataFrame(table)
            df = df.reset_index(drop=True)
            # Hiển thị DataFrame dưới dạng bảng
            st.table(df)
            st.header(f"Kết quả cuối cùng: {Classes[index_maxValue(result)]}")


if __name__ == "__main__":
    main()
