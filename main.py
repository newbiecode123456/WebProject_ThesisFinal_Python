import streamlit as st 
import pandas as pd 
import numpy as np 
import cv2 
from keras.models import load_model
import io
from PIL import Image, ImageOps
import time 
def main():
    # title block
    st.markdown(
        """
        <div style='text-align: center; background-color: #005f69;'>
            <img src='https://www.ueh.edu.vn/images/logo-header.png' alt='Logo UEH'/>
            <h2>Đại học UEH</h2>
            <h3><img src='https://ctd.ueh.edu.vn/wp-content/uploads/2023/07/cropped-TV_trang_CTD.png' alt='Logo CTD UEH' width='100'/> Trường Công nghệ và Thiết kế UEH</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    # end title

    st.header("Dự đoán Viêm phổi (Pneumonia) sử dụng CNN predict + Streamlit cho frontend")
    
    # đưa về dạng thập phân
    np.set_printoptions(suppress=True)
    
    # xử lý chọn model
    menu = ["Home", "Model 1", "Model 2", "Model 3", "Model 4", "Model CNN", "Model TM" ]
    st.sidebar.title('Navigation')
    choice = st.sidebar.selectbox("Chọn một Model", menu)
    isLoaded = False
    pixels = 0
    
    if choice == "Home":
        st.write("Nhấn vào Navigator để lựa chọn model!!!!")
    elif choice == "Model 1":
        st.subheader("Model 1")
        st.write("Model 1: Sử dụng kiến trúc VGG16 mô hình CNN để chuẩn đoán (được train cơ bản trên 10 epochs)")
        st.write("Colab train mô hình VGG16 [tại đây](https://colab.research.google.com/drive/1_HBOtbnRYRp1hp76UGxloqSFiuqwdcgK?ouid=111422511877040041683&usp=drive_link)")
        # load model
        with st.spinner("Đang thực hiện load Model..."):
            model = load_model("models_comparision/cnn_vgg16_10epochs_model.h5", compile=False)
            isLoaded = True
            pixels = 224
    elif choice == "Model 2":
        st.subheader("Model 2")
        st.write("Model 2: Sử dụng kiến trúc ResNet50 với mô hình CNN để chuẩn đoán (được train cơ bản trên 10 epochs)")
        st.write("Colab train mô hình ResNet50 [tại đây](https://colab.research.google.com/drive/137Bbwhi7SgU8hRMV1tAzYsE6yKhD7qTG?ouid=111422511877040041683&usp=drive_link)")
        # load model
        with st.spinner("Đang thực hiện load Model..."):
            model = load_model("models_comparision/cnn_resnet50_10epochs_model.h5", compile=False)
            isLoaded = True
            pixels = 224
    elif choice == "Model 3":
        st.subheader("Model 3")
        st.write("Model 3: Sử dụng kiến trúc DenseNet121 với mô hình CNN để chuẩn đoán (được train cơ bản trên 10 epochs)")
        st.write("Colab train mô hình DenseNet121 [tại đây](https://colab.research.google.com/drive/1Zsve44wV0XKz-K9Xlv-_piS9tIb9G1ad?ouid=111422511877040041683&usp=drive_link)")
        # load model
        with st.spinner("Đang thực hiện load Model..."):
            model = load_model("models_comparision/cnn_densenet_10epoch_model.h5", compile=False)
            isLoaded = True
            pixels = 224
    elif choice == "Model 4":
        st.subheader("Model 4")
        st.write("Model 4: Sử dụng kiến trúc Xception với mô hình CNN để chuẩn đoán (được train cơ bản trên 10 epochs)")
        st.write("Colab train mô hình Xception [tại đây](https://colab.research.google.com/drive/1lKN-nt9sQgNsMUOzKKkREI-UhrvoTb_Y?ouid=111422511877040041683&usp=drive_link)")
        with st.spinner("Đang thực hiện load Model..."):
            model = load_model("models_comparision//cnn_xception_10epochs_final_model.h5", compile=False)
            isLoaded = True
            pixels = 299
    elif choice == "Model CNN":
        st.subheader("Model CNN kiến trúc Xception cuối có điều chỉnh các tham số")
        st.write("Colab train mô hình Xception cuối [tại đây](https://colab.research.google.com/drive/12magZlRm7Nu0G2RQjkW5ICj6G5C3BoHk?ouid=111422511877040041683&usp=drive_link)")
        st.write("Model CNN")
        with st.spinner("Đang thực hiện load Model..."):
            model = load_model("final_xception_model/cnn_xception_50epochs_addbatchnormolization+dropout_final_model.h5", compile=False)
            isLoaded = True
            pixels = 299
    elif choice == "Model TM":
        st.subheader("Model sử dụng Teachable Machine")
        st.write("Model Teachable Machine: Sử dụng dụng mô hình của teachable machine để dự đoán")
        st.write("Trang chủ  [Teachable Machine](https://teachablemachine.withgoogle.com/)")
        with st.spinner("Đang thực hiện load Model..."):
            model = load_model("teachable_model/keras_model.h5", compile=False)
            isLoaded = True
            pixels = 224
    else: st.write("Chưa chọn model!!!")

    # upload file
    if isLoaded:
        uploaded_image = st.file_uploader("Chọn file ảnh")
        data = np.ndarray(shape=(1, pixels, pixels, 3), dtype=np.float32)
        class_names = open("labels.txt", "r").readlines()
        if not (uploaded_image is None):
            img_cap = "Dung lượng file là: " + str(uploaded_image.size) + " kb"
            st.image(uploaded_image, caption=img_cap)
            image_data = uploaded_image.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            # resize image 
            size = (pixels, pixels)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            # turn the image into a numpy array
            image_array = np.asarray(image)
            # normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            # image into the array
            data[0] = normalized_image_array
            # predict image
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            st.info("Các tham số dự đoán: ")
            st.write("Class:", class_name[2:])
            st.write("Confidence Score: ", confidence_score)
            st.info("Kết quả dự đoán: ")
            if int(class_name[0]) == 1 or (class_name[0] == 0 and confidence_score < 0.8):
                percentx = round(confidence_score,2) * 100
                st.error("Mô hình dự đoán Phim X quang được tải lên có nguy cơ viêm phổi với độ chính xác " + str(percentx) + "%. Bạn cần thực hiện các phát đồ chuyên môn tại cơ sở y tế gần nhất sớm nhất có thể.")
            else:
                st.success("Mô hình không dự đoán được bất thường nhưng nếu bạn có BẤT CỨ TRIỆU CHỨNG nào của bệnh Viêm phổi hãy đến ngay cơ sở Y tế gần nhất")
    
def footer_h():
    st.warning("Các mô hình được huấn luyện nhằm mục đích Giáo dục với dữ liệu hạn chế (số lượng ít) và chưa được kiểm định bởi chuyên gia về tính chính xác!")
    st.write("Bộ dữ liệu train các mô hình [tại đây](https://drive.google.com/drive/folders/1CdefiFd3O20duRD3koTLVXWmKztdWbx_?usp=sharing)")
    st.write("Bộ dữ liệu gốc trên Kaggle [tại đây](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)")
    st.write("Giấy phép và tác giả Paul Mooney [tại đây](https://creativecommons.org/licenses/by/4.0/)")
    
    st.write("Một số hình ảnh sample: ")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image_sample_xray/normal_1.jpeg", caption="ảnh Normal Xray Image 1")
        with open("image_sample_xray/normal_1.jpeg", "rb") as file:
            st.download_button(
                label = "Tải ảnh Normal Xray Image 1",
                data = file,
                file_name = "normal_1.jpeg",
                mime = "image/jpeg"
            )
    with col2:
        st.image("image_sample_xray/pneumonia_1.jpeg", caption="ảnh Pneumonia Xray Image 1")
        with open("image_sample_xray/pneumonia_1.jpeg", "rb") as file:
            st.download_button(
                label = "Tải ảnh Pneumonia Xray Image 1",
                data = file,
                file_name = "pneumonia_1.jpeg",
                mime = "image/jpeg"
            )
    with col3:
        st.image("image_sample_xray/pneumonia_5.jpeg", caption="ảnh Pneumonia Xray Image 2")
        with open("image_sample_xray/pneumonia_5.jpeg", "rb") as file:
            st.download_button(
                label = "Tải ảnh Pneumonia Xray Image 2",
                data = file,
                file_name = "pneumonia_5.jpeg",
                mime = "image/jpeg"
            )
        
if __name__ == "__main__":
    main()
    
    footer_h()