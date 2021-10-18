#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/10/18 16:37:33
@Author  :   zhangxl 
@Version :   1.0
@Desc    :   None
'''

# here put the import lib

import streamlit as st
import os
from PIL import Image
import cv2



if __name__ == '__main__':

    st.title('YOLOv5 Streamlit App')

    

    source = ("图片检测", "视频检测")
    source_name = ""
    source_index = st.sidebar.selectbox("选择输入", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "上传图片", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                source_name = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False
    else:
        uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                source_name = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False

    if is_valid:
        print('valid')
        if st.button('开始检测'):

            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    img = cv2.imread(source_name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Detect faces
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    # Draw rectangle around the faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        st.image(img)

                    st.balloons()
            else:
                
                    st.balloons()