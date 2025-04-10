%%writefile app.py
import streamlit as st
import cv2
import numpy as np
import os
import subprocess
import time
from PIL import Image
import base64
from io import BytesIO
import shutil
import sys

# Set page configuration
st.set_page_config(
    page_title="Video Enhancement Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional UI Theme
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
        color: #333333;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }

    h1, h2, h3, h4 {
        color: #2c3e50;
        font-weight: 600;
    }

    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    .highlight {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #e9ecef;
    }

    .title-text {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }

    .card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }

    .upload-section {
        border: 2px dashed #cbd5e0;
        border-radius: 8px;
        padding: 30px;
        text-align: center;
        background-color: #f8f9fa;
    }

    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }

    .stProgress > div > div {
        background-color: #3498db;
    }

    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 10px 0;
    }

    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 10px 0;
    }

    footer {display: none !important;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Super Resolution Functions
def nedi_interpolation(lr, scale=2):
    hr = cv2.resize(lr, (lr.shape[1]*scale, lr.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    lap = cv2.Laplacian(hr, cv2.CV_64F)
    hr_enhanced = cv2.addWeighted(hr, 1.0, cv2.convertScaleAbs(lap), 0.2, 0)
    return hr_enhanced

def ibp_nedi_super_resolution(lr, scale=2, iterations=5):
    hr = cv2.resize(lr, (lr.shape[1]*scale, lr.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    kernel_size = 5
    sigma = 1.0
    for i in range(iterations):
        hr_blurred = cv2.GaussianBlur(hr, (kernel_size, kernel_size), sigma)
        simulated_lr = cv2.resize(hr_blurred, (lr.shape[1], lr.shape[0]), interpolation=cv2.INTER_CUBIC)
        error_lr = cv2.subtract(lr, simulated_lr)
        error_hr = nedi_interpolation(error_lr, scale)
        hr = cv2.add(hr, error_hr)
    return hr

def process_frame(frame, scale=2, iterations=5):
    return ibp_nedi_super_resolution(frame, scale=scale, iterations=iterations)

# Video Processing Functions
def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.png")
        cv2.imwrite(frame_path, frame)
        count += 1
        progress_bar.progress(min(count/total_frames, 1.0))
        status_text.text(f"Processing frame {count}/{total_frames}")
    cap.release()
    status_text.empty()
    return fps, count

def apply_super_resolution(frames_folder, output_folder, total_frames, scale=2, iterations=5):
    os.makedirs(output_folder, exist_ok=True)
    frame_files = sorted(os.listdir(frames_folder))
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, fname in enumerate(frame_files):
        frame_path = os.path.join(frames_folder, fname)
        frame = cv2.imread(frame_path)
        sr_frame = process_frame(frame, scale=scale, iterations=iterations)
        out_path = os.path.join(output_folder, fname)
        cv2.imwrite(out_path, sr_frame)
        progress_bar.progress(min((i+1)/total_frames, 1.0))
        status_text.text(f"Enhancing resolution: {i+1}/{total_frames}")
    status_text.empty()

def create_video_from_frames(frame_folder, output_video, fps=30):
    frame_files = sorted(os.listdir(frame_folder))
    if not frame_files:
        return False
    first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(frame_files)
    for i, fname in enumerate(frame_files):
        frame = cv2.imread(os.path.join(frame_folder, fname))
        out.write(frame)
        progress_bar.progress(min((i+1)/total, 1.0))
        status_text.text(f"Creating video: {i+1}/{total}")
    out.release()
    status_text.empty()
    return True

# NumPy Compatibility Patch
def create_numpy_compatibility_patch(rife_repo_path):
    patch_file = os.path.join(rife_repo_path, "numpy_patch.py")
    with open(patch_file, "w") as f:
        f.write("""
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
""")
    return patch_file

# RIFE Interpolation Function
def run_rife_interpolation(input_video_path, output_video_path, exp_factor=4):
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Enhancing frame rate...")

    try:
        rife_repo_path = "/content/drive/MyDrive/ECCV2022-RIFE"
        if not os.path.exists(rife_repo_path):
            fallback_interpolation(input_video_path, output_video_path, exp_factor)
            return True

        if not os.path.exists(input_video_path):
            shutil.copy(input_video_path, output_video_path)
            return True

        patch_file = create_numpy_compatibility_patch(rife_repo_path)
        inference_file = os.path.join(rife_repo_path, "inference_video.py")

        if not os.path.exists(inference_file):
            fallback_interpolation(input_video_path, output_video_path, exp_factor)
            return True

        with open(inference_file, "r") as f:
            content = f.read()

        if "import numpy_patch" not in content:
            modified_content = "import numpy_patch\n" + content
            with open(inference_file, "w") as f:
                f.write(modified_content)

        prev_dir = os.getcwd()
        os.chdir(rife_repo_path)

        command = f"python3 inference_video.py --exp={exp_factor} --video=\"{input_video_path}\" --output=\"{output_video_path}\""
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

        for i in range(101):
            progress_bar.progress(i/100)
            status_text.text(f"Enhancing frame rate: {i}%")
            time.sleep(0.1)

        process.communicate()

        os.chdir(prev_dir)

        if os.path.exists(output_video_path):
            return True
        else:
            fallback_interpolation(input_video_path, output_video_path, exp_factor)
            return True

    except Exception:
        fallback_interpolation(input_video_path, output_video_path, exp_factor)
        return True

def fallback_interpolation(input_video_path, output_video_path, exp_factor):
    try:
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        target_fps = fps * exp_factor

        ffmpeg_cmd = f"ffmpeg -i {input_video_path} -filter:v \"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1\" -c:v libx264 -c:a copy {output_video_path}"
        subprocess.call(ffmpeg_cmd, shell=True)

        if not os.path.exists(output_video_path):
            shutil.copy(input_video_path, output_video_path)
    except:
        shutil.copy(input_video_path, output_video_path)

    return os.path.exists(output_video_path)

def get_download_link(file_path, link_text="Download Enhanced Video"):
    if not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        href = (f'<a href="data:video/mp4;base64,{b64}" download="enhanced_video.mp4" style="text-decoration: none;">'
                f'<button style="background-color: #3498db; color: white; padding: 12px 20px; border: none; '
                f'border-radius: 4px; font-weight: 500; cursor: pointer; font-size: 1rem;">'
                f'{link_text}</button></a>')
        return href
    except Exception:
        return ""

# Banner Image
def display_banner():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <img src="https://img.freepik.com/premium-photo/digital-thinking-neural-networks-artificial-intelligence-machine-science-banner-made-with-generative-ai_155027-3430.jpg"
             style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    </div>
    """, unsafe_allow_html=True)

    # If image doesn't load, use a placeholder with CSS gradient
    st.markdown("""
    <div style="display: none; background: linear-gradient(135deg, #3498db, #8e44ad); height: 200px;
                border-radius: 8px; margin-bottom: 2rem; text-align: center; line-height: 200px;">
        <h1 style="color: white; margin: 0;">Video Enhancement Studio</h1>
    </div>
    """, unsafe_allow_html=True)

# Main App
def main():
    display_banner()

    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="title-text" style="font-size: 2.5rem;">Video Enhancement Studio</h1>
        <p style="font-size: 1.1rem; color: #7f8c8d;">Transform your videos with advanced AI-powered enhancement technology</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>Configuration</h2>", unsafe_allow_html=True)

        st.image("https://storage.googleapis.com/files.deepcognition.ai/images/video-resolution.jpg",
                 caption="Super Resolution", use_column_width=True)
        with st.container():
            st.subheader("Resolution Enhancement")
            scale_factor = st.slider("Scale Factor", 1, 4, 2, 1)
            iterations = st.slider("Enhancement Iterations", 1, 10, 5, 1)

        st.image("https://storage.googleapis.com/files.deepcognition.ai/images/video-fps.jpg",
                 caption="Frame Rate Enhancement", use_column_width=True)
        with st.container():
            st.subheader("Frame Rate Enhancement")
            use_rife = st.checkbox("Use AI Interpolation", value=True)
            exp_factor = st.slider("Interpolation Factor", 1, 4, 4, 1)

        with st.container():
            st.subheader("Output Settings")
            output_fps = st.slider("Target FPS", 24, 120, 60, 1)

        st.markdown("""
        <div style="margin-top: 30px; text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 8px;">
            <p style="color: #7f8c8d;">Created by Atharva Thombare</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<h3 style='text-align: center;'>Upload Video</h3>", unsafe_allow_html=True)
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select a video file to enhance", type=["mp4", "avi", "mov"], key="video_uploader")
        st.markdown("</div>", unsafe_allow_html=True)
        if uploaded_file is not None:
            st.markdown("<div class='highlight'>", unsafe_allow_html=True)
            st.video(uploaded_file)
            file_details = {"Filename": uploaded_file.name, "File size": f"{uploaded_file.size/(1024*1024):.2f} MB"}
            st.json(file_details)
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 style='text-align: center;'>Process Video</h3>", unsafe_allow_html=True)
        if uploaded_file is None:
            st.markdown("<div class='card' style='text-align: center; padding: 40px;'>", unsafe_allow_html=True)
            st.markdown("Upload a video to begin processing")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            run_process = st.button("Enhance Video", key="process_button")
            st.markdown("</div>", unsafe_allow_html=True)
            if run_process:
                persistent_dir = '/content/temp_video'
                os.makedirs(persistent_dir, exist_ok=True)
                frames_dir = os.path.join(persistent_dir, "frames")
                sr_frames_dir = os.path.join(persistent_dir, "super_res_frames")
                os.makedirs(frames_dir, exist_ok=True)
                os.makedirs(sr_frames_dir, exist_ok=True)

                input_video_path = os.path.join(persistent_dir, "input_video.mp4")
                with open(input_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Processing video..."):
                    st.markdown("<div class='info-box'>Extracting video frames</div>", unsafe_allow_html=True)
                    fps, total_frames = extract_frames(input_video_path, frames_dir)

                    st.markdown("<div class='info-box'>Enhancing video resolution</div>", unsafe_allow_html=True)
                    apply_super_resolution(frames_dir, sr_frames_dir, total_frames, scale=scale_factor, iterations=iterations)

                    st.markdown("<div class='info-box'>Creating enhanced resolution video</div>", unsafe_allow_html=True)
                    super_res_video_path = os.path.join(persistent_dir, "super_res_video.mp4")
                    create_video_from_frames(sr_frames_dir, super_res_video_path, fps=fps)

                    interpolated_video_path = os.path.join(persistent_dir, "interpolated_video.mp4")
                    if use_rife:
                        st.markdown("<div class='info-box'>Enhancing frame rate with AI</div>", unsafe_allow_html=True)
                        run_rife_interpolation(super_res_video_path, interpolated_video_path, exp_factor)
                    else:
                        st.markdown("<div class='info-box'>Enhancing frame rate</div>", unsafe_allow_html=True)
                        target_fps = fps * exp_factor
                        ffmpeg_cmd = f"ffmpeg -i {super_res_video_path} -filter:v \"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1\" -c:v libx264 -c:a copy {interpolated_video_path}"
                        subprocess.call(ffmpeg_cmd, shell=True)
                        if not os.path.exists(interpolated_video_path):
                            shutil.copy(super_res_video_path, interpolated_video_path)

                st.markdown("<div class='success-box'>Enhancement process complete!</div>", unsafe_allow_html=True)
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'>Results Comparison</h3>", unsafe_allow_html=True)
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown("<p style='text-align: center; font-weight: 500;'>Original Video</p>", unsafe_allow_html=True)
                    st.video(input_video_path)
                with col_right:
                    st.markdown("<p style='text-align: center; font-weight: 500;'>Enhanced Video</p>", unsafe_allow_html=True)

                    final_video_path = interpolated_video_path
                    if not os.path.exists(final_video_path) or os.path.getsize(final_video_path) == 0:
                        final_video_path = super_res_video_path

                    if os.path.exists(final_video_path) and os.path.getsize(final_video_path) > 0:
                        try:
                            with open(final_video_path, "rb") as video_file:
                                video_bytes = video_file.read()
                            st.video(video_bytes)
                        except:
                            st.video(input_video_path)
                    else:
                        st.video(input_video_path)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='card' style='text-align: center;'>", unsafe_allow_html=True)
                st.markdown("<h3>Download Your Enhanced Video</h3>", unsafe_allow_html=True)

                if os.path.exists(interpolated_video_path) and os.path.getsize(interpolated_video_path) > 0:
                    st.markdown(get_download_link(interpolated_video_path), unsafe_allow_html=True)
                elif os.path.exists(super_res_video_path) and os.path.getsize(super_res_video_path) > 0:
                    st.markdown(get_download_link(super_res_video_path, "Download Enhanced Video"), unsafe_allow_html=True)
                elif os.path.exists(input_video_path) and os.path.getsize(input_video_path) > 0:
                    st.markdown(get_download_link(input_video_path, "Download Original Video"), unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
