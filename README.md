# Video Frame Interpolation and Super Resolution

This repository provides implementations for **Video Frame Interpolation** using **RIFE** and **Super Resolution** using **ESRGAN**. It enhances video quality by generating smooth interpolated frames and upscaling low-resolution videos using deep learning models. Pre-trained models and custom training options are included for flexibility.

## Features
- **Frame Interpolation (RIFE)**: Generate high-quality intermediate frames to smooth motion or change video frame rates.
- **Super Resolution (ESRGAN)**: Enhance low-resolution videos or frames using state-of-the-art deep learning models to predict high-resolution outputs.
- **Pre-trained Models**: Ready-to-use pre-trained models for both interpolation (RIFE) and super resolution (ESRGAN).
- **Customizable**: Easily train with your own dataset for custom video enhancement needs.

## Requirements
The following dependencies are required to run the project:
- Python 3.6+
- PyTorch (>=1.7.0)
- OpenCV (>=4.5.0)
- NumPy (>=1.19.0)
- tqdm (>=4.50.0)
- scikit-image (>=0.18.3)
- FFmpeg (for video processing)
- CUDA-enabled GPU (Recommended for better performance)

You can install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/video-frame-interpolation-super-resolution.git
cd video-frame-interpolation-super-resolution
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download pre-trained models (optional but recommended):
- **ESRGAN**: Pre-trained models can be downloaded from [here](link-to-esrgan-model).
- **RIFE**: Pre-trained models can be downloaded from [here](link-to-rife-model).

4. Set up FFmpeg to handle video file conversion, or use the default methods provided in the code.

## Usage

### Frame Interpolation (RIFE)
To perform frame interpolation on a video, run the following command:

```bash
python frame_interpolation.py --input_video path_to_input_video.mp4 --output_video path_to_output_video.mp4
```

### Super Resolution (ESRGAN)
To perform super resolution on a frame or image, run:

```bash
python super_resolution.py --input_image path_to_input_image.jpg --output_image path_to_output_image.jpg
```

### Training Your Own Model
To train a model from scratch using your own dataset, use the following:

```bash
python train.py --dataset_path path_to_your_dataset --output_dir path_to_save_model
```

Make sure to preprocess your data and split it into training and testing sets.

## Example
Here is an example of using the frame interpolation and super resolution pipeline:

1. **Step 1**: First, enhance the resolution of the low-resolution video using ESRGAN.

```bash
python super_resolution.py --input_video low_res_video.mp4 --output_video high_res_video.mp4
```

2. **Step 2**: Then, apply frame interpolation using RIFE to generate additional frames.

```bash
python frame_interpolation.py --input_video high_res_video.mp4 --output_video smooth_video.mp4
```

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit pull requests. Please ensure that you follow the coding style and write appropriate tests for new features.

### Steps to contribute:
1. Fork this repository
2. Create a feature branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [ESRGAN](link-to-paper-or-repository) - Enhanced Super-Resolution Generative Adversarial Networks for Image Super-Resolution.
- [RIFE](link-to-paper-or-repository) - Real-Time Intermediate Flow Estimation for Frame Interpolation.

## Contact
If you have any questions or suggestions, feel free to reach out via Issues or contact [your-email@example.com](mailto:your-email@example.com).
