# Natural-Lens #
A powerful deep learning-based software for identifying and classifying animals from images and video streams.

## Introduction ##
AnimalDetect AI is an open-source project designed to provide a fast and accurate solution for animal identification. The model is trained on a large and diverse dataset, enabling it to recognize hundreds of different species with high precision. Especially, typical spicies in Vietnam.
The goal of this project is to bring natural come closer to students, researchers, natural lover,v.v. Moreover, this project helps the researchers and conservationist to locate animal species (which are endanger) around Vietnam. It also help children get closer to animal and keep away phone.
## Table of Contents
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model and Dataset](#model-and-dataset)
- [Results and Evaluation](#results-and-evaluation)
- [Project Structure](#project-structure)
- [Future Development](#future-development)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

## Key Features
- **Multi-Class Detection:** Recognizes **100** different animal species (e.g., cats, dogs, bears, deer).
- **Image Detection:** Analyzes static images (`.jpg`, `.png`) to identify animals.
- **Video & Real-time Detection:** Processes video files (`.mp4`) and live webcam feeds.
- **Confidence Scores:** Displays the species name and the model's confidence level for each detection.

## Tech Stack
- **Language:** Python 3.13
- **Deep Learning Framework:** TensorFlow 2.x / PyTorch
- **Core Libraries:** OpenCV, NumPy, Matplotlib
- **UI (Optional):** Tkinter, PyQt, Streamlit

  ## Installation

Follow these steps to set up the project on your local machine.

1.  **Clone the repository:**
    ```bash git clone https://github.com/tuannguyen2209/Natural-Lens.git```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Create a `requirements.txt` file by running `pip freeze > requirements.txt` in your activated environment.)*
