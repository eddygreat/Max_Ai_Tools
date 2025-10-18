# MNIST Digit Classifier with Streamlit

This project implements a web-based MNIST Digit Classifier using a Convolutional Neural Network (CNN) with a Streamlit interface. Users can either draw a digit on a canvas or upload an image of a digit, and the application will predict the digit using a pre-trained TensorFlow model.

### Live Demo

You can access the live web application here: [MNIST Digit Classifier App](https://maxaitools.streamlit.app/)

## Features

*   **Interactive Drawing:** Draw digits directly in the web application using a drawable canvas.
*   **Image Upload:** Upload 28x28 grayscale images (PNG, JPG, JPEG) for classification.
*   **Real-time Predictions:** Get instant predictions from the trained CNN model.
*   **Confidence Score:** Displays the confidence level of the prediction.
*   **Model Information:** Sidebar provides basic information about the model used.

## Setup and Installation

Follow these steps to set up and run the project locally.

### Prerequisites

*   Python 3.8+
*   pip (Python package installer)

### 1. Clone the Repository (if applicable)

If your project is in a Git repository, clone it using:

```bash
git clone https://github.com/eddygreat/Max_Ai_Tools
cd <your-project-directory>
```

### 2. Create a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

*   **On Windows (Command Prompt):**
    ```bash
    .\venv\Scripts\activate
    ```
*   **On Windows (PowerShell):**
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
*   **On macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

### 4. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 5. Obtain the Trained Model

Ensure you have the `model.h5` file in the same directory as your `app.py`. This file contains the pre-trained TensorFlow model. If you don't have it, you'll need to train one or obtain a pre-trained MNIST model in H5 format.

## Running the Application

Once the setup is complete, you can run the Streamlit application:

```bash
streamlit run app.py
```

This command will open the application in your web browser, typically at `http://localhost:8501` or `http://localhost:8502`.

## Usage

1.  **Draw a Digit:** Navigate to the "✏️ Draw Digit" tab. Use your mouse to draw a digit (0-9) on the canvas. The prediction and confidence will update as you draw.
2.  **Upload an Image:** Switch to the "📁 Upload Image" tab. Click on "Upload a PNG/JPG image" and select a 28x28 grayscale image of a digit. The application will display the uploaded image and its predicted digit.

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── model.h5            # Pre-trained TensorFlow model
├── requirements.txt    # List of Python dependencies
└── README.md           # Project README file (this file)
```

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open-source and available under the [MIT License](LICENSE).

