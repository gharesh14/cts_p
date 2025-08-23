### Sustainable Fuel Blend Predictor

#### Overview

This project provides a comprehensive, end-to-end solution for predicting the properties of blended fuels. The system consists of a machine learning model, a Flask-based backend API for model serving, and a user-friendly frontend for interacting with the application.

-----





### Installation & Setup

#### 1\. Prerequisites

Ensure you have Python 3.8+ installed.

#### 2\. Install Dependencies

All required Python libraries are listed in the `requirements.txt` file. You can install them by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

#### 3\. Model and Preprocessing Files

Before running the application, you must train your model and generate the necessary preprocessing files. Ensure that the following files are in your project directory:

  * `random_forest_multioutput.pkl`
  * `imputation_means.pkl`
  * `outlier_bounds.pkl`

-----

### How to Run the Application

#### 1\. Start the Backend Server

Open your terminal and run the Flask server:

```bash
python app.py
```

The server will start at `http://127.0.0.1:5000`.

#### 2\. Access the Frontend

With the backend running, simply open the `index.html` file in your web browser. The frontend will automatically connect to the backend, allowing you to upload a CSV file and download the predictions.
