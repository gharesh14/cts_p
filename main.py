import joblib
import pandas as pd
from flask import Flask, request, send_file, jsonify, render_template
from werkzeug.utils import secure_filename
from io import BytesIO   # ✅ use BytesIO instead of StringIO
import os

app = Flask(__name__, template_folder="templates")

# Load all necessary files when the app starts
try:
    model = joblib.load('random_forest_multioutput.pkl')
    outlier_bounds = joblib.load('outlier_bounds.pkl')
    imputation_means = joblib.load('imputation_means.pkl')
    print("✅ All necessary files loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error: {e}. Please ensure you have generated these files and they are in the same directory.")
    model, outlier_bounds, imputation_means = None, None, None

# -------------------- Utility Functions -------------------- #

# Imputation function
def impute_data(df, means):
    return df.fillna(means)

# Outlier capping function
def preprocess_data(df, bounds):
    df_processed = df.copy()
    for col, bound_values in bounds.items():
        if col in df_processed.columns:
            lower_bound = bound_values['lower']
            upper_bound = bound_values['upper']
            df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
    return df_processed

# Feature engineering function
def engineer_features(df):
    fraction_cols = [col for col in df.columns if 'fraction' in col]
    engineered_features_df = pd.DataFrame(index=df.index)

    for i in range(1, 10 + 1):  # properties 1-10
        prop_number = str(i)
        weighted_avg = pd.Series([0.0] * len(df), index=df.index)

        for j in range(1, 5 + 1):  # components 1-5
            comp_number = str(j)
            fraction_col = f'Component{comp_number}_fraction'
            property_col = f'Component{comp_number}_Property{prop_number}'

            if fraction_col in df.columns and property_col in df.columns:
                weighted_avg += df[fraction_col] * df[property_col]

        new_feature_name = f'WeightedAvg_Property{prop_number}'
        engineered_features_df[new_feature_name] = weighted_avg

    return pd.concat([df[fraction_cols], engineered_features_df], axis=1)

# -------------------- Routes -------------------- #

# Home route → render index.html
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if not all([model, outlier_bounds, imputation_means]):
        return jsonify({'error': 'Server initialization failed. Files are missing.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save file
        filename = secure_filename(file.filename)
        upload_folder = os.path.join(os.getcwd(), "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Read CSV
        df_input = pd.read_csv(file_path)
        print("📂 Received and read input CSV file.")

        # Drop ID if exists
        if 'ID' in df_input.columns:
            test_ids = df_input['ID']
            df_features = df_input.drop('ID', axis=1)
        else:
            test_ids = None
            df_features = df_input

        # 1. Imputation
        df_imputed = impute_data(df_features, imputation_means)

        # 2. Preprocessing
        df_features_processed = preprocess_data(df_imputed, outlier_bounds)

        # 3. Feature Engineering
        X_processed = engineer_features(df_features_processed)

        # 4. Prediction
        predictions = model.predict(X_processed)

        target_columns = [f'BlendProperty{i}' for i in range(1, 10 + 1)]
        predictions_df = pd.DataFrame(predictions, columns=target_columns)

        if test_ids is not None:
            predictions_df.insert(0, 'ID', test_ids)

        # ✅ Save predictions to memory (binary mode for Flask)
        predictions_csv = BytesIO()
        predictions_df.to_csv(predictions_csv, index=False)
        predictions_csv.seek(0)

        return send_file(
            predictions_csv,
            mimetype="text/csv",
            as_attachment=True,
            download_name="predictions.csv"
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Run App -------------------- #

if __name__ == '__main__':
    app.run(debug=True, port=5000)
