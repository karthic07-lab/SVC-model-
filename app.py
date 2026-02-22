from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load ML artifacts
model = joblib.load('svc_model.joblib')
scaler = joblib.load('scaler.joblib')

FEATURE_COLUMNS = [
    'PackVoltage_V', 'CellVoltage_V', 'DemandVoltage_V',
    'ChargeCurrent_A', 'DemandCurrent_A',
    'SOC_%', 'MaxTemp_C', 'MinTemp_C', 'AvgTemp_C', 'AmbientTemp_C',
    'InternalResistance_mOhm', 'StateOfHealth_%',
    'VibrationLevel_mg', 'MoistureDetected',
    'ChargePower_kW', 'Pressure_kPa',
    'BMS_Status_OK', 'BMS_Status_Warning',
    'ChargingStage_Handshake',
    'ChargingStage_Parameter_Config',
    'ChargingStage_Recharge'
]

NUMERICAL_COLUMNS = [
    'PackVoltage_V', 'CellVoltage_V', 'DemandVoltage_V',
    'ChargeCurrent_A', 'DemandCurrent_A',
    'SOC_%', 'MaxTemp_C', 'MinTemp_C', 'AvgTemp_C',
    'AmbientTemp_C', 'InternalResistance_mOhm',
    'StateOfHealth_%', 'VibrationLevel_mg',
    'ChargePower_kW', 'Pressure_kPa'
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        for col in FEATURE_COLUMNS:
            if col not in df:
                df[col] = 0

        df = df[FEATURE_COLUMNS]
        df[NUMERICAL_COLUMNS] = scaler.transform(df[NUMERICAL_COLUMNS])

        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

        return jsonify({
            "prediction": prediction,
            "probability_of_TR_risk": round(probability, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
