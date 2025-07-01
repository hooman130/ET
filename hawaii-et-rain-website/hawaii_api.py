from flask import Flask, request, jsonify, Response
import pandas as pd
from dotenv import load_dotenv
from hawaii_web import fetch_and_predict_hawaii, farm_coords, HORIZON

load_dotenv()

app = Flask(__name__)

@app.route('/api/predict')
def api_predict():
    """Return forecast data for a farm as JSON or CSV."""
    farm = request.args.get('farm')
    out_format = request.args.get('format', 'json').lower()

    if not farm or farm not in farm_coords:
        return jsonify({'error': 'Invalid or missing farm parameter'}), 400

    results = fetch_and_predict_hawaii(farm)
    if not results:
        return jsonify({'error': 'No prediction available'}), 500

    output = {
        'query_farm': farm,
        'prediction_farm': results['farm'],
        'latitude': results['latitude'],
        'longitude': results['longitude'],
        'last_data_date': results['last_data_date'],
        'forecast': [
            {
                'date': results['prediction_dates'][i],
                'et_mm_day': results['et_mm_day'][i],
                'rain_mm': results['rain_mm'][i],
            } for i in range(HORIZON)
        ]
    }

    if out_format == 'csv':
        rows = []
        for item in output['forecast']:
            rows.append({
                'farm': output['prediction_farm'],
                'latitude': output['latitude'],
                'longitude': output['longitude'],
                'last_data_date': output['last_data_date'],
                'forecast_date': item['date'],
                'et_mm_day': item['et_mm_day'],
                'rain_mm': item['rain_mm'],
            })
        df = pd.DataFrame(rows)
        return Response(df.to_csv(index=False), mimetype='text/csv')

    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
