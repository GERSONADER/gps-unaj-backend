from flask import Flask, request, jsonify
from flask_cors import CORS
import math

app = Flask(__name__)
CORS(app)

class MotorPrecisionToloza:
    def __init__(self):
        self.q = 0.000001  # Ruido de proceso
        self.lat_est = 0.0
        self.lon_est = 0.0
        self.p_error = 1.0
        self.inicializado = False

    def filtrar(self, lat, lon, accuracy):
        if not self.inicializado:
            self.lat_est, self.lon_est = lat, lon
            self.inicializado = True
            return lat, lon

        # 1. FILTRO DE DESCARTE POR ACCURACY
        if accuracy > 25:
            return self.lat_est, self.lon_est

        # 2. FILTRO DE UMBRAL (ZONA MUERTA)
        distancia = math.sqrt((lat - self.lat_est)**2 + (lon - self.lon_est)**2)
        if distancia < 0.00001:  # Ignora movimientos menores a ~1.5 metros
            return self.lat_est, self.lon_est

        # 3. FILTRO DE KALMAN DINÁMICO
        r = (accuracy * 0.00001) ** 2
        p_pred = self.p_error + self.q
        k = p_pred / (p_pred + r)
        
        self.lat_est = self.lat_est + k * (lat - self.lat_est)
        self.lon_est = self.lon_est + k * (lon - self.lon_est)
        self.p_error = (1 - k) * p_pred
        
        return self.lat_est, self.lon_est

motor = MotorPrecisionToloza()

@app.route('/procesar', methods=['POST'])
def procesar():
    try:
        data = request.json
        lat_in = float(data['lat'])
        lon_in = float(data['lon'])
        acc_in = float(data.get('accuracy', 15.0))
        
        l_f, ln_f = motor.filtrar(lat_in, lon_in, acc_in)
        
        return jsonify({
            "lat": l_f, 
            "lon": ln_f,
            "status": "Filtro Triple Activo"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)