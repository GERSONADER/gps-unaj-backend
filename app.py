from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import math
import time

app = Flask(__name__)
CORS(app)

class KalmanGPS:
    def __init__(self):
        self.RT = 6378137.0
        self.lat0 = None
        self.lon0 = None

        # Estado: [x, y, vx, vy]^T
        self.X = np.zeros((4, 1))
        self.P = np.eye(4) # Se inicializará dinámicamente

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.I = np.eye(4)
        self.lastTime = None
        self.samples = 0
        self.outliers = 0
        self.lastSpeed = 0
        
        # Desviación estándar de aceleración (ajusta según el objetivo)
        # 0.5 a 1.0 m/s^2 es ideal para peatones. Para autos usa 2.0 o 3.0.
        self.sigma_a = 1.0 

    def latlon_to_xy(self, lat, lon):
        dlat = math.radians(lat - self.lat0)
        dlon = math.radians(lon - self.lon0)
        latm = math.radians((lat + self.lat0) / 2)
        x = self.RT * dlon * math.cos(latm)
        y = self.RT * dlat
        return np.array([[x], [y]])

    def xy_to_latlon(self, x, y):
        lat = self.lat0 + math.degrees(y / self.RT)
        lon = self.lon0 + math.degrees(x / (self.RT * math.cos(math.radians(self.lat0))))
        return lat, lon

    def procesar(self, lat, lon, accuracy):
        # 1. Límite inferior de seguridad para evitar divisiones por cero o sobreconfianza
        accuracy = max(accuracy, 3.0) 

        # INICIALIZACIÓN
        if self.lat0 is None:
            self.lat0 = lat
            self.lon0 = lon
            self.lastTime = time.time()
            self.X = np.zeros((4, 1))
            
            # Inicializar P basada en la precisión real del primer punto
            self.P = np.array([
                [accuracy**2, 0, 0, 0],
                [0, accuracy**2, 0, 0],
                [0, 0, (accuracy/2.0)**2, 0],
                [0, 0, 0, (accuracy/2.0)**2]
            ])
            self.samples += 1

            return {
                "lat": lat,
                "lon": lon,
                "speed": 0,
                "samples": self.samples,
                "status": "Inicializado",
                "outliers": 0,
                "gain": 1.0
            }

        ahora = time.time()
        dt = ahora - self.lastTime
        self.lastTime = ahora

        if dt <= 0:
            dt = 0.1 # Evitar dt cero

        # MATRIZ DE TRANSICIÓN DE ESTADO (A)
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # MATRIZ DE RUIDO DE PROCESO (Q) DINÁMICA
        # Modelo discreto para aceleración constante por tramos
        dt2 = dt**2
        dt3 = (dt**3) / 2.0
        dt4 = (dt**4) / 4.0
        q = self.sigma_a**2

        self.Q = np.array([
            [dt4*q, 0,     dt3*q, 0],
            [0,     dt4*q, 0,     dt3*q],
            [dt3*q, 0,     dt2*q, 0],
            [0,     dt3*q, 0,     dt2*q]
        ])

        # PREDICCIÓN
        self.X = A @ self.X
        self.P = A @ self.P @ A.T + self.Q

        # PREPARAR MEDICIÓN
        Z = self.latlon_to_xy(lat, lon)
        R = np.array([
            [accuracy**2, 0],
            [0, accuracy**2]
        ])

        # INNOVACIÓN
        Y = Z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + R
        
        # DISTANCIA DE MAHALANOBIS
        distancia = float(Y.T @ np.linalg.inv(S) @ Y)

        status_msg = "Kalman Activo"

        # MITIGACIÓN DE OUTLIERS SUAVE (Filtro Adaptativo)
        if distancia > 9.21:
            self.outliers += 1
            status_msg = "Outlier mitigado suavemente"
            # En lugar de descartar, inflamos R drásticamente para desconfiar de este punto,
            # pero permitimos que empuje ligerisimamente el vector para no perder giros reales.
            factor_castigo = distancia / 2.0
            R = R * factor_castigo
            S = self.H @ self.P @ self.H.T + R

        # CORRECCIÓN
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X = self.X + K @ Y
        self.P = (self.I - K @ self.H) @ self.P

        # CÁLCULOS FINALES
        vx = float(self.X[2, 0])
        vy = float(self.X[3, 0])
        speed = math.sqrt(vx*vx + vy*vy)
        self.lastSpeed = speed
        self.samples += 1

        latf, lonf = self.xy_to_latlon(float(self.X[0, 0]), float(self.X[1, 0]))
        ganancia_media = float(np.mean(np.diag(K)))

        return {
            "lat": latf,
            "lon": lonf,
            "speed": round(speed * 3.6, 2), # km/h
            "samples": self.samples,
            "status": status_msg,
            "outliers": self.outliers,
            "gain": round(ganancia_media, 4)
        }

# Instancia Global
gps = KalmanGPS()

@app.route("/")
def inicio():
    # Asegúrate de tener tu archivo en la carpeta "templates/index.html"
    return render_template("index.html")

@app.route("/procesar", methods=["POST"])
def procesar():
    try:
        datos = request.get_json()
        lat = float(datos["lat"])
        lon = float(datos["lon"])
        accuracy = float(datos.get("accuracy", 5.0))

        resultado = gps.procesar(lat, lon, accuracy)
        return jsonify(resultado)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
