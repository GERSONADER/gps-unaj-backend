from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import math
import time

app = Flask(__name__)
CORS(app)

class KalmanGPS:
    def __init__(self):
        self.RT = 6378137.0 # Radio de la Tierra
        self.lat0 = None
        self.lon0 = None
        self.X = np.zeros((4, 1))
        self.P = np.eye(4)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.I = np.eye(4)
        self.lastTime = None
        self.samples = 0
        self.outliers = 0
        self.lastSpeed = 0
        self.consecutive_outliers = 0 # NUEVO: Contador de saltos

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
        # Límite estricto de precisión
        accuracy = max(accuracy, 2.5) 

        # INICIALIZACIÓN o RESETEO
        if self.lat0 is None:
            self.lat0 = lat
            self.lon0 = lon
            self.lastTime = time.time()
            self.X = np.zeros((4, 1))
            self.P = np.array([
                [accuracy**2, 0, 0, 0],
                [0, accuracy**2, 0, 0],
                [0, 0, (accuracy/2.0)**2, 0],
                [0, 0, 0, (accuracy/2.0)**2]
            ])
            self.samples += 1
            self.consecutive_outliers = 0
            return {"lat": lat, "lon": lon, "speed": 0, "status": "Inicializado"}

        ahora = time.time()
        dt = ahora - self.lastTime
        self.lastTime = ahora
        if dt <= 0: dt = 0.1 

        # MATRIZ A (Cinemática)
        A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # PRECISIÓN EXTREMA: Ruido de proceso dinámico basado en velocidad
        # Si estás quieto (vel < 1 m/s), el sigma es muy bajo (0.1) para que el punto no tiemble.
        # Si te mueves, el sigma sube (0.8) para seguirte rápido.
        sigma_a = 0.1 if self.lastSpeed < 1.0 else 0.8
        
        dt2, dt3, dt4 = dt**2, (dt**3) / 2.0, (dt**4) / 4.0
        q = sigma_a**2

        self.Q = np.array([
            [dt4*q, 0,     dt3*q, 0],
            [0,     dt4*q, 0,     dt3*q],
            [dt3*q, 0,     dt2*q, 0],
            [0,     dt3*q, 0,     dt2*q]
        ])

        # PREDICCIÓN
        self.X = A @ self.X
        self.P = A @ self.P @ A.T + self.Q

        # MEDICIÓN
        Z = self.latlon_to_xy(lat, lon)
        R = np.array([[accuracy**2, 0], [0, accuracy**2]])
        Y = Z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + R
        
        # MAHALANOBIS (Mitigación de Outliers)
        distancia = float(Y.T @ np.linalg.inv(S) @ Y)
        status_msg = "Kalman Activo"

        if distancia > 9.21:
            self.outliers += 1
            self.consecutive_outliers += 1
            
            # PRECISIÓN EXTREMA: Si hay más de 4 saltos seguidos, el sistema asume que el salto es real 
            # (ej. saliste de un túnel) y resetea el filtro para no quedarse atascado.
            if self.consecutive_outliers > 4:
                self.lat0 = None
                return self.procesar(lat, lon, accuracy)
                
            status_msg = "Outlier mitigado"
            factor_castigo = distancia / 2.0
            R = R * factor_castigo
            S = self.H @ self.P @ self.H.T + R
        else:
            self.consecutive_outliers = 0 # Si el dato es bueno, resetea el contador de saltos

        # CORRECCIÓN
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X = self.X + K @ Y
        self.P = (self.I - K @ self.H) @ self.P

        vx = float(self.X[2, 0])
        vy = float(self.X[3, 0])
        self.lastSpeed = math.sqrt(vx*vx + vy*vy)
        self.samples += 1

        latf, lonf = self.xy_to_latlon(float(self.X[0, 0]), float(self.X[1, 0]))

        return {
            "lat": latf,
            "lon": lonf,
            "speed": round(self.lastSpeed * 3.6, 2),
            "status": status_msg
        }

# ==========================================
# SOLUCIÓN MULTI-USUARIO (EL DICCIONARIO)
# ==========================================
usuarios_activos = {}

@app.route("/")
def inicio():
    return render_template("index.html")

@app.route("/procesar", methods=["POST"])
def procesar():
    try:
        datos = request.get_json()
        lat = float(datos["lat"])
        lon = float(datos["lon"])
        accuracy = float(datos.get("accuracy", 5.0))
        
        # Recibimos el ID único del celular. Si no envía, le damos uno por defecto.
        user_id = datos.get("user_id", "default_user")

        # Si el usuario no existe en la memoria, le creamos su propio Filtro de Kalman
        if user_id not in usuarios_activos:
            usuarios_activos[user_id] = KalmanGPS()
            print(f"[NUEVO CLIENTE] Filtro asignado al usuario: {user_id}")

        # Procesamos la coordenada usando SOLO la memoria de ese usuario
        resultado = usuarios_activos[user_id].procesar(lat, lon, accuracy)
        
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
