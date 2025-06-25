from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import os
import cv2
import json
import numpy as np
from deepface import DeepFace

app = Flask(__name__, static_folder="dist")
CORS(app, resources={r"/*": {"origins": "*"}})

DATA_DIR = "registered_faces"
USERS_FILE = "users.json"
EMBEDDINGS_FILE = "embeddings.json"

os.makedirs(DATA_DIR, exist_ok=True)

# Cargar embeddings
if os.path.exists(EMBEDDINGS_FILE) and os.path.getsize(EMBEDDINGS_FILE) > 0:
    with open(EMBEDDINGS_FILE, "r") as f:
        embeddings_db = json.load(f)
else:
    embeddings_db = {}

# Cargar usuarios
if os.path.exists(USERS_FILE) and os.path.getsize(USERS_FILE) > 0:
    with open(USERS_FILE, "r") as f:
        users_db = json.load(f)
else:
    users_db = {}

# Función para obtener el embedding de una imagen
def get_deepface_embedding(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    try:
        embedding_obj = DeepFace.represent(img_path=img, model_name='Facenet', enforce_detection=True)[0]
        return embedding_obj['embedding']
    except:
        return None

# Función para limpiar y crear nombres de archivos
def limpiar_nombre(nombre):
    return nombre.strip().replace(" ", "_")

# Registro
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    nombre = data.get("nombre")
    apellido = data.get("apellido") or data.get("apellidos")  # Acepta ambos
    email = data.get("email")
    telefono = data.get("telefono")
    imagen_b64 = data.get("imagen")

    if not all([nombre, apellido, email, telefono, imagen_b64]):
        return jsonify({"error": "Faltan datos"}), 400

    if "," in imagen_b64:
        imagen_b64 = imagen_b64.split(",")[1]

    imagen_bytes = base64.b64decode(imagen_b64)
    nombre_archivo = f"{limpiar_nombre(nombre)}_{limpiar_nombre(apellido)}.png"
    filename = nombre_archivo
    filepath = os.path.join(DATA_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(imagen_bytes)

    embedding = get_deepface_embedding(imagen_bytes)
    if embedding is None:
        return jsonify({"error": "No se detectó rostro"}), 400

    embeddings_db[filename] = embedding
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings_db, f)

    users_db[filename] = {
        "nombre": nombre,
        "apellido": apellido,
        "email": email,
        "telefono": telefono
    }
    with open(USERS_FILE, "w") as f:
        json.dump(users_db, f)

    return jsonify({"message": "Registro exitoso", "archivo": filename}), 200

# Reconocimiento
@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.json
    imagen_b64 = data.get("imagen")
    if not imagen_b64:
        return jsonify({"error": "No se proporcionó imagen"}), 400

    if "," in imagen_b64:
        imagen_b64 = imagen_b64.split(",")[1]

    imagen_bytes = base64.b64decode(imagen_b64)
    input_embedding = get_deepface_embedding(imagen_bytes)
    if input_embedding is None:
        return jsonify({"error": "No se detectó rostro"}), 400

    mejor_match = None
    menor_distancia = float("inf")
    UMBRAL = 9.5  # DeepFace recomienda 0.7 para Facenet

    for filename, embedding_guardado in embeddings_db.items():
        distancia = np.linalg.norm(np.array(input_embedding) - np.array(embedding_guardado))
        print(f"Comparando con {filename}: distancia={distancia}")
        if distancia < menor_distancia:
            menor_distancia = distancia
            mejor_match = filename

    if menor_distancia < UMBRAL:
        user = users_db.get(mejor_match)
        return jsonify({
            "message": f"Rostro reconocido: {user['nombre']} {user['apellido']}",
            "nombre": user["nombre"],
            "apellido": user["apellido"],
            "email": user["email"],
            "telefono": user["telefono"],
            "distancia": menor_distancia
        }), 200
    else:
        return jsonify({"message": "No se encontró coincidencia", "distancia": menor_distancia}), 404

# Listar imágenes
@app.route("/images", methods=["GET"])
def list_images():
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return jsonify({"images": files})

# Servir imágenes de registered_faces
@app.route('/registered_faces/<path:filename>')
def serve_registered_face(filename):
    return send_from_directory(DATA_DIR, filename)

# --- ENDPOINT: Eliminar imagen y datos asociados ---
@app.route("/delete_image", methods=["POST"])
def delete_image():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "Falta filename"}), 400
    # Normalizar filename
    if filename.endswith('.png'):
        nombre, apellido = filename[:-4].split('_', 1)
        filename = f"{limpiar_nombre(nombre)}_{limpiar_nombre(apellido)}.png"

    # Eliminar imagen física
    img_path = os.path.join(DATA_DIR, filename)
    print("[DEBUG] Intentando borrar:", img_path)
    if os.path.exists(img_path):
        os.remove(img_path)
    else:
        print("[DEBUG] Imagen NO encontrada para borrar")

    # Eliminar usuario
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            users = json.load(f)
        if filename in users:
            del users[filename]
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump(users, f, indent=2, ensure_ascii=False)

    # Eliminar embedding
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            embeddings = json.load(f)
        if filename in embeddings:
            del embeddings[filename]
            with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, indent=2, ensure_ascii=False)

    # Recargar embeddings_db y users_db en memoria
    global embeddings_db, users_db
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            embeddings_db = json.load(f)
    else:
        embeddings_db = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            users_db = json.load(f)
    else:
        users_db = {}

    return jsonify({"message": "Imagen y datos eliminados"}), 200

# --- ENDPOINT: Editar datos de usuario ---
@app.route("/edit_user", methods=["POST"])
def edit_user():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "Falta filename"}), 400
    # Normalizar filename
    if filename.endswith('.png'):
        nombre, apellido = filename[:-4].split('_', 1)
        filename = f"{limpiar_nombre(nombre)}_{limpiar_nombre(apellido)}.png"

    nuevo_nombre = data.get("nombre")
    nuevo_apellido = data.get("apellido")
    nuevo_email = data.get("email")
    nuevo_telefono = data.get("telefono")

    if not (nuevo_nombre and nuevo_apellido and nuevo_email and nuevo_telefono):
        return jsonify({"error": "Faltan datos"}), 400

    if not os.path.exists(USERS_FILE):
        return jsonify({"error": "No hay usuarios"}), 404
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        users = json.load(f)
    if filename not in users:
        return jsonify({"error": "Usuario no encontrado"}), 404

    # Nuevo nombre de archivo
    nuevo_filename = f"{limpiar_nombre(nuevo_nombre)}_{limpiar_nombre(nuevo_apellido)}.png"

    # Renombrar archivo físico si el nombre cambió
    if filename != nuevo_filename:
        old_path = os.path.join(DATA_DIR, filename)
        new_path = os.path.join(DATA_DIR, nuevo_filename)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)

    # Renombrar clave en users.json
    users[nuevo_filename] = {
        "nombre": nuevo_nombre,
        "apellido": nuevo_apellido,
        "email": nuevo_email,
        "telefono": nuevo_telefono
    }
    if filename != nuevo_filename:
        users.pop(filename, None)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

    # Renombrar clave en embeddings.json
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            embeddings = json.load(f)
        if filename in embeddings:
            embeddings[nuevo_filename] = embeddings[filename]
            if filename != nuevo_filename:
                embeddings.pop(filename, None)
            with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, indent=2, ensure_ascii=False)

    # Recargar en memoria
    global embeddings_db, users_db
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            embeddings_db = json.load(f)
    else:
        embeddings_db = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            users_db = json.load(f)
    else:
        users_db = {}

    return jsonify({"message": "Datos actualizados", "nuevo_filename": nuevo_filename}), 200

# --- ENDPOINT: Obtener datos de usuario por filename ---
@app.route("/user_data", methods=["GET"])
def get_user_data():
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "Falta filename"}), 400
    # Normalizar filename
    if filename.endswith('.png'):
        nombre, apellido = filename[:-4].split('_', 1)
        filename = f"{limpiar_nombre(nombre)}_{limpiar_nombre(apellido)}.png"
    if not os.path.exists(USERS_FILE):
        return jsonify({"error": "No hay usuarios"}), 404
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        users = json.load(f)
    if filename not in users:
        return jsonify({"error": "Usuario no encontrado"}), 404
    return jsonify(users[filename]), 200

for filename, emb in embeddings_db.items():
    embeddings_db[filename] = [float(x) for x in emb]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
