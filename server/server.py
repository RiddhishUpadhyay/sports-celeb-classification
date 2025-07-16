from flask import Flask, request, jsonify, send_from_directory
import util

app = Flask(__name__,static_folder="../ui", static_url_path="", template_folder="../ui")

@app.route("/")
def serve_ui():
    return send_from_directory("../ui", "app.html")

@app.route("/img/<path:filename>")
def serve_image(filename):
    return send_from_directory("../ui/img", filename)

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory("../ui", filename)

@app.route('/classify_image', methods=['POST'])
def classify_image():
    image_data = request.form.get('image_data')
    
    if not image_data:
        return jsonify([])

    result = util.classify_image(image_data)
    
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')  # fixed typo
    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    app.run(port=5000)