from flask import Flask, render_template, jsonify, request
import json

app = Flask(__name__)

# Load JSON data
with open('predicted_clusters.json', 'r') as f:
    clusters = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_clusters', methods=['GET'])
def get_clusters():
    return jsonify(clusters)

@app.route('/update_clusters', methods=['POST'])
def update_clusters():
    data = request.json
    with open('predicted_clusters.json', 'w') as f:
        json.dump(data, f, indent=4)
    return jsonify({'status': 'success', 'message': 'Clusters updated successfully'})

if __name__ == '__main__':
    app.run(debug=True)
