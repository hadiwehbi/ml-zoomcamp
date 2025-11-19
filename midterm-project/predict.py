import pickle
from flask import Flask, request, jsonify

# load model
with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("house-price")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    house = request.get_json()

    X = dv.transform([house])
    y_pred = model.predict(X)[0]

    result = {
        "SalePrice": float(y_pred)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
