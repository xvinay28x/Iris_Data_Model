from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
app = Flask(__name__)

@app.route('/')
def man():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]],dtype="float64")
    pre = model.predict(arr)
    output = round(pre[0], 2)
    return render_template("index.html", result = output)

if __name__ == "__main__":
    app.run(debug=True)