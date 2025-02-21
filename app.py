from flask import Flask, request, jsonify
import pickle
import pandas as pd


app= Flask(__name__)


# Load the trained model
with open("model.pkl",'rb') as file:
    model = pickle.load(file)            # read


# Encoding Dictionary
dict1 = {
    "Gender":{"Male":1, "Female":0},
    "State":{"MH":0,"UP":1,"KA":2,"DL":3},
    "Policy_Type":{"Basic":0,"Premium":1},
    "Provider":{"P001":0,"P002":1,"P003":2},
    "Diagnosis":{"D001":0,"D002":1,"D003":2,"D004":3},
    "Procedure":{"PR001":0,"PR002":1,"PR003":2,"PR004":3}

}

def preprocess_input(d):
    for i, v in dict1.items():
        if i in d:
            d[i] = v.get(d[i],-1)
    return pd.DataFrame(d, index=[0])

@app.route("/predict",methods = ["POST"])
def predict():
    data = request.get_json()          # in dictonary

    # convert user input into df and preprocess categorical value
    df = preprocess_input(data)

    # make predict
    prediction = model.predict(df)[0]        # 0 1 


    # we show message base on prediction
    message = "Claim Approved" if prediction == 1 else "Claim Denied"

    return jsonify({"prediction": int(prediction),"message":message})

if __name__ == "__main__":
    app.run(debug=True)


