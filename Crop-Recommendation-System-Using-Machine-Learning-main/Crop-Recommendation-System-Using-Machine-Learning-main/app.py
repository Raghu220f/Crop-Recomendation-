from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/contact')
def contact():
    return render_template("contactus.html")

@app.route('/about')
def about():
    return render_template("aboutus.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice" , 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    crop_comp= {
    "Rice": [
        "Sakthi Seeds - ABC has 3.9 ⭐ and 533 ratings with a price of Rs.380.",
        "Rasi Seeds - XYZ has 4.2 ⭐ and 417 ratings with a price of Rs.420.",
        "Mahyco Seeds - PQR has 4.5 ⭐ and 493 ratings with a price of Rs.390."
    ],
    "Maize": [
        "Kaveri Seeds - DEF has 4.0 ⭐ and 343 ratings with a price of Rs.480.",
        "Bioseed - GHI has 4.3 ⭐ and 393 ratings with a price of Rs.410.",
        "Nuziveedu Seeds - JKL has 4.1 ⭐ and 453 ratings with a price of Rs.440."
    ],
    "Jute": [
        "Ajeet Seeds - MNO has 3.7 ⭐ and 373 ratings with a price of Rs.360.",
        "Bioseeds - PQR has 4.0 ⭐ and 323 ratings with a price of Rs.390.",
        "National Seeds - STU has 3.8 ⭐ and 383 ratings with a price of Rs.370."
    ],
    "Cotton": [
        "Mayco Seeds - VWX has 3.9 ⭐ and 453 ratings with a price of Rs.390.",
        "Rasi Seeds - YZA has 4.1 ⭐ and 423 ratings with a price of Rs.430.",
        "VNR Seeds - BCD has 4.2 ⭐ and 493 ratings with a price of Rs.420."
    ],
    "Coconut": [
        "Bioseed - EFG has 3.8 ⭐ and 423 ratings with a price of Rs.430.",
        "Syngenta - HIJ has 4.0 ⭐ and 453 ratings with a price of Rs.450.",
        "Mahyco Seeds - KLM has 4.3 ⭐ and 503 ratings with a price of Rs.460."
    ],
    "Papaya": [
        "Bharat Seeds - NOP has 3.5 ⭐ and 303 ratings with a price of Rs.310.",
        "Kaveri Seeds - QRS has 3.8 ⭐ and 353 ratings with a price of Rs.340.",
        "Nuziveedu Seeds - TUV has 3.6 ⭐ and 323 ratings with a price of Rs.330."
    ],
    "Orange": [
        "Syngenta - WXY has 4.0 ⭐ and 443 ratings with a price of Rs.450.",
        "Bioseed - ZAB has 4.2 ⭐ and 483 ratings with a price of Rs.470.",
        "Mahyco Seeds - CDE has 4.4 ⭐ and 513 ratings with a price of Rs.480."
    ],
    "Apple": [
        "National Seeds - FGH has 4.1 ⭐ and 433 ratings with a price of Rs.440.",
        "Kaveri Seeds - IJK has 4.3 ⭐ and 463 ratings with a price of Rs.460.",
        "Bioseed - LMN has 4.5 ⭐ and 493 ratings with a price of Rs.490."
    ],
    "Muskmelon": [
        "Rasi Seeds - QRS has 3.8 ⭐ and 353 ratings with a price of Rs.340.",
        "Mahyco Seeds - TUV has 4.0 ⭐ and 393 ratings with a price of Rs.370.",
        "Syngenta - WXY has 4.2 ⭐ and 423 ratings with a price of Rs.390."
    ],
    "Watermelon": [
        "Bharat Seeds - YZA has 3.5 ⭐ and 303 ratings with a price of Rs.310.",
        "Bioseed - BCD has 3.8 ⭐ and 353 ratings with a price of Rs.340.",
        "Kaveri Seeds - EFG has 3.6 ⭐ and 323 ratings with a price of Rs.330."
    ],
    "Grapes": [
        "Nuziveedu Seeds - HIJ has 4.0 ⭐ and 443 ratings with a price of Rs.450.",
        "Syngenta - KLM has 4.2 ⭐ and 483 ratings with a price of Rs.470.",
        "Bioseed - NOP has 4.4 ⭐ and 513 ratings with a price of Rs.480."
    ],
    "Mango": [
        "Mahyco Seeds - STU has 4.1 ⭐ and 433 ratings with a price of Rs.440.",
        "National Seeds - VWX has 4.3 ⭐ and 463 ratings with a price of Rs.460.",
        "Kaveri Seeds - YZA has 4.5 ⭐ and 493 ratings with a price of Rs.490."
    ],
    "Banana": [
        "Rasi Seeds - ABC has 3.9 ⭐ and 533 ratings with a price of Rs.380.",
        "Kaveri Seeds - XYZ has 4.2 ⭐ and 417 ratings with a price of Rs.420.",
        "Mahyco Seeds - PQR has 4.5 ⭐ and 493 ratings with a price of Rs.390."
    ],
    "Pomegranate": [
        "Kaveri Seeds - DEF has 4.0 ⭐ and 343 ratings with a price of Rs.480.",
        "Bioseed - GHI has 4.3 ⭐ and 393 ratings with a price of Rs.410.",
        "Nuziveedu Seeds - JKL has 4.1 ⭐ and 453 ratings with a price of Rs.440."
    ],
    "Lentil": [
        "Ajeet Seeds - MNO has 3.7 ⭐ and 373 ratings with a price of Rs.360.",
        "Bioseeds - PQR has 4.0 ⭐ and 323 ratings with a price of Rs.390.",
        "National Seeds - STU has 3.8 ⭐ and 383 ratings with a price of Rs.370."
    ],
    "Blackgram": [
        "Mayco Seeds - VWX has 3.9 ⭐ and 453 ratings with a price of Rs.390.",
        "Rasi Seeds - YZA has 4.1 ⭐ and 423 ratings with a price of Rs.430.",
        "VNR Seeds - BCD has 4.2 ⭐ and 493 ratings with a price of Rs.420."
    ],
    "Mungbean": [
        "Bioseed - EFG has 3.8 ⭐ and 423 ratings with a price of Rs.430.",
        "Syngenta - HIJ has 4.0 ⭐ and 453 ratings with a price of Rs.450.",
        "VNR Seeds - BCD has 4.2 ⭐ and 493 ratings with a price of Rs.420."
    ],
    "Mothbeans": [
        "Bioseed - EFG has 3.8 ⭐ and 423 ratings with a price of Rs.430.",
        "Syngenta - HIJ has 4.0 ⭐ and 453 ratings with a price of Rs.450.",
        "VNR Seeds - BCD has 4.2 ⭐ and 493 ratings with a price of Rs.420."
    ],
     "Pigeonpeas": [
        "Bioseed - EFG has 3.8 ⭐ and 423 ratings with a price of Rs.430.",
        "Syngenta - HIJ has 4.0 ⭐ and 453 ratings with a price of Rs.450.",
        "VNR Seeds - BCD has 4.2 ⭐ and 493 ratings with a price of Rs.420."
    ],
    "Kidneybeans": [
        "Bioseed - EFG has 3.8 ⭐ and 423 ratings with a price of Rs.430.",
        "Syngenta - HIJ has 4.0 ⭐ and 453 ratings with a price of Rs.450.",
        "VNR Seeds - BCD has 4.2 ⭐ and 493 ratings with a price of Rs.420."
    ],
     "Chickpea": [
        "Bioseed - EFG has 3.8 ⭐ and 423 ratings with a price of Rs.430.",
        "Syngenta - HIJ has 4.0 ⭐ and 453 ratings with a price of Rs.450.",
        "VNR Seeds - BCD has 4.2 ⭐ and 493 ratings with a price of Rs.420."
    ],
    "Coffee": [
        "Bioseed - EFG has 3.8 ⭐ and 423 ratings with a price of Rs.430.",
        "Syngenta - HIJ has 4.0 ⭐ and 453 ratings with a price of Rs.450.",
        "VNR Seeds - BCD has 4.2 ⭐ and 493 ratings with a price of Rs.420."
    ]
    }



    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
        
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result,companies=crop_comp[crop])




# python main
if __name__ == "__main__":
    app.run(debug=True)