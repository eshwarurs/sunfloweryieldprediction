from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
app = Flask(__name__)   # Initializing flask



@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods = ["GET", "POST"])
def predict():
    if request.method == "POST":
        
        # water
        water = float(request.form["water"])
        
        # UV
        UV = float(request.form["UV"])
        
        # area
        area = float(request.form["area"])
        
        # fertilizer
        fertilizer = float(request.form["fertilizer"])
        
        # Pesticide
        Pesticide = float(request.form["Pesticide"])
        
        # Region
        Region = float(request.form["Region"])

        sample_data = [water,UV,area,fertilizer,Pesticide,Region]
        clean_data = [float(i) for i in sample_data]
        ex1 = np.array(clean_data).reshape(1,-1)


        data=pd.read_csv(r"dataset.csv")
        data=data.drop(columns=['id','categories'],axis=1)

        data.water = pd.DataFrame(data.water).replace(np.nan, pd.DataFrame.median(data.water))#Replacing NaN with median
        data.uv = pd.DataFrame(data.uv).replace(np.nan, pd.DataFrame.median(data.uv))#Replacing NaN with median

        data.drop(data[data['water']>200].index,inplace=True)

        X=data.iloc[:,:-1].values
        y=data.iloc[:,-1].values
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
        
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 50)
        regressor.fit(X_train, y_train)
        yhat=regressor.predict(ex1)
        res = yhat[0]  

        result= 27.6*res
                
    return render_template('CropResult.html', prediction_text=res, result=result)
        

if __name__ == '__main__':
    app.run(debug=True)