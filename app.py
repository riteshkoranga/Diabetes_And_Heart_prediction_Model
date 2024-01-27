from flask import Flask,request,jsonify,render_template,redirect,url_for
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn  import svm

app=Flask(__name__)


filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))






@app.route("/")
def diabetes():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")


##############################################
#loading the diabetes dataset to pandas dataframe
df1=pd.read_csv('diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df1 = df1.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df1.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure',
                                                                                    'SkinThickness', 'Insulin',
                                                                                    'BMI']].replace(0, np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)


df1.groupby('Outcome').mean()
#separating the data and labels
x=df1.drop(columns='Outcome',axis=1)
y=df1['Outcome']

##Data standardisation
scaler=StandardScaler()
scaler.fit(x)
standard_data=scaler.transform(x)

x=standard_data
y=df1['Outcome']

##train and test split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


##Training  the model
classifier=svm.SVC(kernel='linear')
#trainig the support vector machine
classifier.fit(x_train,y_train)
#####################################################################


@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        data = np.asarray([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        #reshape the array as we are predicting on one instance
        input_data_reshaped=data.reshape(1,-1)

        # standardise the input data
        std_data=scaler.transform(input_data_reshaped)
        my_prediction = classifier.predict(std_data)

        return render_template('dia_result.html', prediction=my_prediction)






#####################################################
# Heart deasease prediction
 #loading the csv data
heart_data=pd.read_csv('heart.csv')

    #checking for  missing values
heart_data.isnull().sum()

    #checking the distribution of target variable
heart_data['target'].value_counts()

    #Splitting the features and target
x=heart_data.drop(columns='target',axis=1)
y=heart_data['target']

    #Splitting the data into training data and test data
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

    #Model training
    #Logistic regression
model=LogisticRegression()

    #training the Logisticregression model with data
model.fit(x_train,y_train)    

   





@app.route('/predictheart', methods=['POST'])
def predictheart():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

    data = pd.DataFrame(features_value, columns=features_name)
    #reshape the array as we are predicting on one instance
    #input_data_reshaped=data.reshape(1,-1)
    output=model.predict(data)

    if output == 1:
        res_val = "A high risk of Heart Disease"
    else:
        res_val = "Low risk of Heart Disease"

    return render_template('heart_result.html', prediction_text='Patient has {}'.format(res_val))

if __name__=="__main__":
    app.run(debug=True)