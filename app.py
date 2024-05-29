from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file
import mysql.connector, os
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense,Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import backend as K
import joblib
def create_model(input_shape, num_classes_category, num_classes_days):
    # Input layer
    inputs = Input(shape=input_shape)

    # First Convolutional Layer
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    
    # Second Convolutional Layer
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Flatten the feature map
    x = Flatten()(x)
    
    # Output layer for category classification
    category_output = Dense(num_classes_category, activation='softmax', name='category_output')(x)
    
    # Output layer for days classification
    days_output = Dense(num_classes_days, activation='softmax', name='days_output')(x)
    
    # Custom layer to calculate probabilities
    def get_probs(x):
        return K.softmax(x)
    
    category_probs = Lambda(get_probs, name='category_probs')(category_output)
    days_probs = Lambda(get_probs, name='days_probs')(days_output)
    
    # Define the model with input and output layers
    model = Model(inputs=inputs, outputs=[category_output, days_output, category_probs, days_probs])
    
    return model
saved_model_path = 'cnn.h5'  
loaded_model = create_model(input_shape=(224, 224, 3), num_classes_category=2, num_classes_days=4)  
loaded_model.load_weights(saved_model_path)

app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='sapota'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])
        
        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')



@app.route('/rotten', methods=["GET", "POST"])
def rotten():
    if request.method == "POST":
        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join(r'static\images', fn)
        myfile.save(mypath)

        target_size = (224, 224)
        image = load_img(mypath, target_size=target_size)
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Predictions and probabilities
        category_pred, days_pred, category_probs, days_probs = loaded_model.predict(image)

        category_labels = ['Not Rotten', 'Rotten']
        days_labels = [str(i) for i in range(2, 6)]

        predicted_category = category_labels[np.argmax(category_pred)]
        current_day = days_labels[np.argmax(days_pred)]
        predicted_days = 8 - int(current_day)

        # Probabilities for predictions
        category_prob = category_probs[0][np.argmax(category_pred)]
        days_prob = days_probs[0][np.argmax(days_pred)]

        print("Predicted Category:", predicted_category)
        print("Predicted Days:", predicted_days)
        print("Category Probability:", category_prob)
        print("Days Probability:", days_prob)
        
        return render_template('rotten.html', mypath=mypath, predicted_category=predicted_category,
                               predicted_days=predicted_days, current_day=current_day,
                               category_prob=category_prob, days_prob=days_prob)
    return render_template('rotten.html')

def load_and_preprocess_image(image_path):
        # Load the image file, resizing it to 224x224 pixels (as expected by MobileNet)
        img = load_img(image_path, target_size=(224, 224))
        # Convert the image to a numpy array
        img_array = img_to_array(img)
        # Expand dimensions to match the shape expected by the pre-trained model: (1, 224, 224, 3)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        # Preprocess the image for the pre-trained model
        return preprocess_input(img_array_expanded)

# Load the feature extraction model
feature_extraction_model_path = 'feature_extractor.h5'
feature_extraction_model = load_model(feature_extraction_model_path)

# Load the scaler
scaler_path = 'scaler.save'
scaler = joblib.load(scaler_path)

# Load the SVM classifier
svm_classifier_path = 'svm_model.pkl'
svm_classifier = joblib.load(svm_classifier_path)

@app.route('/bruises', methods=["GET", "POST"])
def bruises():
    if request.method == "POST":
        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join('static/images/', fn)
        myfile.save(mypath)
        classes=["Bruises","No Bruises"]
        
        # Assuming you have a function to load and preprocess images named `load_and_preprocess_image`
        preprocessed_image = load_and_preprocess_image(mypath)

        # Extract features
        features = feature_extraction_model.predict(preprocessed_image)

        # Scale features
        scaled_features = scaler.transform(features.reshape(1, -1))

        # Predict with the SVM model
        predicted_class = svm_classifier.predict(scaled_features)
        print("Predicted class:", predicted_class)
        if predicted_class == 0:
            result = "Bruises"
        else:
            result = "No Bruises"

        return render_template('bruises.html', mypath = mypath, prediction = result)
    return render_template('bruises.html')

if __name__ == '__main__':
    app.run(debug = True)