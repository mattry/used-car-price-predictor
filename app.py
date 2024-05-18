from flask import Flask, render_template, request, jsonify, url_for
import json
import pickle
import numpy as np
app = Flask(__name__)

def read_files():
    global __data_columns
    global __manufacturers
    global __conditions
    global __fuels
    global __transmissions
    global __drives
    global __types
    global __paint_colors
    global __states
    global __model


    with open("columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        data = __data_columns
        __manufacturers = [column[13:] for column in data if column.startswith('manufacturer_')]
        __conditions = [column[10:] for column in data if column.startswith('condition_')]
        __fuels = [column[5:] for column in data if column.startswith('fuel_')]
        __transmissions = [column[13:] for column in data if column.startswith('transmission_')]
        __drives = [column[6:] for column in data if column.startswith('drive_')]
        __types = [column[5:] for column in data if column.startswith('type')]
        __paint_colors = [column[12:] for column in data if column.startswith('paint_color_')]
        __states = [column[6:] for column in data if column.startswith('state_')]
    
    with open("used_car_price_predictor.pickle", 'rb') as f:
        __model = pickle.load(f)

read_files()

def predict_price(manufacturer, condition, fuel, transmission, drive, type_, paint_color, state, year, odometer):
    # Construct column names for categorical variables
    manufacturer_column = 'manufacturer_' + manufacturer
    condition_column = 'condition_' + condition
    fuel_column = 'fuel_' + fuel
    transmission_column = 'transmission_' + transmission
    drive_column = 'drive_' + drive
    type_column = 'type_' + type_
    paint_color_column = 'paint_color_' + paint_color
    state_column = 'state_' + state
    
    # Check if all the categorical columns exist in the data columns
    if all(col in __data_columns for col in [manufacturer_column, condition_column, fuel_column, transmission_column, drive_column, type_column, paint_color_column, state_column]):
        # Find the indices of the specified categorical variables
        manufacturer_index = __data_columns.index(manufacturer_column)
        condition_index = __data_columns.index(condition_column)
        fuel_index = __data_columns.index(fuel_column)
        transmission_index = __data_columns.index(transmission_column)
        drive_index = __data_columns.index(drive_column)
        type_index = __data_columns.index(type_column)
        paint_color_index = __data_columns.index(paint_color_column)
        state_index = __data_columns.index(state_column)

        # Create an array of zeros with the length of __data_columns
        x = np.zeros(len(__data_columns))
        
        # Assign values to the features
        x[0] = year
        x[1] = odometer
        
        # Set the corresponding dummy variables to 1
        x[manufacturer_index] = 1
        x[condition_index] = 1
        x[fuel_index] = 1
        x[transmission_index] = 1
        x[drive_index] = 1
        x[type_index] = 1
        x[paint_color_index] = 1
        x[state_index] = 1

        # Make prediction
        return round(__model.predict([x])[0], 2)
    else:
        raise ValueError("One or more categorical columns not found in data columns.")

@app.route("/submit", methods=["POST"])
def submit():
    year = int(request.form['year'])
    odometer = int(request.form['odometer'])
    manufacturer = request.form['manufacturer']
    condition = request.form['condition']
    fuel = request.form['fuel']
    transmission = request.form['transmission']
    drive = request.form['drive']
    type_ = request.form['type']
    paint_color = request.form['paint_color']
    state = request.form['state']

    price = predict_price(manufacturer, condition, fuel, transmission, drive, type_, paint_color, state, year, odometer)

    return render_template('index.html', __manufacturers = __manufacturers, __types = __types, __fuels = __fuels,
                           __paint_colors = __paint_colors, __states = __states,
                             prediction_text =f"Your car is worth ${price}!")

@app.route('/')
def index():
    return render_template("index.html", __manufacturers = __manufacturers, __types = __types, __fuels = __fuels,
                           __paint_colors = __paint_colors, __states = __states)

@app.route('/visuals')
def visuals():
    return render_template("visuals.html")




if __name__ == "__main__":
    app.run()