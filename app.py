from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Tải các mô hình đã lưu
linear_model = joblib.load('model/linear_model.joblib')
lasso_model = joblib.load('model/lasso_model.joblib')
mlp_model = joblib.load('model/mlp_model.joblib')
stacking_model = joblib.load('model/stacking_model.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    form_data = {}  # Dictionary to store form data
    if request.method == 'POST':
        # Nhận dữ liệu từ form
        form_data['Density'] = request.form['Density']
        form_data['Weight'] = request.form['Weight']
        form_data['Chest'] = request.form['Chest']
        form_data['Abdomen'] = request.form['Abdomen']
        form_data['Hip'] = request.form['Hip']
        form_data['Thigh'] = request.form['Thigh']
        form_data['Knee'] = request.form['Knee']
        form_data['Biceps'] = request.form['Biceps']
        form_data['method'] = request.form['method']

        # Chuyển đổi danh sách thành mảng hai chiều
        input_data = [
            float(form_data['Density']),
            float(form_data['Weight']),
            float(form_data['Chest']),
            float(form_data['Abdomen']),
            float(form_data['Hip']),
            float(form_data['Thigh']),
            float(form_data['Knee']),
            float(form_data['Biceps']),
        ]
        input_array = np.array(input_data).reshape(1, -1)

        # Nhận phương pháp từ form
        method = form_data['method']
        if method == 'Linear Regression':
            prediction = linear_model.predict(input_array)
        elif method == 'Lasso':
            prediction = lasso_model.predict(input_array)
        elif method == 'MLP':
            prediction = mlp_model.predict(input_array)
        elif method == 'Stacking':
            prediction = stacking_model.predict(input_array)

    return render_template('index.html', prediction=prediction, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
