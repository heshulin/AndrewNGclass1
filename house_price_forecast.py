import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets,linear_model

#利用pandas获取csv中的数据
def getdata_from_csv(file_name):
    data = pd.read_csv(file_name)
    X_parameter = []
    Y_parameter = []
    for single_house_area,single_house_price in zip(data['area'],data['price']):
        X_parameter.append([float(single_house_area)])
        Y_parameter.append(float(single_house_price))
    return X_parameter,Y_parameter

#利用sklearn进行线性回归
def linear_model_main(X_parameter,Y_parameter,predict_value):
    regr = linear_model.LinearRegression()
    regr.fit(X_parameter,Y_parameter)
    predict_outcome = regr.predict(predict_value)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions

#利用matplotlib进行绘图
def show_linear_line(X_parameters,Y_parameters):
# Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    plt.scatter(X_parameters,Y_parameters,color='blue')
    plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)
    plt.xticks(())
    plt.yticks(())
    plt.show()

if __name__=="__main__":
    print("main")
    X, Y = getdata_from_csv('house_price.csv')
    predictvalue = 700
    result = linear_model_main(X, Y, predictvalue)
    print("Intercept value ", result['intercept'])
    print("coefficient", result['coefficient'])
    print("Predicted value: ", result['predicted_value'])
    show_linear_line(X, Y)