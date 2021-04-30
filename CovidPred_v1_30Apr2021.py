import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output,Input
import numpy as np
import datetime as dt
from datetime import timedelta

#from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

import statsmodels.api as sm
from statsmodels.tsa.api import Holt#,SimpleExpSmoothing,ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller

# from pyramid.arima import auto_arima

std=StandardScaler()

app=dash.Dash(__name__)
df = pd.read_csv("owid-covid-data(1).csv")
#App Layout

app.layout=html.Div([
    html.H1("Progression of Covid"),
    dcc.Dropdown(id="my_option",
                 options=[{'label':i,'value':i}
                          for i in df["location"].unique()],
                 value="Afghanistan"

                 ),
    html.Br(),
    html.Div(id="deathno",title="Deaths",draggable="true"),
    dcc.Graph(id="fig_Lreg",figure={}),
    dcc.Graph(id="fig_Polyreg",figure={}),
    dcc.Graph(id="fig_Holtreg",figure={})


])

#call back

@app.callback(
    [Output(component_id="fig_Lreg",component_property="figure"),
     Output(component_id="fig_Polyreg",component_property="figure"),
     Output(component_id="fig_Holtreg",component_property="figure")],
    Input(component_id="my_option",component_property="value")
)


def update_graph(option_slctd):
    # filterdata=df[df["location"]==option_slctd]
    # deaths=int(filterdata["new_deaths"].sum())


    opted_country = option_slctd #'Brazil'  # input("Select the country - ")
    print(opted_country)
    dt_one_country = df[df["location"] == opted_country][['date', 'new_cases']]
    dt_one_country['new_cases'] = dt_one_country['new_cases'].fillna(0)
    dt_one_country['date'] = pd.to_datetime(dt_one_country['date'])
    dt_one_country['Days Since'] = dt_one_country['date'] - dt_one_country['date'].min()
    dt_one_country['Days Since'] = dt_one_country['Days Since'].dt.days

    train_ml = dt_one_country.iloc[:int(dt_one_country.shape[0] * 0.95)]
    valid_ml = dt_one_country.iloc[int(dt_one_country.shape[0] * 0.95):]

    fitinput_x = np.array(train_ml["Days Since"]).reshape(-1, 1);
    fitinput_y = np.array(train_ml["new_cases"]).reshape(-1, 1)
    lin_reg = LinearRegression(normalize=True)

    lin_reg.fit(fitinput_x, fitinput_y)
    x_pred = np.array(valid_ml["Days Since"]).reshape(-1, 1)
    y_pred = lin_reg.predict(x_pred)

    # lin_reg.score(x_pred,valid_ml['new_cases'])
    print(np.sqrt(mean_squared_error(valid_ml["new_cases"], y_pred)))

    plt.figure(figsize=(11, 6))
    prediction_linreg = lin_reg.predict(np.array(dt_one_country["Days Since"]).reshape(-1, 1))
    linreg_output = []

    for i in range(prediction_linreg.shape[0]):
        linreg_output.append(prediction_linreg[i][0])

    fig_LinearReg = go.Figure()
    fig_LinearReg.add_trace(go.Scatter(x=dt_one_country['date'], y=dt_one_country["new_cases"],
                             mode='lines+markers', name="Train Data for new Cases"))
    fig_LinearReg.add_trace(go.Scatter(x=dt_one_country['date'], y=linreg_output,
                             mode='lines', name="Linear Regression Best Fit Line",
                             line=dict(color='black', dash='dot')))
    fig_LinearReg.update_layout(title="new Cases Linear Regression Prediction " +str(opted_country),
                      xaxis_title="Date", yaxis_title="new Cases", legend=dict(x=0, y=1, traceorder="normal"))
    # fig.show()

    poly = PolynomialFeatures(degree=8)
    train_poly = poly.fit_transform(fitinput_x)

    fitin_valid = np.array(valid_ml["Days Since"]).reshape(-1, 1)
    valid_poly = poly.fit_transform(fitin_valid)
    y_train_to_compare = train_ml['new_cases']

    lin_reg = LinearRegression(normalize=True)
    lin_reg.fit(train_poly, y_train_to_compare)

    prediction_poly = lin_reg.predict(valid_poly)
    lin_reg.score(valid_poly, valid_ml['new_cases'].values)
    print(np.sqrt(mean_squared_error(valid_ml["new_cases"], prediction_poly)))

    additional_30days = np.linspace(1, 30, 30)
    pred_input_compiled_data = []
    pred_input_compiled_data = np.array(dt_one_country["Days Since"]).reshape(-1, 1)
    pred_input_compiled_data = np.append(pred_input_compiled_data, pred_input_compiled_data[-1] + additional_30days)

    # add_pred_dates = pd.DataFrame(columns=['date'])
    add_pred_dates = dt_one_country['date']

    for i in range(1, 31):
        add_pred_dates = add_pred_dates.append(add_pred_dates.iloc[-1:] + timedelta(days=1), ignore_index=True)  #

    # comp_data=poly.fit_transform(np.array(dt_one_country["Days Since"]).reshape(-1,1))
    comp_data = poly.fit_transform(pred_input_compiled_data.reshape(-1, 1))
    plt.figure(figsize=(11, 6))
    predictions_poly = lin_reg.predict(comp_data)

    fig_PolyReg = go.Figure()
    fig_PolyReg.add_trace(go.Scatter(x=dt_one_country['date'], y=dt_one_country["new_cases"],
                             mode='lines+markers', name="Train Data for new Cases in "+str(opted_country)))
    # fig.add_trace(go.Scatter(x=dt_one_country['date'], y=predictions_poly,
    fig_PolyReg.add_trace(go.Scatter(x=add_pred_dates, y=predictions_poly,
                             mode='lines', name="Polynomial Regression Best Fit",
                             line=dict(color='black', dash='dot')))
    fig_PolyReg.update_layout(title="new Cases Polynomial Regression Prediction",
                      xaxis_title="Date", yaxis_title="new Cases",
                      legend=dict(x=0, y=1, traceorder="normal"))
    # fig.show()

    # train_ml=dt_one_country.iloc[:int(dt_one_country.shape[0]*0.95)]
    # valid_ml=dt_one_country.iloc[int(dt_one_country.shape[0]*0.95):]

    model_train = dt_one_country.iloc[:int(dt_one_country.shape[0] * 0.95)]
    valid = dt_one_country.iloc[int(dt_one_country.shape[0] * 0.95):]
    y_pred = valid.copy()

    holt = Holt(np.asarray(model_train["new_cases"])).fit(smoothing_level=0.9, smoothing_trend=0.4, optimized=False)
    y_pred["Holt"] = holt.forecast(len(valid))
    # y_holt_pred["Holt"]=holt.forecast(len(valid)+30)
    print(np.sqrt(mean_squared_error(y_pred["new_cases"], y_pred["Holt"])))

    fig_Holt = go.Figure()
    fig_Holt.add_trace(go.Scatter(x=model_train['date'], y=model_train["new_cases"],
                             mode='lines+markers', name="Train Data for new Cases " +str(opted_country)))
    fig_Holt.add_trace(go.Scatter(x=valid['date'], y=valid["new_cases"],
                             mode='lines+markers', name="Validation Data for new Cases "+str(opted_country) ))
    fig_Holt.add_trace(go.Scatter(x=valid['date'], y=y_pred["Holt"],
                             mode='lines+markers', name="Prediction of new Cases "+str(opted_country) ))
    fig_Holt.update_layout(title="new Cases Holt's Linear Model Prediction",
                      xaxis_title="Date", yaxis_title="new Cases", legend=dict(x=0, y=1, traceorder="normal"))
    # fig.show()

    # fig=px.line(filterdata,x="date",y="total_cases")
    # fig2=px.line(filterdata,x="date",y="new_cases",title="New Cases With Date")

    return fig_LinearReg,fig_PolyReg,fig_Holt #fig,fig2

if __name__ == '__main__':
    app.run_server(debug=True)



# x_train = train_ml['Days Since']
# y_train_1 = train_ml['new_cases']
# y_train_1 = y_train_1.astype('float64')
# y_train_1 = y_train_1.apply(lambda x: np.log1p(x))
# y_train_1.replace([np.inf, -np.inf], 0, inplace=True)
# x_test = valid_ml['Days Since']
# y_test = valid_ml['new_cases']
# # y_test = y_test.astype('float64')
# # y_test = y_test.apply(lambda x: np.log1p(x))
# # y_test.replace([np.inf, -np.inf], 0, inplace=True)
#
# regr = LinearRegression(normalize=True)
# regr.fit(np.array(x_train).reshape(-1,1),np.array(y_train_1).reshape(-1,1))
#
# ypred = regr.predict(np.array(x_test).reshape(-1,1))
# print(np.sqrt(mean_squared_error(y_test,np.expm1(ypred))))
#
#
# # Plot results
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
#
# ax1.plot(valid_ml['date'], np.expm1(ypred))
# ax1.plot(dt_one_country['date'], dt_one_country['new_cases'])
# ax1.axvline(valid_ml['date'].iloc[0], linewidth=2, ls=':', color='grey', alpha=0.5)
# ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')
# ax1.set_xlabel("Day count ")
# ax1.set_ylabel("new Cases")
#
# ax2.plot(valid_ml['date'], ypred)
# ax2.plot(dt_one_country['date'], np.log1p(dt_one_country['new_cases']))
# ax2.axvline(valid_ml['date'].iloc[0], linewidth=2, ls=':', color='grey', alpha=0.5)
# ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')
# ax2.set_xlabel("Day count ")
# ax2.set_ylabel("Logarithm new Cases")
#
# plt.suptitle(("newCases predictions based on Log-Lineal Regression for " + opted_country))


#
# model_sarima= auto_arima(model_train["new_cases"],trace=True, error_action='ignore',
#                          start_p=0,start_q=0,max_p=2,max_q=2,m=7,
#                    suppress_warnings=True,stepwise=True,seasonal=True)
#
# model_sarima.fit(model_train["new_cases"])
#
# prediction_sarima=model_sarima.predict(len(valid))
# y_pred["SARIMA Model Prediction"]=prediction_sarima
#
# print("Root Mean Square Error for SARIMA Model: ",np.sqrt(mean_squared_error(y_pred["new_cases"],y_pred["SARIMA Model Prediction"])))
#
# fig=go.Figure()
# fig.add_trace(go.Scatter(x=model_train['date'], y=model_train["new_cases"],
#                     mode='lines+markers',name="Train Data for new Cases for %{opted_country}"))
# fig.add_trace(go.Scatter(x=valid['date'], y=valid["new_cases"],
#                     mode='lines+markers',name="Validation Data for new Cases",))
# fig.add_trace(go.Scatter(x=valid['date'], y=y_pred["SARIMA Model Prediction"],
#                     mode='lines+markers',name="Prediction for new Cases",))
# fig.update_layout(title="new Cases SARIMA Model Prediction",
#                  xaxis_title="Date",yaxis_title="new cases",legend=dict(x=0,y=1,traceorder="normal"))
# fig.show()

