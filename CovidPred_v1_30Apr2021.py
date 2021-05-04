# https://www.kaggle.com/neelkudu28/covid-19-visualizations-predictions-forecasting
# https://www.kaggle.com/saga21/covid-global-forecast-sir-model-ml-regressions
import warnings
warnings.filterwarnings('ignore')
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
def dt_process(df2,option_slctd):

    df = df2.copy()
    opted_country = option_slctd  # 'Brazil'  # input("Select the country - ")
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
    model_scores = []
    # lin_reg.score(x_pred,valid_ml['new_cases'])
    # print(np.sqrt(mean_squared_error(valid_ml["new_cases"], y_pred)))

    # plt.figure(figsize=(11, 6))
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
    fig_LinearReg.add_vline(x=valid_ml['date'].iloc[0], line_dash="dash")  # ,
    fig_LinearReg.update_layout(title="new Cases Linear Regression Prediction " + str(opted_country),
                                xaxis_title="Date", yaxis_title="new Cases", legend=dict(x=0, y=1, traceorder="normal"))
    # fig_LinearReg.show()

    poly = PolynomialFeatures(degree=8)
    train_poly = poly.fit_transform(fitinput_x)

    fitin_valid = np.array(valid_ml["Days Since"]).reshape(-1, 1)
    valid_poly = poly.fit_transform(fitin_valid)
    y_train_to_compare = train_ml['new_cases']

    lin_reg = LinearRegression(normalize=True)
    lin_reg.fit(train_poly, y_train_to_compare)

    prediction_poly = lin_reg.predict(valid_poly)
    lin_reg.score(valid_poly, valid_ml['new_cases'].values)
    # print(np.sqrt(mean_squared_error(valid_ml["new_cases"], prediction_poly)))
    model_scores.append(np.sqrt(mean_squared_error(valid_ml["new_cases"], prediction_poly)))
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
    # plt.figure(figsize=(11, 6))
    predictions_poly = lin_reg.predict(comp_data)

    fig_PolyReg = go.Figure()
    fig_PolyReg.add_trace(go.Scatter(x=dt_one_country['date'], y=dt_one_country["new_cases"],
                                     mode='lines+markers', name="Train Data for new Cases in " + str(opted_country)))
    # fig.add_trace(go.Scatter(x=dt_one_country['date'], y=predictions_poly,
    fig_PolyReg.add_trace(go.Scatter(x=add_pred_dates, y=predictions_poly,
                                     mode='lines', name="Polynomial Regression Best Fit",
                                     line=dict(color='black', dash='dot')))
    fig_PolyReg.add_vline(x=valid_ml['date'].iloc[0], line_dash="dash")  # ,
    fig_PolyReg.update_layout(title="new Cases Polynomial Regression Prediction",
                              xaxis_title="Date", yaxis_title="new Cases",
                              legend=dict(x=0, y=1, traceorder="normal"))
    # fig_PolyReg.show()

    # train_ml=dt_one_country.iloc[:int(dt_one_country.shape[0]*0.95)]
    # valid_ml=dt_one_country.iloc[int(dt_one_country.shape[0]*0.95):]

    model_train = dt_one_country.iloc[:int(dt_one_country.shape[0] * 0.95)]
    valid = dt_one_country.iloc[int(dt_one_country.shape[0] * 0.95):]
    y_pred = valid.copy()

    holt = Holt(np.asarray(model_train["new_cases"])).fit(smoothing_level=0.9, smoothing_trend=0.4, optimized=False)
    y_pred["Holt"] = holt.forecast(len(valid))
    # y_holt_pred["Holt"]=holt.forecast(len(valid)+30)
    # print(np.sqrt(mean_squared_error(y_pred["new_cases"], y_pred["Holt"])))
    model_scores.append(np.sqrt(mean_squared_error(y_pred["new_cases"], y_pred["Holt"])))

    fig_Holt = go.Figure()
    fig_Holt.add_trace(go.Scatter(x=model_train['date'], y=model_train["new_cases"],
                                  mode='lines+markers', name="Train Data for new Cases " + str(opted_country)))
    fig_Holt.add_trace(go.Scatter(x=valid['date'], y=valid["new_cases"],
                                  mode='lines+markers', name="Validation Data for new Cases " + str(opted_country)))
    fig_Holt.add_vline(x=valid['date'].iloc[0], line_dash="dash")  # ,
    fig_Holt.add_trace(go.Scatter(x=valid['date'], y=y_pred["Holt"],
                                  mode='lines+markers', name="Prediction of new Cases " + str(opted_country)))
    fig_Holt.update_layout(title="new Cases Holt's Linear Model Prediction",
                           xaxis_title="Date", yaxis_title="new Cases", legend=dict(x=0, y=1, traceorder="normal"))
    # fig_Holt.show()

    x_train = train_ml['Days Since']
    y_train_1 = train_ml['new_cases']
    y_train_1 = y_train_1.astype('float64')
    y_train_1 = y_train_1.apply(lambda x: np.log1p(x))
    y_train_1.replace([np.inf, -np.inf], 0, inplace=True)
    x_test = valid_ml['Days Since']
    y_test = valid_ml['new_cases']
    # y_test = y_test.astype('float64')
    # y_test = y_test.apply(lambda x: np.log1p(x))
    # y_test.replace([np.inf, -np.inf], 0, inplace=True)

    regr = LinearRegression(normalize=True)
    regr.fit(np.array(x_train).reshape(-1, 1), np.array(y_train_1).reshape(-1, 1))

    ypred = regr.predict(np.array(x_test).reshape(-1, 1))
    # print(np.sqrt(mean_squared_error(y_test, np.expm1(ypred))))

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

    train_days = int(dt_one_country.shape[0] * 0.95)  # allocating for 50days testing,prediction
    test_days = dt_one_country['Days Since'].iloc[-1] - train_days
    lag_size = 30
    lagpred_data_features = dt_one_country.copy()
    lagpred_data_features = calculate_lag(lagpred_data_features, range(1, lag_size), 'new_cases')

    filter_col_new_cases = [col for col in lagpred_data_features if col.startswith('new_cases')]
    lagpred_data_features[filter_col_new_cases] = lagpred_data_features[filter_col_new_cases].apply(
        lambda x: np.log1p(x))
    lagpred_data_features.replace([np.inf, -np.inf], 0, inplace=True)
    lagpred_data_features.fillna(0, inplace=True)

    start_fcst = 1 + lagpred_data_features['Days Since'].iloc[train_days]  # prediction day 1
    end_fcst = lagpred_data_features['Days Since'].iloc[-1]  # prediction day 30

    for d in list(range(start_fcst, end_fcst + 1)):
        X_train, Y_train_1, X_test = split_data_one_day(lagpred_data_features, d)
        model_1, pred_1 = lin_reg_lag(X_train, Y_train_1, X_test)
        lagpred_data_features.new_cases.iloc[d] = pred_1

        # Recompute lags
        lagpred_data_features = calculate_lag(lagpred_data_features, range(1, lag_size), 'new_cases')

        lagpred_data_features.replace([np.inf, -np.inf], 0, inplace=True)
        lagpred_data_features.fillna(0, inplace=True)

        # print("Process for ", country_name, "finished in ", round(time.time() - ts, 2), " seconds")

    predicted_data = lagpred_data_features.new_cases
    real_data = dt_one_country.new_cases
    # dates_list_num = list(range(0,len(dates_list)))
    dates_list_num = dt_one_country['date']
    # Plot results
    model_scores.append(np.sqrt(mean_squared_error(real_data, np.expm1(predicted_data))))
    fig_LagPred = go.Figure()
    fig_LagPred.add_trace(go.Scatter(x=dates_list_num, y=np.expm1(predicted_data),
                                     mode='lines+markers', name="Prediction new Cases " + str(opted_country)))
    fig_LagPred.add_trace(go.Scatter(x=dates_list_num, y=real_data,
                                     mode='lines+markers', name="Validation Data for new Cases " + str(opted_country)))
    fig_LagPred.add_vline(x=dates_list_num.iloc[start_fcst], line_dash="dash")  # ,
    # annotation=dict())#, annotation_position="top right")
    # fig_LagPred.add_trace(go.Scatter(x=valid['date'], y=y_pred["Holt"],
    #                               mode='lines+markers', name="Prediction of new Cases " + str(opted_country)))
    fig_LagPred.update_layout(title="new Cases Linear Lagged Model Prediction",
                              xaxis_title="Date", yaxis_title="new Cases", legend=dict(x=0, y=1, traceorder="normal"))

    # fig_LagPred.show()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
    #
    # ax1.plot(dates_list_num, np.expm1(predicted_data))
    # ax1.plot(dates_list_num, real_data)
    # ax1.axvline(dates_list_num.iloc[start_fcst], linewidth=2, ls = ':', color='grey', alpha=0.5)
    # ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')
    # ax1.set_xlabel("Day count ")
    # ax1.set_ylabel("new Cases")
    #
    # ax2.plot(dates_list_num, predicted_data)
    # ax2.plot(dates_list_num, np.log1p(real_data))
    # ax2.axvline(dates_list_num.iloc[start_fcst], linewidth=2, ls = ':', color='grey', alpha=0.5)
    # ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')
    # ax2.set_xlabel("Day count ")
    # ax2.set_ylabel("Log new Cases")

    # plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for "+country_name))
    model_names = ["Polynomial Regression","Holts Linear Prediction","Linear Regression Lagged Model"]
    model_summary = pd.DataFrame(zip(model_names, model_scores),
                                 columns=["Model Name", "Root Mean Squared Error"]).sort_values(
        ["Root Mean Squared Error"])
    print(model_summary)
    return fig_PolyReg, fig_Holt, fig_LagPred

def calculate_lag(df, lag_list, column):
    for lag in lag_list:
        column_lag = column + "_" + str(lag)
        df[column_lag] = df[column].shift(lag, fill_value=0)
    return df


# New split function, for one forecast day
def split_data_one_day(df, d):#, train_lim, test_lim):
    # df.loc[df['Day_num'] <= train_lim, 'ForecastId'] = -1
    # df = df[df['Day_num'] <= test_lim]

    # Train
    x_train_lag = df[df['Days Since'] < d]
    y_train_lag = x_train_lag.new_cases#ConfirmedCases
    # y_train_2 = x_train.Fatalities
    x_train_lag.drop(['date','new_cases'], axis=1, inplace=True)

    # Test
    x_test_lag = df[df['Days Since'] == d]
    x_test_lag.drop(['date','new_cases'], axis=1, inplace=True)

    # Clean Id columns and keep ForecastId as index
    # x_train.drop('Id', inplace=True, errors='ignore', axis=1)
    # x_train.drop('ForecastId', inplace=True, errors='ignore', axis=1)
    # x_test.drop('Id', inplace=True, errors='ignore', axis=1)
    # x_test.drop('ForecastId', inplace=True, errors='ignore', axis=1)

    return x_train_lag, y_train_lag, x_test_lag


# Linear regression model
def lin_reg_lag(X_train, Y_train, X_test):
    # Create linear regression object
    regr = LinearRegression(normalize=True)

    # Train the model using the training sets
    regr.fit(X_train, Y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    return regr, y_pred



std=StandardScaler()
option_slctd = 'India'
app=dash.Dash(__name__)
df = pd.read_csv("owid-covid-data(1).csv")

# App Layout

app.layout=html.Div([
    html.H1("Progression of Covid"),
    dcc.Dropdown(id="my_option",
                 options=[{'label':i,'value':i}
                          for i in df["location"].unique()],
                 value="Afghanistan"

                 ),
    html.Br(),
    html.Div(id="deathno",title="Deaths",draggable="true"),
    dcc.Graph(id="fig_PolyReg",figure={}),
    dcc.Graph(id="fig_Holt",figure={}),
    dcc.Graph(id="fig_LagPred",figure={})


])

#call back

@app.callback(
    [Output(component_id="fig_PolyReg",component_property="figure"),
     Output(component_id="fig_Holt",component_property="figure"),
     Output(component_id="fig_LagPred",component_property="figure")],
    Input(component_id="my_option",component_property="value")
)


def update_graph(option_slctd):
    # filterdata=df[df["location"]==option_slctd]
    # deaths=int(filterdata["new_deaths"].sum())

    fig_PolyReg_ret, fig_Holt_ret, fig_LagPred_ret = dt_process(df, option_slctd)

    # fig=px.line(filterdata,x="date",y="total_cases")
    # fig2=px.line(filterdata,x="date",y="new_cases",title="New Cases With Date")

    return fig_PolyReg_ret,fig_Holt_ret,fig_LagPred_ret #fig,fig2

if __name__ == '__main__':
    app.run_server(debug=True)





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

