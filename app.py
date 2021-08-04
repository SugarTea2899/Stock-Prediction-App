import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from gru import predict_gru
from TestLSTM import predict_lstm
from xgboost_prediction import xgboost_prediction

df1 = "FB.csv"
df2 = "TSLA.csv"
df3 = "MSFT.csv"
app = dash.Dash()
server = app.server

gru_df_roc_fb, gru_valid_roc_fb = predict_gru(df1, True)
gru_df_close_fb, gru_valid_close_fb = predict_gru(df1, False)

gru_df_roc_tsla, gru_valid_roc_tsla = predict_gru(df2, True)
gru_df_close_tsla, gru_valid_close_tsla = predict_gru(df2, False)

gru_df_roc_msft, gru_valid_roc_msft = predict_gru(df3, True)
gru_df_close_msft, gru_valid_close_msft = predict_gru(df3, False)

lstm_df_roc_fb, lstm_valid_roc_fb = predict_lstm(df1, True)
lstm_df_close_fb, lstm_valid_close_fb = predict_lstm(df1, False)

lstm_df_roc_tsla, lstm_valid_roc_tsla = predict_lstm(df2, True)
lstm_df_close_tsla, lstm_valid_close_tsla = predict_lstm(df2, False)

lstm_df_roc_msft, lstm_valid_roc_msft = predict_lstm(df3, True)
lstm_df_close_msft, lstm_valid_close_msft = predict_lstm(df3, False)

xgboost_df_roc_fb, xgboost_valid_roc_fb = xgboost_prediction(pd.read_csv(df1), True)
xgboost_df_close_fb, xgboost_valid_close_fb = xgboost_prediction(pd.read_csv(df1), False)

xgboost_df_roc_tsla, xgboost_valid_roc_tsla = xgboost_prediction(pd.read_csv(df2), True)
xgboost_df_close_tsla, xgboost_valid_close_tsla = xgboost_prediction(pd.read_csv(df2), False)

xgboost_df_roc_msft, xgboost_valid_roc_msft = xgboost_prediction(pd.read_csv(df3), True)
xgboost_df_close_msft, xgboost_valid_close_msft = xgboost_prediction(pd.read_csv(df3), False)

app.layout = html.Div([

    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    dcc.Dropdown(id='my-dropdown',
        options=[
            {'label': 'Tesla', 'value': 'TSLA'},
            {'label': 'Facebook', 'value': 'FB'},
            {'label': 'Microsoft', 'value': 'MSFT'}],
                value='FB',
                style={"display": "block", "margin-left": "auto",
                "margin-right": "auto", "width": "60%"}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab( id = "GRU", label='GRU Stock Data',children=[
            html.Div([
                html.H2("GRU-CLOSE Predicted price FB", style={"textAlign": "center"}),
                dcc.Graph(
                    id="GRU-CLOSE",
                    figure={
                        "data": [
                            go.Scatter(
                                x=gru_df_close_fb['Date'],
                                y=gru_df_close_fb["Close"],
                                mode='lines',
                                name='Closing price'
                            ),
                            go.Scatter(
                                x=gru_valid_close_fb.index,
                                y=gru_valid_close_fb["Predictions"],
                                mode='lines',
                                name='GRU-CLOSE Predicted price'
                            )

                        ],
                        "layout": go.Layout(
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                ),

                html.H2("GRU-ROC Predicted price FB", style={"textAlign": "center"}),
                    dcc.Graph(
                        id="GRU-ROC",
                        figure={
                            "data": [
                                go.Scatter(
                                    x=gru_df_roc_fb['Date'],
                                    y=gru_df_roc_fb["ROC"],
                                    mode='lines',
                                    name='Closing price'
                                ),
                                go.Scatter(
                                    x=gru_valid_roc_fb.index,
                                    y=gru_valid_roc_fb["Predict_ROC"],
                                    mode='lines',
                                    name='GRU-ROC Predicted price'
                                )

                            ],
                            "layout": go.Layout(
                                xaxis={'title': 'Date'},
                                yaxis={'title': 'Closing Rate'}
                            )
                        }

                    ),

            ])

        ]),

        dcc.Tab(id ="LSTM",label='LSTM Stock Data',children=[
            html.Div([
                html.H2("LSTM-CLOSE Predicted price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="LSTM-CLOSE",
                    figure={
                        "data": [
                            go.Scatter(
                                x=lstm_df_close_fb['Date'],
                                y=lstm_df_close_fb["Close"],
                                mode='lines',
                                name='Closing price'
                            ),
                            go.Scatter(
                                x=lstm_valid_close_fb.index,
                                y=lstm_valid_close_fb["Predictions"],
                                mode='lines',
                                name='LSTM-CLOSE Predicted price'
                            )

                        ],
                        "layout": go.Layout(
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                ),

                html.H2("LSTM-ROC Predicted price", style={"textAlign": "center"}),
                    dcc.Graph(
                        id="LSTM-ROC",
                        figure={
                            "data": [
                                go.Scatter(
                                    x=lstm_df_roc_fb['Date'],
                                    y=lstm_df_roc_fb["ROC"],
                                    mode='lines',
                                    name='Closing price'
                                ),
                                go.Scatter(
                                    x=lstm_valid_roc_fb.index,
                                    y=lstm_valid_roc_fb["Predict_ROC"],
                                    mode='lines',
                                    name='LSTM-ROC Predicted price'
                                )

                            ],
                            "layout": go.Layout(
                                xaxis={'title': 'Date'},
                                yaxis={'title': 'Closing Rate'}
                            )
                        }

                    ),

            ])

        ]),

        dcc.Tab(id = 'XGBOOST',label='XGBOOST Stock Data', children=[
            html.Div([
                html.H2("XGBOOST-CLOSE Predicted price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="XGBOOST-CLOSE",
                    figure={
                        "data": [
                            go.Scatter(
                                x=xgboost_df_close_fb['Date'],
                                y=xgboost_df_close_fb["Close"],
                                mode='lines',
                                name='Closing price'
                            ),
                            go.Scatter(
                                x=xgboost_valid_close_fb.index,
                                y=xgboost_valid_close_fb["Prediction"],
                                mode='lines',
                                name='XGBOOST-CLOSE Predicted price'
                            )

                        ],
                        "layout": go.Layout(
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                ),

                html.H2("XGBOOST-ROC Predicted price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="XGBOOST-ROC",
                    figure={
                        "data": [
                            go.Scatter(
                                x=xgboost_df_roc_fb['Date'],
                                y=xgboost_df_roc_fb["ROC"],
                                mode='lines',
                                name='Closing price'
                            ),
                            go.Scatter(
                                x=xgboost_valid_roc_fb.index,
                                y=xgboost_valid_roc_fb["Prediction"],
                                mode='lines',
                                name='XGBOOST-ROC Predicted price'
                            )

                        ],
                        "layout": go.Layout(
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                ),

            ])

        ]),

    ])
])

@app.callback(Output('GRU', 'children'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):

    if (selected_dropdown_value == "FB"):
        gru_df_roc,gru_valid_roc,gru_df_close,gru_valid_close = gru_df_roc_fb,gru_valid_roc_fb,gru_df_close_fb,gru_valid_close_fb
    if (selected_dropdown_value == "TSLA"):
        gru_df_roc, gru_valid_roc, gru_df_close, gru_valid_close = gru_df_roc_tsla, gru_valid_roc_tsla, gru_df_close_tsla, gru_valid_close_tsla
    if (selected_dropdown_value == "MSFT"):
        gru_df_roc, gru_valid_roc, gru_df_close, gru_valid_close = gru_df_roc_msft, gru_valid_roc_msft, gru_df_close_msft, gru_valid_close_msft

    children =  html.Div([
                html.H2("GRU-CLOSE Predicted price " + selected_dropdown_value, style={"textAlign": "center"}),
                dcc.Graph(
                    id="GRU-CLOSE",
                    figure={
                        "data": [
                            go.Scatter(
                                x=gru_df_close['Date'],
                                y=gru_df_close["Close"],
                                mode='lines',
                                name='Closing price'
                            ),
                            go.Scatter(
                                x=gru_valid_close.index,
                                y=gru_valid_close["Predictions"],
                                mode='lines',
                                name='GRU-CLOSE Predicted price'
                            )

                        ],
                        "layout": go.Layout(
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                ),

                html.H2("GRU-ROC Predicted price "  + selected_dropdown_value, style={"textAlign": "center"}),
                    dcc.Graph(
                        id="GRU-ROC",
                        figure={
                            "data": [
                                go.Scatter(
                                    x=gru_df_roc['Date'],
                                    y=gru_df_roc["ROC"],
                                    mode='lines',
                                    name='Closing price'
                                ),
                                go.Scatter(
                                    x=gru_valid_roc.index,
                                    y=gru_valid_roc["Predict_ROC"],
                                    mode='lines',
                                    name='GRU-ROC Predicted price'
                                )

                            ],
                            "layout": go.Layout(
                                xaxis={'title': 'Date'},
                                yaxis={'title': 'Closing Rate'}
                            )
                        }

                    ),
                ])
    return children

@app.callback(Output('LSTM', 'children'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):

    if (selected_dropdown_value == "FB"):
        lstm_df_roc,lstm_valid_roc,lstm_df_close,lstm_valid_close = lstm_df_roc_fb,lstm_valid_roc_fb,lstm_df_close_fb,lstm_valid_close_fb
    if (selected_dropdown_value == "TSLA"):
        lstm_df_roc, lstm_valid_roc, lstm_df_close, lstm_valid_close = lstm_df_roc_tsla, lstm_valid_roc_tsla, lstm_df_close_tsla, lstm_valid_close_tsla
    if (selected_dropdown_value == "MSFT"):
        lstm_df_roc, lstm_valid_roc, lstm_df_close, lstm_valid_close = lstm_df_roc_msft, lstm_valid_roc_msft, lstm_df_close_msft, lstm_valid_close_msft

    children =  html.Div([
                html.H2("lstm-CLOSE Predicted price " + selected_dropdown_value, style={"textAlign": "center"}),
                dcc.Graph(
                    id="lstm-CLOSE",
                    figure={
                        "data": [
                            go.Scatter(
                                x=lstm_df_close['Date'],
                                y=lstm_df_close["Close"],
                                mode='lines',
                                name='Closing price'
                            ),
                            go.Scatter(
                                x=lstm_valid_close.index,
                                y=lstm_valid_close["Predictions"],
                                mode='lines',
                                name='lstm-CLOSE Predicted price'
                            )

                        ],
                        "layout": go.Layout(
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                ),

                html.H2("lstm-ROC Predicted price "  + selected_dropdown_value, style={"textAlign": "center"}),
                    dcc.Graph(
                        id="lstm-ROC",
                        figure={
                            "data": [
                                go.Scatter(
                                    x=lstm_df_roc['Date'],
                                    y=lstm_df_roc["ROC"],
                                    mode='lines',
                                    name='Closing price'
                                ),
                                go.Scatter(
                                    x=lstm_valid_roc.index,
                                    y=lstm_valid_roc["Predict_ROC"],
                                    mode='lines',
                                    name='lstm-ROC Predicted price'
                                )

                            ],
                            "layout": go.Layout(
                                xaxis={'title': 'Date'},
                                yaxis={'title': 'Closing Rate'}
                            )
                        }

                    ),
                ])
    return children

@app.callback(Output('XGBOOST', 'children'),
                  [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):

    if (selected_dropdown_value == "FB"):
        xgboost_df_roc, xgboost_valid_roc, xgboost_df_close, xgboost_valid_close = xgboost_df_roc_fb, xgboost_valid_roc_fb, xgboost_df_close_fb, xgboost_valid_close_fb
    if (selected_dropdown_value == "TSLA"):
        xgboost_df_roc, xgboost_valid_roc, xgboost_df_close, xgboost_valid_close = xgboost_df_roc_tsla, xgboost_valid_roc_tsla, xgboost_df_close_tsla, xgboost_valid_close_tsla
    if (selected_dropdown_value == "MSFT"):
        xgboost_df_roc, xgboost_valid_roc, xgboost_df_close, xgboost_valid_close = xgboost_df_roc_msft, xgboost_valid_roc_msft, xgboost_df_close_msft, xgboost_valid_close_msft

    children = html.Div([
        html.H2("xgboost-CLOSE Predicted price " + selected_dropdown_value, style={"textAlign": "center"}),
            dcc.Graph(
                id="xgboost-CLOSE",
                figure={
                    "data": [
                        go.Scatter(
                            x=xgboost_df_close['Date'],
                            y=xgboost_df_close["Close"],
                            mode='lines',
                            name='Closing price'
                        ),
                        go.Scatter(
                            x=xgboost_valid_close.index,
                            y=xgboost_valid_close["Prediction"],
                            mode='lines',
                            name='xgboost-CLOSE Predicted price'
                        )

                    ],
                    "layout": go.Layout(
                        xaxis={'title': 'Date'},
                        yaxis={'title': 'Closing Rate'}
                    )
                }

            ),

            html.H2("xgboost-ROC Predicted price " + selected_dropdown_value, style={"textAlign": "center"}),
            dcc.Graph(
                id="xgboost-ROC",
                figure={
                    "data": [
                        go.Scatter(
                            x=xgboost_df_roc['Date'],
                            y=xgboost_df_roc["ROC"],
                            mode='lines',
                            name='Closing price'
                        ),
                        go.Scatter(
                            x=xgboost_valid_roc.index,
                            y=xgboost_valid_roc["Prediction"],
                            mode='lines',
                            name='xgboost-ROC Predicted price'
                        )

                    ],
                    "layout": go.Layout(
                        xaxis={'title': 'Date'},
                        yaxis={'title': 'Closing Rate'}
                    )
                }

            ),
        ])
    return children

if __name__ == '__main__':
    app.run_server(debug=False)