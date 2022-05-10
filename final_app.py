import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import warnings                     
warnings.filterwarnings("ignore")
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


header  = st.container()
dataset = st.container()
visual  = st.container()
model   = st.container()



with header:
    st.title('Forecasting...???')
    
    im = Image.open("image/image3.jpeg")
    st.image(im, width=700)



with dataset:
    st.header('DATA')
    df_bird = pd.read_csv('clean_data/df_bird.csv')
    df_lif  = pd.read_csv('clean_data/df_lieferando.csv')
    df_all  = pd.read_csv('clean_data/df_all.csv')
    df_holidays = pd.read_csv('clean_data/df_holidays.csv')
    
    
    df_list = ['bird', 'lif', 'all', 'holidays']
    

    
    #st.subheader("What do my datasets look like?")
    #x = st.selectbox('Choise a dataset', options={}) # Problem!!
    #st.write(x.head())
    
    #for i in df_list:
    #    if st.selectbox(f'choise', options=df_list):
    #        st.subheader('asd')
    #        st.write(i.head())
    
    #st.text_input("Give a DataFrame name", key="df")
    #st.write(st.session_state.df.head())
    #st.session_state.df
    
    #st.sidebar.write('List of DataFraeme')
    #df = st.sidebar.selectbox('Choise a DataFrame', df_list)
    #if st.checkbox(f'asdsdgh {df_list}'):
    #    st.write(df_list.head())


with visual:
    st.subheader("Let's see some visual about our data")
    
    # Plot for OrderBird
    fig_bird = px.line(df_bird, x='date', y='bird',
                       hover_data={"date": "|%B %d, %Y"},
                       title='Data OrderBird for 18 Months')
    fig_bird.update_xaxes(dtick="M1", tickformat="%b\n%Y", rangeslider_visible=True, rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m",  step="month", stepmode="backward"),
            dict(count=6, label="6m",  step="month", stepmode="backward"),
            dict(count=1, label="1y",  step="year",  stepmode="backward"),
            dict(step="all")
        ])))
    st.plotly_chart(fig_bird)
    
    
    # Plot for All Data
    fig_all  = px.line(df_all, x = 'date', y =df_all.columns,
                   hover_data={"date": "|%B %d, %Y"},
                   title='OrderBird and Lieferando for 12 Months')
    fig_all.update_xaxes(dtick="M1", tickformat="%b\n%Y", rangeslider_visible=True, rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m",  step="month", stepmode="backward"),
            dict(count=6, label="6m",  step="month", stepmode="backward"),
            dict(count=1, label="1y",  step="year",  stepmode="backward"),
            dict(step="all")
        ])))
    st.plotly_chart(fig_all)
    
    # check all_data with Normalisation:
    scaler = MinMaxScaler()
    
    df_all[['total','feelslike','humidity']] = scaler.fit_transform(df_all[['total','feelslike','humidity']])
    
    fig_all_nor  = px.line(df_all, x = 'date', y =df_all.columns,
                   hover_data={"date": "|%B %d, %Y"},
                   title='Totat - Temperature - Humidity')
    fig_all_nor.update_xaxes(dtick="M1", tickformat="%b\n%Y", rangeslider_visible=True, rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m",  step="month", stepmode="backward"),
            dict(count=6, label="6m",  step="month", stepmode="backward"),
            dict(count=1, label="1y",  step="year",  stepmode="backward"),
            dict(step="all")
        ])))
    st.plotly_chart(fig_all_nor)

with model:
    st.header('See some Models Result')
    im2 = Image.open("image/image1.jpeg")
    st.image(im2, width=700)
    
    col1, col2 = st.columns(2)
    
    # Train Test Split:
    size = int(len(df_bird)*0.8)
    df_bird_train, df_bird_test = df_bird.iloc[:size], df_bird.iloc[size:]
    
    start_date = "2022-01-09"
    end_date   = "2022-04-30"
    
    # ARIMA just with data bird
    auto_arima = pickle.load(open('model/auto_arima.sav', 'rb'))
    
    # ARIMA with other parametres- without normalisation:
    auto_arima_X = pickle.load(open('model/auto_arima_X.sav', 'rb'))
    
    # ARIMA with other parametres- with normalisation:
    auto_arima_XN = pickle.load(open('model/auto_arima_XN.sav', 'rb'))
    
    # RANDOM - FOREST
    model_random_forest = pickle.load(open('model/random_forest.sav', 'rb'))
    
    col1.text('ARIMA - MRSE result: ' )
    col1.text('>>>>>>>>>>>>>>>>>>>>' )
    col2.success(130.35)
    col1.text('ARIMA-features -MRSE result:' )
    col1.text('>>>>>>>>>>>>>>>>>>>>' )
    col2.warning(406.39)
    col1.text('ARIMA-Norm. featuers - MRSE result:' )
    col1.text('>>>>>>>>>>>>>>>>>>>>' )
    col2.info(261.05)
    col1.text('Random - Forest - MRSE result:' )
    col1.text('>>>>>>>>>>>>>>>>>>>>' )
    col2.warning(318.76)
    col1.text('Prophet - MRSE result:' )
    col1.text('>>>>>>>>>>>>>>>>>>>>' )
    col2.success(166.84)
    
    st.subheader('Which features are most important for my model?')
    im3 = Image.open("image/features.png")
    st.image(im3, width=700)
    
    # FACEBOOK PROPHET
    df_prophet = df_bird[['date','bird']]
   
    df_prophet = df_prophet.rename(columns={'date': 'ds', 'bird': 'y'})
    df_holidays = df_holidays.rename(columns={'date': 'ds', 'day_name': 'holiday'})
    
    model_prophet = Prophet(holidays=df_holidays, yearly_seasonality=True, daily_seasonality=True)
    model_prophet.fit(df_prophet)
    
    future = model_prophet.make_future_dataframe(periods=14,freq='D')
    df_forecast = model_prophet.predict(future)
    
    st.subheader('Forecast with Facebook Prophet')
    fig_prop1 = plot_plotly(model_prophet, df_forecast)
    st.plotly_chart(fig_prop1)
    
    st.subheader('Model Prophet Components')
    fig_prop2 = plot_components_plotly(model_prophet, df_forecast)
    st.plotly_chart(fig_prop2)