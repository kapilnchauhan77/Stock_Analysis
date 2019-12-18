import os
import numpy as np
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from datetime import datetime as dt
import pandas_datareader.data as web
import math
import pickle as pkl
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor


# style.use('fivethirtyeight')
style.use('ggplot')

ticker = input("Please provide stock ticker: ")
ticker = ticker.upper()
mv_forward = True

if not os.path.exists(f"{ticker}.csv"):
    QUANDL_API_KEY = os.environ.get("QUANDL_API_KEY")

    start = dt(2010, 1, 1)
    end = dt.now()

    try:
        df = web.DataReader(ticker, "quandl", start, end, api_key=QUANDL_API_KEY)
        ticker = ticker.split("/")[-1] if "/" in ticker else ticker
        ticker = ticker.split("\\")[-1] if "\\" in ticker else ticker
        df.to_csv(f"{ticker}.csv")
        print("File Downloaded")
    except Exception as e:
        print(str(e))
        try:
            start = dt(2015, 1, 1)
            df = web.DataReader(ticker, "yahoo", start, end)
            df.to_csv(f"{ticker}.csv")
            print("File Downloaded")
        except Exception as e:
            print(str(e))
            mv_forward = False
            print("Wrong ticker provided!!!")

if mv_forward:
    df = pd.read_csv(f"{ticker}.csv", parse_dates=True, index_col="Date")[::-1]
    # print(df.columns)
    df['25ma'] = df['Close'].rolling(window=25, min_periods=0).mean()
    df['50ma'] = df['Close'].rolling(window=50, min_periods=0).mean()
    df['10ma'] = df['Close'].rolling(window=10, min_periods=0).mean()
    try:
        split_at = np.where(df['SplitRatio'] != 1)
    except:
        pass
    try:
        print(f"{ticker} stock was split on {str(df.index[split_at[0][0]])[:10]} at the ratio: {df['SplitRatio'][split_at[0][0]]}")
    except:
        print(f"{ticker} stock was never split in the given time frame: from {str(df.index[0])[:10]} to {str(df.index[-1])[:10]}")
    fig1 = plt.figure(figsize=(100, 100))
    fig1.suptitle(f"Open, High, Low, Close value of {ticker} Stock")
    plt.subplot(2, 2, 1)
    plt.title('Open and Close')
    df['Open'].plot(color='k')
    df['Close'].plot()
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.title('High')
    df['High'].plot(color='y')
    plt.subplot(2, 2, 3)
    plt.title('Low')
    df['Low'].plot(color='m')
    plt.subplot(2, 2, 4)
    plt.title('Moving averages of Closing price')
    df['25ma'].plot(color='r')
    df['50ma'].plot(color='g')
    df['10ma'].plot(color='b')
    plt.legend()
    plt.subplot_tool()
    plt.show()
    try:
        df_ohlc = df['AdjClose'].resample('10D').ohlc()
        df_volume = df['Volume'].resample('10D').sum()
        fig2 = plt.figure(figsize=(100, 100))
        fig2.suptitle(f"Candle stick Graph and Volume of {ticker} Stock sold")
        ax1 = plt.subplot2grid((6, 1), (0, 0), colspan=1, rowspan=5)
        ax2 = plt.subplot2grid((6, 1), (5, 0), colspan=1, rowspan=1, sharex=ax1)
        ax1.xaxis_date()
        df_ohlc.reset_index(inplace=True)
        df_ohlc['Date'] = df_ohlc["Date"].map(mdates.date2num)
        candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup="g")
        # plt.title("Candle Stick Plot", fontsize=10)
        ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
        # plt.title("Volume", fontsize=10)
        # plt.tight_layout()
        plt.subplot_tool()
        plt.show()
        df['HL_PCT'] = (df['AdjHigh'] - df['AdjClose']) / df['AdjClose'] * 100
        df['PCT_change'] = (df['AdjClose'] - df['AdjOpen']) / df['AdjOpen'] * 100

        forecast_col = 'AdjClose'
        df = df[["AdjClose", "HL_PCT", "PCT_change", "AdjVolume"]]
        df.fillna(-99999, inplace=True)
        forecast_out = math.ceil(0.01 * len(df))
        df["label"] = df[forecast_col].shift(-forecast_out)
        df.dropna(inplace=True)

        print(f"Dataframe size: {len(df)}")

        x = np.array(df.drop(['label'], axis=1))
        y = np.array(df['label'])

        x = preprocessing.scale(x)

        x_lately = x[-forecast_out:]
        x = x[:-forecast_out]

        y_lately = y[-forecast_out:]
        y = y[:-forecast_out]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        rfr = RandomForestRegressor()
        rfr.fit(x_train, y_train)
        logreg = LinearRegression()
        logreg.fit(x_train, y_train)
        print(f"Score of linear regression: {logreg.score(x_test, y_test)}")
        with open('LinearRegression.pkl', 'wb') as f:
            pkl.dump(logreg, f)
        print(f"Score of Random Forest: {rfr.score(x_test, y_test)}")
        with open('RandomForestRegressor.pkl', 'wb') as f:
            pkl.dump(rfr, f)
        forecast_set = rfr.predict(x_lately)

        df['Forecast'] = np.nan

        last_date = df.iloc[-1].name
        last_unix = last_date.timestamp()
        one_day = 86400
        next_unix = last_unix + one_day
        for i in forecast_set:
            next_date = dt.fromtimestamp(next_unix)
            next_unix += 86400
            df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
        fig3 = plt.figure(figsize=(100, 100))
        fig3.suptitle(f"Forecast of {ticker} Stock")
        df.loc[str(df.loc[df["AdjClose"].isnull()].index[0])][0] = df.loc[str(
            df.loc[df["AdjClose"].isnull()].index[0])][-1]
        df['AdjClose'].plot()
        df['Forecast'].plot()
        plt.legend(loc='best')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.show()
    except Exception as e:
        print(str(e))
        pass
