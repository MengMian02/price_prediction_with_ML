import yfinance as yf
import pandas as pd
ticker = '3988.HK'
data = yf.download(ticker, start='2017-10-18', end='2024-11-12')
close_price = data[['Close']]
close_price.index = close_price.index.tz_localize(None)
close_price.to_excel('boc_close_price.xlsx')
