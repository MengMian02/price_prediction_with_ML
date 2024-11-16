import yfinance as yf
ticker = '000001.SS'
start_date = '2017-10-18'
end_date = '2024-11-12'

def get_yf_data_to_excel(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    close_price = data[['Close']]
    close_price.index = close_price.index.tz_localize(None)
    close_price = close_price.reset_index()
    close_price.columns = ['Date', ticker]
    close_price.set_index('Date', inplace=True)
    close_price.to_excel('sse.xlsx')

# get_yf_data_to_excel(ticker, start_date, end_date)

