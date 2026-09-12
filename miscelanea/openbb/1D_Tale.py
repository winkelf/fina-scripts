from openbb_terminal.sdk import openbb 
import matplotlib.pyplot as plt 

stock_1D = openbb.stocks.load(symbol = 'AAPL') 
stock_1D['Close'].plot(title="1D Data - AAPL Stock Price", grid=True) 
plt.show() 
