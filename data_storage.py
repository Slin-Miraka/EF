import streamlit as st
import numpy as np


list_ = []
HECK10 = ['COST','BP','AMZN','TM','XOM','AAPL','CVS','BRK','UNH','T']
top10weights = ['GOOGL','FB','AMZN','MSFT','AAPL','GOOG','BRK','JPM','TSLA','JNJ'] 
FEMALE10 = ["ORCL","GSK","ACN","BBY","GM","ANTM","UPS","GD","PGR","NOC"]

def get_list():
    row1_1, row1_2, row1_3 = st.beta_columns((3,2,2))
    row1_3.write("Customized Portfolios")
    select = row1_3.radio("",["Customized portfolio 1", "Customized portfolio 2","Customized portfolio 3"])
    row1_3.write("Sample Portfolios")
    select1 = row1_3.radio("",["Top 10 S&P 500 Stocks By Index Weight", "10 Biggest Companies with Female CEOs","10 Biggest Companies That Have Been Hacked"])
    if select == "Customized portfolio 1":
        list_A = HECK10
    elif select == "Customized portfolio 2":
        list_A = top10weights
    elif select == "Customized portfolio 3":
        list_A = FEMALE10
        
    symbol = row1_1.text_input("Input Tickers")
    row1_1.write("You can add **YAHOO** Tickers to the portfolio. Eg. Input **MCD** to the portfolio.")
    if row1_1.button("Add Tickers"):
        list_A.append(symbol.upper())
    drop = row1_1.selectbox("Drop a Ticker from the stock list.",np.sort(list_A))
    if row1_1.button("Drop Tickers"):   
        list_A.remove(drop)

    

    if select1 == "Top 10 S&P 500 Stocks By Index Weight":
        list_ = ['GOOGL','FB','AMZN','MSFT','AAPL','GOOG','BRK','JPM','TSLA','JNJ'] 
    elif select1 == "10 Biggest Companies with Female CEOs":
        list_ = ["ORCL","GSK","ACN","BBY","GM","ANTM","UPS","GD","PGR","NOC"]
    elif select1 == "10 Biggest Companies That Have Been Hacked":
        list_ = ['COST','BP','AMZN','TM','XOM','AAPL','CVS','BRK','UNH','T']
        
    row1_2.write("Current components of portfolio")
    list_ = [x for x in list_ if x != '']
    list_ = list(dict.fromkeys(list_))
    list_.sort()
    row1_2.write(list_)
    

    return list_