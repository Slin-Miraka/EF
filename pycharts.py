import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import scipy.optimize as sco
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
import datetime
from data_storage import get_list

########################################################initial settings###################################
st.set_page_config(
    page_title="Efficient Frontier APP",
    page_icon="🐖",
    layout="wide",
    initial_sidebar_state="expanded",
)
#######################################################define functions###################################


def get_date():
    today = datetime.date.today()
    start_date = st.sidebar.date_input("Selecting the Start date",datetime.date(2017,12,29))
    end_date = st.sidebar.date_input("Selecting the End date",today)
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    else:
        st.sidebar.error('Error: End date must fall after start date.')
    return start_date, end_date

def get_portf_num():
    portf_num = st.sidebar.slider("Selecting the number of simulating portfolio",min_value= 1000, max_value= 10000, value =2000, step=1000)
    return portf_num

@st.cache(allow_output_mutation=True)#
def load_quotes(asset, start, end):
    return yf.download(asset,start, end, adjusted=True)

########################################################################################################################


author = "Miraka"


RF = st.sidebar.number_input(
                            label="Input Annualized risk-free rate",
                            min_value=0.00,
                            value=0.015,
                            step=0.0001,
                            format="%.6f",
                            )
N_DAYS = 252

START_DATE,END_DATE = get_date()


st.title("Efficient Frontier APP")
st.write("Edit by: ",author)
st.subheader("Stock list control panel")
with st.beta_expander(label="Manage stock list", expanded=False):
    RISKY_ASSETS = get_list()
    
    data = load_quotes(RISKY_ASSETS, start=START_DATE,end=END_DATE)['Adj Close']
    length = len(data) 
    all = st.checkbox("Select all to generate Efficient Frontier")
    if all:
        selected_options = st.multiselect("Select one or more options:",
             RISKY_ASSETS,RISKY_ASSETS)
    else:
        selected_options =  st.multiselect("Select one or more options:",
            RISKY_ASSETS)
    
    select_data = data.loc[:,selected_options]
    st.write("Check the selected data (Retrived data length: {})".format(length))
    st.write(select_data)
    n_assets = len(selected_options)
############################################################SELECTED DATA###################################################
returns_df = select_data.pct_change().dropna() * 100
growth = (select_data/select_data.iloc[0,:]-1)
num_missing = (returns_df == 0).sum()


if returns_df.empty:
    st.info("Use control panel to select stocks to start. 😊")
else:
    st.subheader("Ticker's plots")
    row2_1, row2_2 = st.beta_columns((1,9))
    
    initial_plots = row2_1.radio("  ", ["Price plot", "Return plot","Acc Return"])
    if initial_plots == "Price plot":
        fig = go.Figure()
        if isinstance(returns_df, pd.Series) == True:
            fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df
                        ,name=returns_df.name
                        
                        ))
        else: 
            for idx, col_name in enumerate(returns_df):
                fig.add_trace(go.Scatter(x=returns_df.index, y=select_data.iloc[:,idx]
                            ,name=returns_df.columns[idx]
                            
                            ))
        fig.update_layout(height=600, width=1300, title_text="Stock price")
        fig.update_xaxes(title_text="Date")
        #fig.update_yaxes(title_text="Stock Price")
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))        
        row2_2.plotly_chart(fig)
    if initial_plots == "Return plot":
        fig = go.Figure()
        if isinstance(returns_df, pd.Series) == True:
            fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df
                        ,name=returns_df.name
                        
                        ))
        else: 
            for idx, col_name in enumerate(returns_df):
                fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df.iloc[:,idx]
                            ,name=returns_df.columns[idx]
                            
                            ))
        fig.update_layout(height=600, width=1300, title_text="Stock Returns")
        fig.update_xaxes(title_text="Date")
        #fig.update_yaxes(title_text="Stock Return")
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
                
        row2_2.plotly_chart(fig)
    if initial_plots == "Acc Return":
        fig = go.Figure()
        if isinstance(returns_df, pd.Series) == True:
            fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df
                        ,name=returns_df.name
                        
                        ))
        else: 
            for idx, col_name in enumerate(returns_df):
                fig.add_trace(go.Scatter(x=returns_df.index, y=growth.iloc[:,idx]
                            ,name=returns_df.columns[idx]
                            
                            ))
        fig.update_layout(height=600, width=1300, title_text="Accumulated Stock Returns")
        fig.update_xaxes(title_text="Date")
        #fig.update_yaxes(title_text="Stock Return")
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
                
        row2_2.plotly_chart(fig)
    with st.beta_expander(label="View the returns' summary", expanded=False):
        summary = pd.DataFrame(returns_df.agg({'count','min','median','mean','max','var','std','skew','kurt'})).T
        summary["Missing num"] = num_missing
        summary = summary[['count',"Missing num",'min','median','mean','max','var','std','skew','kurt']]
        summary["Annualized average"] = 252 * summary["mean"] 
        summary["Annualized volatility"] = np.sqrt(252) * summary["std"] 
        st.write(summary)
    st.subheader("Efficient Frontier")
