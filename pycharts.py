import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import scipy.optimize as sco
import plotly.graph_objects as go
import datetime
from data_storage import get_list
from scipy.stats import norm
import akshare as ak
########################################################initial settings###################################
st.set_page_config(
    page_title="Efficient Frontier APP",
    page_icon="🐖",
    layout="wide",
    initial_sidebar_state="expanded",
)
#######################################################define functions###################################


def get_date():
    st.sidebar.subheader("Selecte the time interval")
    today = datetime.date.today()
    start_date = st.sidebar.date_input("Start date",datetime.date(2017,12,29))
    end_date = st.sidebar.date_input("End date",today)
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
    else:
        st.sidebar.error('Error: End date must fall after start date.')
    return start_date, end_date

def get_portf_num():
    portf_num = st.slider("Selecting the number of simulating portfolio",min_value= 1000, max_value= 10000, value =2000, step=1000)
    return portf_num

@st.cache(allow_output_mutation=True)#
def load_quotes(asset, start, end):
    return yf.download(asset,start, end, adjusted=True)

@st.cache(allow_output_mutation=True)#
def load_risk_free_rate(start, end):
    bond_zh_us_rate_df = ak.bond_zh_us_rate()
    return bond_zh_us_rate_df.query("日期 >= @start  and 日期 < @end")

#2. Define functions for calculating portfolio returns and volatility:
def get_portf_rtn(w, avg_rtns):
    return np.sum(avg_rtns * w)
def get_portf_vol(w, avg_rtns, cov_mat):
    return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

#3. Define the function calculating the Efficient Frontier:

def get_efficient_frontier(avg_rtns, cov_mat, rtns_range):
    efficient_portfolios = []
    n_assets = len(avg_returns)
    args = (avg_returns, cov_mat)
    bounds = tuple((0,1) for asset in range(n_assets))
    initial_guess = n_assets * [1. / n_assets, ]
    for ret in rtns_range:
        constraints = ({'type': 'eq',
                        'fun': lambda x: get_portf_rtn(x, avg_rtns)
                        - ret},
                        {'type': 'eq',
                        'fun': lambda x: np.sum(x) - 1})
        efficient_portfolio = sco.minimize(get_portf_vol,
                                            initial_guess,
                                            args=args,
                                            method='SLSQP',
                                            constraints=constraints,
                                            bounds=bounds)
        efficient_portfolios.append(efficient_portfolio)
    return efficient_portfolios

def plotstocks(df):
    """Plot the stocks in the dataframe df"""
    figure = go.Figure()
    alpha=0.3
    lw=1
    for stock in df.columns.values:
        if stock == 'Max Sharpe':
            alpha=1
            lw = 3
        elif stock == 'Min Variance':
            alpha=1
            lw = 3
        elif stock == 'Equal Weight':
            alpha=1
            lw = 3
        else:
            alpha=0.3
            lw=1
        figure.add_trace(go.Scatter(
            x=df.index,
            y=df[stock],
            name=stock,
            mode='lines',
            opacity=alpha,
            line={'width': lw}
        ))
    figure.update_layout(height=600,width=1500,
                         xaxis_title='Date'
                        )
    figure.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
                )
    
    return figure

def IGARCH(y , beta):
        conditional_var = [np.var(y)]
        for i in range(1, len(y)):
            var = beta *  conditional_var[i-1]+ (1-beta) * y[i-1]** 2 
            conditional_var.append(var)
        return conditional_var
########################################################################################################################


author = "Miraka"



N_DAYS = 252




st.title("Efficient Frontier APP")
st.write("Edit by: ",author)
st.subheader("Stock list control panel")
with st.beta_expander(label="Manage stock list", expanded=False):
    START_DATE,END_DATE = get_date()
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

st.sidebar.subheader("Setting a proper risk-free proxy")
bond = st.sidebar.checkbox("Use US Government bond yield")
if bond:
    rf_df = load_risk_free_rate(start=START_DATE,end=END_DATE)
    rf_selection = st.sidebar.selectbox("Select a risk-free proxy",["US Government Bond 2Y", "US Government Bond 5Y", "US Government Bond 10Y","US Government Bond 30Y"])
    if rf_selection =="US Government Bond 2Y":
        RF = np.mean(rf_df.loc[:,"美国国债收益率2年"])
    elif rf_selection =="US Government Bond 5Y":
        RF = np.mean(rf_df.loc[:,"美国国债收益率5年"])
    elif rf_selection =="US Government Bond 10Y":
        RF = np.mean(rf_df.loc[:,"美国国债收益率10年"])
    elif rf_selection =="US Government Bond 30Y":
        RF = np.mean(rf_df.loc[:,"美国国债收益率30年"])
    
else:
    RF = st.sidebar.number_input(
                                label="Input Annualized risk-free rate",
                                min_value=0.00,
                                value=0.015,
                                step=0.0001,
                                format="%.6f",
                                )
    RF = RF *100
    
st.sidebar.info("Current risk free rate: {:.4f}%".format(RF))
    
############################################################SELECTED DATA###################################################
returns_df = select_data.pct_change().dropna(axis='columns',how = "all") * 100
growth = (select_data/select_data.iloc[0,:])
num_missing = (returns_df == 0).sum()
n_assets = len(returns_df.columns)

if returns_df.empty:
    st.info("Use control panel to select stocks to start. 😊")
elif length <2:
    st.error("Input one more ticker please, the portfolio should contain at least two components")
else:
    st.subheader("Ticker's plots")
    row2_1, row2_2,row2_3 = st.beta_columns(3)
    
    initial_plots1 = row2_1.checkbox("Stock Prices Plot",1)
    initial_plots2 = row2_2.checkbox("Stock Returns Plot")
    initial_plots3 = row2_3.checkbox("Accumulated Returns Plot")
    if initial_plots1:
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
        fig.update_xaxes(
                        rangeslider_visible=True,
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        )
                        )
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))        
        st.plotly_chart(fig)
    if initial_plots2:
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
        fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
                )
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
                
        st.plotly_chart(fig)
    if initial_plots3:
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
        fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
                )
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
                
        st.plotly_chart(fig)
    with st.beta_expander(label="View the returns' summary", expanded=False):
        summary = pd.DataFrame(returns_df.agg({'count','min','median','mean','max','var','std','skew','kurt'})).T
        summary["Missing num"] = num_missing
        summary = summary[['count',"Missing num",'min','median','mean','max','var','std','skew','kurt']]
        summary["Annualized average"] = N_DAYS * summary["mean"] 
        summary["Annualized volatility"] = np.sqrt(N_DAYS) * summary["std"] 
        st.write(summary)
    
#################################################################################
    ef = st.button("Generate optimum portfolio report")
    
    avg_returns = returns_df.mean() * N_DAYS
    cov_mat = returns_df.cov() * N_DAYS
    
    def Efficient_Frontier_Generating():
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        
        
        N_PORTFOLIOS =  1000
        #Calculate annualized average returns and the corresponding standard deviation 

    
        #Simulate random portfolio weights:
        np.random.seed(42)
        weights = np.random.random(size=(N_PORTFOLIOS, n_assets))
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
        weights[:n_assets] = np.identity(n_assets)
    
        progress_bar.progress(10)
        status_text.text("Generating random portfolio")
                         
        #Calculate the portfolio metrics:
        portf_rtns = np.dot(weights, avg_returns)
        portf_vol = []
        for i in range(0, len(weights)):
            portf_vol.append(np.sqrt(np.dot(weights[i].T, np.dot(cov_mat, weights[i]))))
        portf_vol = np.array(portf_vol)
        portf_sharpe_ratio = (portf_rtns - RF) / portf_vol
        
        progress_bar.progress(20)
        status_text.text("Calculating the portfolio metrics")
    
        #Create a DataFrame containing all the data:
        portf_results_df = pd.DataFrame({'returns': portf_rtns,
                                        'volatility': portf_vol,
                                        'sharpe_ratio': portf_sharpe_ratio})
    ###################################################################
        min_ret_ind = np.argmin(portf_results_df.returns)
        min_ret_portf = portf_results_df.iloc[min_ret_ind,:]
        min_ret = round(min_ret_portf[0],6)
        
        max_ret_ind = np.argmax(portf_results_df.returns)
        max_ret_portf = portf_results_df.iloc[max_ret_ind,:]
        max_ret = round(max_ret_portf[0],6)
        
        
        max_sharp_ind = np.argmax(portf_results_df.sharpe_ratio)
        
        
        min_vol_ind = np.argmin(portf_results_df.volatility)
        min_vol_portf_rtn = portf_results_df.iloc[min_vol_ind,:]
        
        max_vol_ind = np.argmax(portf_results_df.volatility)
        max_vol_portf_rtn = portf_results_df.iloc[max_vol_ind,:]
        range_for_cal = round(max_vol_portf_rtn[1]+1,1)
        range_for_cal = range_for_cal *10000
        progress_bar.progress(40)
        status_text.text("Defining boundaries")
    #####################################################################   
        #####################################################################
    
    
    
        #4. Define the considered range of returns:
        rtns_range = np.linspace(min_ret, max_ret, 300)
    
        #5. Calculate the Efficient Frontier:
        efficient_portfolios = get_efficient_frontier(avg_returns,
                                                        cov_mat,
                                                        rtns_range)
    
        #6. Extract the volatilities of the efficient portfolios:
        vols_range = [x['fun'] for x in efficient_portfolios]
        progress_bar.progress(60)
        status_text.text("Calculating Efficient Frontier")

    
        ########################################################################
        ######          new range for calculate the efficient frontier    ######
        ########################################################################
    
    
    
    
    
    
        min_vol_ind = np.argmin(vols_range)
        min_vol_portf_rtn = rtns_range[min_vol_ind]
        min_vol_portf_vol = efficient_portfolios[min_vol_ind]['fun']
        min_vol_portf_sharp = (min_vol_portf_rtn -RF)/min_vol_portf_vol
    
        max_sharp_ind = np.argmax(((rtns_range - RF)/vols_range))
        max_sharp_portf_rtn = rtns_range[max_sharp_ind]
        max_sharp_portf_vol = efficient_portfolios[max_sharp_ind]['fun']
        max_sharp_portf_sharp = (max_sharp_portf_rtn - RF)/max_sharp_portf_vol
        
        slope = max_sharp_portf_sharp
        x = list(range(int(range_for_cal)))
        af = pd.DataFrame({'x': x})
        af['sigma'] = af['x']/10000
        af['sml'] = af['sigma'] * slope + RF
        progress_bar.progress(80)
        status_text.text("Calculating Security Market line")
        
        max_sharpe_ = returns_df @ efficient_portfolios[max_sharp_ind]['x'] 
        min_vol_ = returns_df @ efficient_portfolios[min_vol_ind]['x']
        equal_weight = returns_df.mean(axis = 1)
        returns_df['Max Sharpe'] = max_sharpe_
        returns_df['Min Variance'] = min_vol_
        returns_df['Equal Weight'] = equal_weight
        
        acc = (returns_df/100 + 1).cumprod()
        
        min_vol = pd.DataFrame([*zip(RISKY_ASSETS, np.round(efficient_portfolios[min_vol_ind]['x'],4))],columns=["Tickers","Weights"])
        min_vol = min_vol.set_index("Tickers")
        min_vol = min_vol[(min_vol.T != 0).any()]
        min_vol = min_vol.sort_values(by = "Weights", ascending=False)
        
        
        max_sharp = pd.DataFrame([*zip(RISKY_ASSETS, np.round(efficient_portfolios[max_sharp_ind]['x'],4))],columns=["Tickers","Weights"])
        max_sharp = max_sharp.set_index("Tickers")
        max_sharp = max_sharp[(max_sharp.T != 0).any()]
        max_sharp = max_sharp.sort_values(by = "Weights", ascending=False)
        
        risk_aversion = [1,2,3,4,5]
        Y = []
        for i in risk_aversion:
            yi = (max_sharp_portf_rtn/100 - RF/100)/(i * (max_sharp_portf_vol/100)**2)
            Y.append(yi)
        rf_weight = [1-j for j in Y]
        max_sharp_ = np.array(max_sharp)
        index_ = max_sharp.index
       
        LAMUDA = pd.DataFrame({ "A = 1":max_sharp_.flatten()*Y[0]
                               ,"A = 2":max_sharp_.flatten()*Y[1]
                               ,"A = 3":max_sharp_.flatten()*Y[2]
                               ,"A = 4":max_sharp_.flatten()*Y[3]
                               ,"A = 5":max_sharp_.flatten()*Y[4]}
                               , index =index_ )
        
        risk_aversion_df = pd.DataFrame({"Weights of Risky Asset Allocated (Y)":Y, "Weights of Risk-free Asset Allocated": rf_weight},index =LAMUDA.columns)
        risk_aversion_df["Expected return (%)"] = risk_aversion_df["Weights of Risky Asset Allocated (Y)"] * max_sharp_portf_rtn + risk_aversion_df["Weights of Risk-free Asset Allocated"] * RF
        risk_aversion_df["Volatility (%)"] = risk_aversion_df["Weights of Risky Asset Allocated (Y)"] * max_sharp_portf_vol
        risk_aversion_df["Sharpe Ratio"] = (risk_aversion_df["Expected return (%)"] - RF)/risk_aversion_df["Volatility (%)"]
        
        n_weight = len(LAMUDA)
    
    
    
        #Plot the Efficient Frontier:
        def plot_efficient_frontier():
            st.subheader("Efficient Frontier Plot")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vols_range, y=rtns_range
                        ,name="Efficient Frontier"
                        ,mode="lines"
                        ,opacity=1
                        ,marker=dict(size=2
                                    ,color = 1
                                    ,line=dict(width=1
                                            ,color = 1
                                                )
                                    )
                        ))
            fig.add_trace(go.Scatter(x=af['sigma'] , y=af['sml']
                ,name="Capital Allocation line"
                ,mode="lines"
                ,opacity=1
                ,line = dict(color='firebrick', width=4, dash='dot')
                ))
            fig.add_trace(go.Scatter(x=portf_results_df.volatility, y=portf_results_df.returns
                                    ,name="Simulating portfolio"
                                    ,mode="markers"
                                    ,opacity=0.8
                                    ,marker=dict(size=10
                                                ,color = portf_results_df.sharpe_ratio
                                                ,colorscale='Viridis'
                                                #,colorbar=dict(thickness=5, tickvals=[-5, 5], ticktext=['Low', 'High'], outlinewidth=0)
                                                ,line=dict(width=2
                                                            )
                                                )
                                    ))

            fig.add_trace(go.Scatter(x= [0], y=[RF]
                    ,name="Risk-free asset"
                    ,mode="markers"
                    ,opacity=0.8
                    ,marker_symbol=[204]
                    ,marker=dict(size= 10
                                ,colorscale='Viridis'
                                ,color = "green"
                                ,line=dict(width=1
                                            )
                                )
                    ))

        
            fig.add_trace(go.Scatter(x=np.sqrt(returns_df.var() * N_DAYS), y=avg_returns 
                                    ,name="Indivial Stock"
                                    ,mode="markers+text"
                                    ,textposition = "bottom center"
                                    ,marker_symbol=18
                                    ,text = list(avg_returns.index)
                                    ,opacity=1
                                    ,marker=dict(size=10
                                                ,color = "yellow"
                                                ,line=dict(width=2
                                                            )
                                                )
                                    ))
        
            fig.add_trace(go.Scatter(x=[min_vol_portf_vol,max_sharp_portf_vol], y=[min_vol_portf_rtn,max_sharp_portf_rtn]
                                    ,name="Optimized portfolios"
                                    ,mode="markers"
                                    ,marker_symbol=204
                                    ,text = ["Minimum variance portfolio","Maximum sharp ratio porfolio"]
                                    ,opacity=1
                                    ,marker=dict(size=10
                                                ,colorscale='Viridis'
                                                ,color = "red"
                                                ,line=dict(width=1
                                                            )
                                                )
                                    ))
            fig.add_trace(go.Scatter(x=risk_aversion_df.iloc[3:5,3], y=risk_aversion_df.iloc[3:5,2]
                        ,name="Risk-aversion portfolios"
                        ,mode="markers"
                        ,marker_symbol=22
                        ,text = ["A = 4","A = 5"]
                        ,opacity=1
                        ,marker=dict(size=10
                                    ,colorscale='Viridis'
                                    ,color = "blue"
                                    ,line=dict(width=1
                                                )
                                    )
                        ))
        
            fig.update_layout(height=700, width=1500)#, title_text="{} Portforlio Efficient Frontier".format(N_PORTFOLIOS))
            fig.update_xaxes(title_text="Annualize Volatility (%)")
            fig.update_yaxes(title_text="Annualize Expected return(%)")
            fig.update_layout(legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ))            
            st.plotly_chart(fig)
            
    
        
    
        
        plot_efficient_frontier()
        progress_bar.progress(100)
        status_text.text("Done")
        st.balloons()
        
        
        
        row3_1,row3_2 = st.beta_columns((5,5))
        

        
        row3_1.subheader("Maximum Sharpe ratio portfolio")
        row3_1.write("**Return:** {:.4f}%   \n  **Volatility:** {:.4f}%  \n   **Sharpe ratio:** {:.4f}".format(max_sharp_portf_rtn, max_sharp_portf_vol,max_sharp_portf_sharp))
        row3_1.write("**Weights**")
             
        fig_pie1 = go.Figure(data=[go.Pie(labels=max_sharp.index, values=max_sharp["Weights"], textinfo='label+percent',
                                     insidetextorientation='radial'
                                    )])
        fig_pie1.update_layout(height=400, width=500)
        fig_pie1.update_layout(showlegend=False)
        row3_1.plotly_chart(fig_pie1)
        

        row3_2.subheader("Minimum Volatility portfolio")
        row3_2.write("**Return:** {:.4f}%  \n   **Volatility:** {:.4f}%  \n   **Sharpe ratio:** {:.4f}".format(min_vol_portf_rtn, min_vol_portf_vol,min_vol_portf_sharp))

        row3_2.write("**Weights**")
        
        fig_pie2 = go.Figure(data=[go.Pie(labels=min_vol.index, values=min_vol["Weights"], textinfo='label+percent',
                             insidetextorientation='radial'
                            )])
        fig_pie2.update_layout(height=400, width=500)
        fig_pie2.update_layout(showlegend=False)
        row3_2.plotly_chart(fig_pie2)



        
        st.subheader("Accumulated return of Optimized Portfolios")
        st.plotly_chart(plotstocks(acc))
        
        st.subheader("Risk of Optimized Portfolios")
        
        

        
        fig_risk1 = go.Figure()
        max_port1 = max_sharpe_.iloc[1:]
        Conditional_Std_Deviation1 = np.sqrt(IGARCH(max_port1, 0.94))
        VaR_RM1 = - norm.ppf(0.95)  * Conditional_Std_Deviation1
        fig_risk1.add_trace(go.Scatter(x=max_port1.index, y=max_port1
                    ,name='Max Sharpe'
                    ))
        fig_risk1.add_trace(go.Scatter(x=max_port1.index, y=VaR_RM1 
            ,name='95% VaR (Risk Metric)'
            ))
        fig_risk1.update_layout(height=600, width=1500, title_text="Max Sharpe Ratio Portfolio")
        fig_risk1.update_xaxes(
                        rangeslider_visible=True,
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        )
                        )
        st.plotly_chart(fig_risk1)
        
        
        fig_risk2 = go.Figure()
        port2 = min_vol_.iloc[1:]
        Conditional_Std_Deviation2 = np.sqrt(IGARCH(port2, 0.94))
        VaR_RM2 = - norm.ppf(0.95)  * Conditional_Std_Deviation2
        fig_risk2.add_trace(go.Scatter(x=port2.index, y=port2
                    ,name='Min Variance'
                    ,line = dict(color='green', width=2)
                            
                    ))
        fig_risk2.add_trace(go.Scatter(x=port2.index, y=VaR_RM2 
            ,name='95% VaR (Risk Metric)'
            ))
        fig_risk2.update_layout(height=600, width=1500, title_text="Min Variance Portfolio")
        fig_risk2.update_xaxes(
                        rangeslider_visible=True,
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        )
                        )
        st.plotly_chart(fig_risk2)
        
        
        
        st.subheader("Asset allocation Between Optimized asset and Risk-free asset")
        st.latex(r'''Y = \frac{E(r_p)-r_f}{A * \sigma_p^2}''')
        st.markdown("**Where:**")
        st.markdown(" **A** is the risk aversion level of an investor.")
        st.markdown("**Y** is the total proportion of risky assets.")


        fig_bar = go.Figure()
        for k in range(n_weight):
            fig_bar.add_trace(go.Bar(name=LAMUDA.index[k], x=LAMUDA.columns, y=LAMUDA.iloc[k,:]
                        ))
       
        fig_bar.update_layout(barmode='stack')
        fig_bar.update_layout(legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        ))
        fig_bar.update_layout(height=400, width=800)
        st.subheader("Risk Aversion Assets")
        
        rown1,rown2,rown3 = st.beta_columns((2,5,2))
        rown2.table(risk_aversion_df.T)
        
        
        fig_bar2 = go.Figure()
        fig_bar2.add_trace(go.Bar(name="Risk free Asset", x=LAMUDA.columns, y=rf_weight)) 
        fig_bar2.update_layout(height=400, width=800)
        
        
        row3_1, row3_2 = st.beta_columns(2)
        row3_1.subheader("Weights of Risky Asset Allocated")
        row3_1.plotly_chart(fig_bar)
        row3_2.subheader("Weights of Risk-free Asset Allocated")
        row3_2.plotly_chart(fig_bar2)

        
    if ef:
        Efficient_Frontier_Generating()

 
