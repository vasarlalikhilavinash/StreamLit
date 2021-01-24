import streamlit as st
from datetime import time
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs  as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import plotly.tools
import plotly.express as px
import numpy as np
import pandas as pd
#import geopandas
import streamlit as st
import pyodbc

sns.set_style("darkgrid")


st.title("Slovakia Active Player Day Prediction and KPI's Influence in Day Level")

st.markdown("""
	The data set contains information about Slovakia(SK) Active Player per day and there infulencing factors.


	Here are a few important questions that you might seek to address:
	- Is there a relationship between APD's  and remaining variables?
	- How strong is the relationship between the APD'S and remaining variables?
	- Which Bonus and others KPI's contribute to APD's in daily run?
	- How accurately can we estimate the effect of each medium on APD's?
	- How accurately can we predict future APD's?
	- Is the relationship linear?

	We want to find a function that given input KPI's and the output APD's
	and visualize the relationship between the features and the response using scatter plots.

	The objective is to use Multivariate linear regression to understand how KPI's impacts APD's.
	
	### Data Description
	APD_AC - Active players per day
	New_Actives_RS_3 - New Actives running sum 
	Event_Count - No. of events per day
	Count_Bonus_Offered_SB - Bonus offered count per day
    Win_Count - winning players count end of previous day 
    Active_Store_Count - Actives stores count per day 


""")




st.sidebar.title("Operations on the Dataset")

#st.subheader("Checkbox")
w1 = st.sidebar.checkbox("show table", False)

plotscatter= st.sidebar.checkbox("show scatter plots", False)
plothist= st.sidebar.checkbox("show hist plots", False)
#trainmodel= st.sidebar.checkbox("Train model", False)
#dokfold= st.sidebar.checkbox("DO KFold", False)
#distView=st.sidebar.checkbox("Dist View", False)
linechart=st.sidebar.checkbox("APD's Actuals Vs Predictions",False)
_3dplot=st.sidebar.checkbox("Independent Variable Infulence On Regression Prediction", False)

#st.write(w1)
conn = pyodbc.connect(Driver="{SQL Server Native Client 11.0}",
                      Server="server name",
                      Database="DB name",
                      UID='User name',
                      PWD='Password'
                     )

cur = conn.cursor()

coef = pd.read_sql_query("""
select * from devbox.APD_coff_SK where date ='2021-01-15'
""", conn)
coef.set_index("date", inplace = True)
coef=coef.astype('float')

new_coef = pd.read_sql_query("""
select * from devbox.APD_coff_SK where date ='2021-01-18'
""", conn)
new_coef.set_index("date", inplace = True)
new_coef=new_coef.astype('float')

feb_coef = pd.read_sql_query("""
select * from devbox.APD_coff_SK where date ='2020-12-13'
""", conn)
feb_coef.set_index("date", inplace = True)
feb_coef=feb_coef.astype('float')


df = pd.read_sql_query("""
Select * from devbox.SK_MV_Input order by date
""", conn)

df.set_index("date", inplace = True)
df=df.astype('int64')
df=df.iloc[:,:]
x_apd,y_apd =  df.iloc[:,1:], df.iloc[:,:1]
APD_Actuals=y_apd
Input=x_apd

#New Coeff
df_new=df.iloc[19:,:]
x_apd_new,y_apd_new =  df_new.iloc[:,1:], df_new.iloc[:,:1]
APD_Actuals_new=y_apd_new
Input_new=x_apd_new

#feb Coeff
df_feb=df.iloc[:,:]
x_apd_feb,y_apd_feb =  df_feb.iloc[:,1:], df_feb.iloc[:,:1]
APD_Actuals_feb=y_apd_feb
Input_feb=x_apd_feb


#bsts = pd.read_sql_query("""
#select * from devbox.bsts_sk_dec_2020 order by Date
#""", conn)

#bsts.set_index("Date", inplace = True)
#bsts_oct=bsts.iloc[:,:1]
#bsts_oct=bsts_oct.astype('int64')
#bsts_nov=bsts.iloc[29:,1:]
#bsts_nov=bsts_nov.astype('int64')




conn.close()

#prediction
df_day=Input
val=coef.iloc[:,1:]
val_1=1/val
prec=df_day.values/val_1.values
df_prec = pd.DataFrame(prec, columns=["Event_Count","New_Actives_RS_3","Count_Bonus_Offered_SB","Win_Count","Active_Store_Count"],index=df_day.index)
df_prec['Intercept']=coef['intercept'].values[0]
df_prec['APD_Regression_Predict']=df_prec.sum(axis='columns')
df_prec['APD_Actuals']=APD_Actuals.values
rmse_oct = round(np.mean((df_prec['APD_Regression_Predict'] - df_prec['APD_Actuals'])**2)**.5,2)


#new coef pred
df_day_new=Input_new
new_val=new_coef.iloc[:,1:]
new_val_1=1/new_val
new_prec=df_day_new.values/new_val_1.values
df_new_prec = pd.DataFrame(new_prec, columns=["Event_Count","New_Actives_RS_3","Count_Bonus_Offered_SB","Win_Count","Active_Store_Count"],index=df_day_new.index)
df_new_prec['Intercept']=new_coef['intercept'].values[0]
df_new_prec['APD_Regression_Predict_base_nov']=df_new_prec.sum(axis='columns')
df_new_prec['APD_Actuals']=APD_Actuals_new.values
rmse_nov = round(np.mean((df_new_prec['APD_Regression_Predict_base_nov'] - df_new_prec['APD_Actuals'])**2)**.5,2)

#feb coef pred
df_day_feb=Input_feb
feb_val=feb_coef.iloc[:,1:]
feb_val_1=1/feb_val
feb_prec=df_day_feb.values/feb_val_1.values
df_feb_prec = pd.DataFrame(feb_prec, columns=["Event_Count","feb_Actives_RS_3","Count_Bonus_Offered_SB","Win_Count","Active_Store_Count"],index=df_day_feb.index)
df_feb_prec['Intercept']=feb_coef['intercept'].values[0]
df_feb_prec['APD_Regression_Predict_base_feb']=df_feb_prec.sum(axis='columns')
df_feb_prec['APD_Actuals']=APD_Actuals_feb.values
rmse_feb = round(np.mean((df_feb_prec['APD_Regression_Predict_base_feb'] - df_feb_prec['APD_Actuals'])**2)**.5,2)







@st.cache
def read_data():
    #return pd.read_csv("C:/Users/LIKHI/OneDrive/Desktop/sk_main.csv")[['Summary_Date','APD_AC','New_Actives_RS_3','Event_Count','Count_Bonus_Offered_SB','Win_Count','Active_Store_Count']]
	return df

df=read_data()





#st.write(df)
if w1:
    st.dataframe(df,width=2000,height=500)

#species = st.multiselect('Show Columns?', df.columns)
if plotscatter:
    col1 = st.selectbox('Which feature on x?', df.columns)
    col2 = st.selectbox('Which feature on y?', df.columns)
    fig = px.scatter(df, x =col1,y=col2)
    st.plotly_chart(fig)

if plothist:
    st.subheader("Distributions of each columns")
    options = ('APD','New_Actives_RS_3','Event_Count','Count_Bonus_Offered_SB','Win_Count','Active_Store_Count')
    sel_cols = st.selectbox("select columns", options,1)
    st.write(sel_cols)
    #f=plt.figure()
    fig = go.Histogram(x=df[sel_cols],nbinsx=50)
    st.plotly_chart([fig])

if linechart:
	import plotly.express as px
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df_prec.index, y=df_prec['APD_Actuals'],mode='lines+markers',name='APD_Actuals'))
	fig.add_trace(go.Scatter(x=df_feb_prec.index, y=df_feb_prec['APD_Regression_Predict_base_feb'],mode='lines+markers',name='APD_Regression_Predict_Base_Feb RMSE:'+str(rmse_feb)))
	fig.add_trace(go.Scatter(x=df_prec.index, y=df_prec['APD_Regression_Predict'],mode='lines+markers',name='APD_Regression_Predict_Base_Oct RMSE:'+str(rmse_oct)))
	fig.add_trace(go.Scatter(x=df_new_prec.index, y=df_new_prec['APD_Regression_Predict_base_nov'],mode='lines+markers',name='APD_Regression_Predict_Base_Nov RMSE:'+str(rmse_nov)))
	#fig.add_trace(go.Scatter(x=bsts_oct.index, y=bsts_oct['Oct_Base_Pred'],mode='lines+markers',name='BSTS_Oct_Base_Pred'))
	#fig.add_trace(go.Scatter(x=bsts_nov.index, y=bsts_nov['Nov_Base_Pred'],mode='lines+markers',name='BSTS_Nov_Base_Pred'))
	st.subheader("APD's Actual vs Predict")
	fig.update_layout(yaxis=dict(title_text="APD's",tickmode="array",titlefont=dict(size=15),),autosize=False,width=1000,height=400,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',title={'text': "APD's Actual vs Predicted",'y':0.96,'x':0.5,'xanchor': 'center','yanchor': 'top'})
	#st.line_chart(fig.show())
	#st.line_chart(df_prec[['APD_Actuals','APD_Regression_Predict']])
	st.plotly_chart(fig)
    

#    plt.hist(df[sel_cols])
#    plt.xlabel(sel_cols)
#    plt.ylabel("sales")
#    plt.title(f"{sel_cols} vs Sales")
    #plt.show()	
#    st.plotly_chart(f)


#if distView:
#	st.subheader("Combined distribution viewer")
	# Add histogram data

	# Group data together
#	hist_data = [df["APD"].values,df["New_Actives_RS_3"].values,df["Event_Count"].values,df["Count_Bonus_Offered_SB"].values,df["Win_Count"].values,df["Active_Store_Count"].values]

#	group_labels = ['APD','New_Actives_RS_3','Event_Count','Count_Bonus_Offered_SB','Win_Count','Active_Store_Count']

	# Create distplot with custom bin_size
#	fig = ff.create_distplot(hist_data, group_labels)

	# Plot!
#	st.plotly_chart(fig)

if _3dplot:
	import plotly.graph_objects as px
	import plotly.graph_objects as go
	x = df.index
	df_total = df_prec['APD_Regression_Predict']
	df_rel = df_prec[df_prec.columns[:-2]].div(df_total, 0)*100
	fig = go.Figure()
	fig.add_trace(go.Bar(y=df_rel["Event_Count"],x=df_rel.index,name="Event_Count (Avg ="+str(round(df_rel['Event_Count'].mean(),2))+" %)",text=df_rel['Event_Count'].round(2).astype(str) + '%',textposition='auto',marker=dict(color='rgba(0,128,0, 0.6)',line=dict(color='rgba(0,128,0, 0.5)', width=0.05))))
	fig.add_trace(go.Bar(y=df_rel["New_Actives_RS_3"],x=df_rel.index,name="New_Actives_RS_3 (Avg ="+str(round(df_rel['New_Actives_RS_3'].mean(),2))+" %)",text=df_rel['New_Actives_RS_3'].round(2).astype(str) + '%',textposition='auto',marker=dict(color='rgba(0,0,255, 0.6)',line=dict(color='rgba(0,0,255, 0.5)', width=0.05))))
	fig.add_trace(go.Bar(y=df_rel["Count_Bonus_Offered_SB"],x=df_rel.index,name="Count_Bonus_Offered_SB (Avg ="+str(round(df_rel['Count_Bonus_Offered_SB'].mean(),2))+" %)",text=df_rel['Count_Bonus_Offered_SB'].round(2).astype(str) + '%',textposition='auto',marker=dict(color='rgba(240, 52, 52, 1)',line=dict(color='rgba(240, 52, 52, 1)', width=0.05))))
	fig.add_trace(go.Bar(y=df_rel["Win_Count"],x=df_rel.index,name="Win_Count (Avg ="+str(round(df_rel['Win_Count'].mean(),2))+" %)",text=df_rel['Win_Count'].round(2).astype(str) + '%',textposition='auto',marker=dict(color='rgba(0, 181, 204, 1)',line=dict(color='rgba(0, 181, 204, 1)', width=0.05))))
	fig.add_trace(go.Bar(y=df_rel["Active_Store_Count"],x=df_rel.index,name="Active_Store_Count (Avg ="+str(round(df_rel['Active_Store_Count'].mean(),2))+" %)",text=df_rel['Active_Store_Count'].round(2).astype(str) + '%',textposition='auto',marker=dict(color='rgba(224, 130, 131, 1)',line=dict(color='rgba(224, 130, 131, 1)', width=0.05))))
	fig.add_trace(go.Bar(y=df_rel["Intercept"],x=df_rel.index,name="Intercept (Avg ="+str(round(df_rel['Intercept'].mean(),2))+" %)",text=df_rel['Intercept'].round(2).astype(str) + '%',textposition='auto',marker=dict(color='rgba(250, 190, 88, 1)',line=dict(color='rgba(250, 190, 88, 1)', width=0.05))))
	fig.update_layout(yaxis=dict(title_text="Percentage %",ticktext=["0%", "20%", "40%", "60%","80%","100%"],tickvals=[0, 20, 40, 60, 80, 100],tickmode="array",titlefont=dict(size=15),),autosize=False,width=1000,height=400,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',title={'text': "Independent Variable Infulence On Regression Prediction %",'y':0.96,'x':0.5,'xanchor': 'center','yanchor': 'top'},barmode='stack')

 

                  
	st.plotly_chart(fig.update_layout(barmode='stack'))


# trainmodel= st.checkbox("Train model", False)

#if trainmodel:
#	st.header("Modeling")
#	y=df.iloc[:402,:1]
#	X=df.iloc[:402,1:]
#	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

#	lrgr = LinearRegression()
#	lrgr.fit(X_train,y_train)
#	pred = lrgr.predict(X_test)

#	mse = mean_squared_error(y_test,pred)
#	rmse = sqrt(mse)
#	r2 = r2_score(y_test,pred)
#	adjr2 = 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)        
#	st.markdown(f"""

#	Linear Regression model trained :
#		- MSE:{mse}
#	    - RMSE:{rmse}
#        - R SQUARE:{r2}
#        - ADJ R SQUARE:{adjr2}
#        """)
#	st.success('Model trained successfully')


#if dokfold:
#	st.subheader("KFOLD Random sampling Evalution")
#	st.empty()
#	my_bar = st.progress(0)

#	from sklearn.model_selection import KFold

#	X=df.values[:,-1].reshape(-1,1)
#	y=df.values[:,-1]
	#st.progress()
#	kf=KFold(n_splits=10)
	#X=X.reshape(-1,1)
#	mse_list=[]
#	rmse_list=[]
#	r2_list=[]
#	idx=1
#	fig=plt.figure()
#	i=0
#	for train_index, test_index in kf.split(X):
	#	st.progress()
#		my_bar.progress(idx*10)
#		X_train, X_test = X[train_index], X[test_index]
#		y_train, y_test = y[train_index], y[test_index]
#		lrgr = LinearRegression()
#		lrgr.fit(X_train,y_train)
#		pred = lrgr.predict(X_test)
		
#		mse = mean_squared_error(y_test,pred)
#		rmse = sqrt(mse)
#		r2=r2_score(y_test,pred)
#		mse_list.append(mse)
#		rmse_list.append(rmse)
#		r2_list.append(r2)
#		plt.plot(pred,label=f"dataset-{idx}")
#		idx+=1
#	plt.legend()
#	plt.xlabel("Data points")
#	plt.ylabel("PRedictions")
#	plt.show()
#	st.plotly_chart(fig)

#	res=pd.DataFrame(columns=["MSE","RMSE","r2_SCORE"])
#	res["MSE"]=mse_list
#	res["RMSE"]=rmse_list
#	res["r2_SCORE"]=r2_list

#	st.write(res)
#	st.balloons()
#st.subheader("results of KFOLD")

#f=res.plot(kind='box',subplots=True)
#st.plotly_chart([f])
