import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pymrio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

##########################333HAPPINESS#######################
df = pd.read_csv("/home/grover/Documents/pythondoc/happiness_data/happiness.csv", sep=",")

#Print the top 10 entries
df.head(10)

t = np.linspace(0, 10, 214)
df11 = df[df['Time']=='2011']


import plotly.express as px
from plotly.subplots import make_subplots
pd.options.plotting.backend = "plotly"
#import plotly.graph_objects as go
dfa = df11
'''
#fig = px.scatter(dfa, x='Life expectancy at birth, total (years) [SP.DYN.LE00.IN]', y='Population growth (annual %) [SP.POP.GROW]', color="species")
#fig.show()

fig = dfa.plot.scatter(x=t, y='Population growth (annual %) [SP.POP.GROW]')
fig.show()

fig2 = dfa.plot.scatter(x=t, y='Life expectancy at birth, total (years) [SP.DYN.LE00.IN]')
fig2.show()
'''

fig = go.Figure()
fig = make_subplots(rows=8, cols=1)
# Add traces
fig.add_trace(go.Scatter(x=t, y=dfa['CO2 emissions (kg per PPP $ of GDP) [EN.ATM.CO2E.PP.GD]'],
                    mode='markers',
                    text=dfa['CO2 emissions (kg per PPP $ of GDP) [EN.ATM.CO2E.PP.GD]'],
                    name='markers'))




fig.update_layout(height=10000, width=2000, title_text="Side By Side Subplots")
fig.show()

#WELLBEING
###########################################WELLBEING#####################################
#######################################################
df2 = pd.read_csv("/home/grover/Documents/pythondoc/wellbeing/wellbeing.csv", sep=",")

#Print the top 10 entries
t2 = np.linspace(0, 10, 386)
#fig3 = df2.plot.scatter(x=t2, y='Value', text='Regions')
#fig3.show()
fig3 = go.Figure(data=go.Scatter(x=t2,
                                y=df2['Value'],
                                mode='markers',
                                
                                text=df2['Regions'])) # hover text goes here

fig3.update_layout(title='Wellbeing 2011  Data')
fig3.show()



#################################COnsumption##################################3
# df2 is for the final consumption
'''
df3=pd.read_excel('output.xlsx', index_col=0)


###############ENvironmental impacts######################333

exio3 = pymrio.parse_exiobase3('/home/grover/Documents/pythondoc/exiobase3.4_iot_2011_pxp/IOT_2011_pxp')
exio3.calc_all()
Cm = exio3.Y
CmSE = exio3.Y.SE
CmSEg = exio3.Y.SE.groupby(level=1, sort=False).sum()
CmSEgAll = exio3.Y.SE.groupby(level=1, sort=False).sum()
CmSEgAll = CmSEgAll.sum(axis=1)
CmSEgAll = pd.DataFrame(CmSEgAll)
CmSEgAll.columns = ['Final consumption expenditure in Sweden by all stakeholder groups']
DcbaAll = exio3.satellite.D_cba

cf = pd.read_csv('Cha_Fac_footprints_RL.csv',';', header=[0], index_col = [0])

cor = pd.read_excel('Exio_COICOP12.xlsx', header=[0], index_col = [0])

DcbaAll = exio3.satellite.D_cba

names=exio3.get_regions()
nl=names.values.tolist()
type(nl)
var=nl[4]
var='AT'

D_cbaSE = exio3.satellite.D_cba
D_cbaSE=D_cbaSE.loc[:, var]


D_cbaSE_fp = D_cbaSE.T.dot(cf.T) #footprints of 200 exiobase consumption categories
SEfp = D_cbaSE_fp.T.dot(cor.T).T #footprints of 12 coicop categories





'''