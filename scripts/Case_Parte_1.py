# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python [conda env:root] *
#     language: python
#     name: conda-root-py
# ---

import pandas as pd
import plotly.express as px
from sklearn.preprocessing import KBinsDiscretizer

# Load data
premium = pd.read_json('./data/BASE A/premium_students.json')
premium.head()

# Check nulls
premium.isnull().mean()

# +
# Data prep
premium.loc[:, 'RegisteredDate'] = pd.to_datetime(premium['RegisteredDate'], infer_datetime_format=True)
premium.loc[:, 'SubscriptionDate'] = pd.to_datetime(premium['SubscriptionDate'], infer_datetime_format=True)

premium.loc[:, 'days_until_subscription'] = (premium['SubscriptionDate'] - premium['RegisteredDate']).dt.days

premium.loc[:, 'subscription_date'] = (premium['SubscriptionDate']).dt.date
premium.loc[:, 'registration_date'] = (premium['RegisteredDate']).dt.date

premium.head()
# -

fig = px.histogram(premium, x=['registration_date', 'subscription_date'], facet_col='variable')
# fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1]))
fig.update_layout(showlegend=False,
                  yaxis_title='Quantidade de usuários',
                  title='Cadastro e conversão, no período'
                 )
fig.update_xaxes(title='Período')
plot(fig)


def plot(fig):
    fig.update_layout(
        font=dict(family="Roboto", size=16),
        template='plotly_white'
    )
    colors = ['#250541', '#F93D55']
    fig.show()


# +
df = premium\
        .groupby(['days_until_subscription'], as_index=False)\
        .agg({'StudentId':'nunique'})

fig = px.bar(df, x='days_until_subscription', y='StudentId')
fig.update_layout(
    xaxis_title='Dias entre cadastro e primeira compra',
    yaxis_title='Quantidade de usuários',
    title='Distribuição do tempo até conversão'
)
fig.update_xaxes(tickmode = 'linear',
        dtick = 30
    )
plot(fig)

# +
premium.loc[premium['days_until_subscription'] == 0, 'class'] = '0 dias'
premium.loc[premium['days_until_subscription'].between(1, 30), 'class'] = '1 a 30 dias'
premium.loc[premium['days_until_subscription'] > 30, 'class'] = '+30 dias'

premium['class'].value_counts()

# +
df = premium\
        .groupby(['class'], as_index=False)\
        .agg({'StudentId':'count', 'days_until_subscription':'mean'})\
        .sort_values(by='days_until_subscription')

df.loc[:,'total'] = df['StudentId'].sum()
df.loc[:,'percentage'] = 100*df['StudentId']/df['total']

fig = px.bar(df, x='class', y='percentage')
fig.update_xaxes(type='category', title='Categoria')
fig.update_yaxes(tickvals=[], title='Percentual (%)')
fig.update_traces(texttemplate='<b>%{y:.1f}%<b>', textposition='outside', )
fig.update_layout(
    title='Distribuição dos usuários, por categoria'
)
plot(fig)
# -


