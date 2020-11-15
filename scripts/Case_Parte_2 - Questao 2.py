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


def plot(fig):
    fig.update_layout(
        font=dict(family="Roboto", size=16),
        template='plotly_white'
    )
    colors = ['#250541', '#F93D55']
    fig.show()


# # Data Load and Prep

plans_raw = pd.read_csv('./data/prep/premium_plans_info.csv')

# +
# Consider only not cancelled plans
plans = plans_raw.loc[plans_raw['cancelled_at'].isnull()].copy().sort_values(by='created_at')

plans.loc[:, 'cohort'] = plans['created_at'].str[:7]

plans.head()
# -

plans['created_at'].max()

# # Analysis

# +
# Group users all time revenue

user_ltv = plans\
            .groupby(['student_id'], as_index=False)\
            .agg({'plan_id':'nunique', 'created_at':'min', 'plan_type':'nunique', 'cost':'sum'})

user_ltv.loc[:, 'cohort'] = user_ltv['created_at'].str[:7]

cohort_ltv = user_ltv.groupby(['cohort'], as_index=False).agg({'cost':['sum', 'mean'], 'student_id':'nunique'})

cohort_ltv.columns = ['cohort', 'total_cohort_revenue', 'user_ltv', 'total_users']
cohort_ltv.loc[:, 'total_users'] = cohort_ltv['total_users'].astype(float)
cohort_ltv.head()

df = cohort_ltv.loc[cohort_ltv['cohort'] >= '2017-10']
fig = px.line(df, x='cohort', y='user_ltv')
fig.update_yaxes(matches=None, showticklabels=True, title='')
fig.update_layout(showlegend=False, 
                  xaxis_title='Cohort do usuário',
                  yaxis_title='LTV (R$)',
                  title='LTV por cohort'
                 )
fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1]))

plot(fig)
# -
cohort_ltv.loc[cohort_ltv['cohort'].between('2017-11', '2018-02'), 'user_ltv'].median()



