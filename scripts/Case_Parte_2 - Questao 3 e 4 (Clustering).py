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
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from scipy import stats
from sklearn.decomposition import PCA
import numpy as np
pd.set_option('max_columns', None)
pd.set_option('use_inf_as_na', True)


# + [markdown] heading_collapsed=true
# # Common functions

# + hidden=true
# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2

def find_cluster_size(data, kmax, pca_var=0.95, minibatch=True, batch_size=100):
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    pca = PCA(n_components=pca_var)
    points = pca.fit_transform(data_scaled)
    print('Original Data: {}, PCA: {}'.format(data_scaled.shape[1], points.shape[1]))
    
    wss = []
    cluster_sizes = []
    silhouettes = []
    
    for k in range(1, kmax):
        if k % 5 == 0:
            print(k)
        if minibatch:
            kmeans = MiniBatchKMeans(n_clusters = k, batch_size=batch_size).fit(points)
        else:
            kmeans = KMeans(n_clusters = k).fit(points)

        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)

        wss.append(kmeans.inertia_)
#         if k > 1:
#             silhouettes.append(silhouette_score(points, pred_clusters, metric = 'euclidean'))
    
    cluster_sizes = [f for f in range(1, kmax)]
    
    metrics = pd.DataFrame({'cluster_size':cluster_sizes, 'wss':wss})
    
    return metrics


# + [markdown] heading_collapsed=true
# # User Profile

# + [markdown] heading_collapsed=true hidden=true
# ## Data Load

# + hidden=true
students_df = pd.read_csv('./data/prep/user_infos.csv').set_index('student_id')

# + hidden=true
students_df.isnull().mean()

# + hidden=true
students_df.loc[:, 'on_top_20_university'] = students_df['on_top_20_university'].astype(int)

# + hidden=true
student_profile = students_df.loc[:, ['signup_source', 'origin', 'course_area', 'on_top_20_university', 'region']]

# + hidden=true
student_profile_dummies = pd.get_dummies(student_profile)
student_profile_dummies.head()

# + [markdown] heading_collapsed=true hidden=true
# ## Cluster

# + [markdown] heading_collapsed=true hidden=true
# ### Find cluster size

# + hidden=true
metrics = find_cluster_size(student_profile_dummies.values, 30, 0.8, False)    

# + hidden=true
fig = px.line(metrics, x='cluster_size', y='wss')
fig.show()

# + [markdown] hidden=true
# Apesar do método nos indicar uma melhor separação em clusters por volta de 8, para fins de simplificar nossa análise, vamos escolher 5 como o número de clusters a analisar, dado uma lógica de atribuição mais genérica.

# + hidden=true
# import hdbscan

# data = student_profile_dummies
# clusterer = hdbscan.HDBSCAN(min_cluster_size=int(data.shape[0]/15), gen_min_span_tree=True, min_samples=1)

# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(student_profile_dummies.values)
# pca = PCA(n_components=0.95)
# points = pca.fit_transform(data_scaled)

# clusterer.fit(points)
# clusterer.labels_.max()

# + [markdown] heading_collapsed=true hidden=true
# ## Find clusters

# + hidden=true
scaler = StandardScaler()
data_scaled = scaler.fit_transform(student_profile_dummies.values)
pca = PCA(n_components=0.95)
points = pca.fit_transform(data_scaled)
kmeans = KMeans(n_clusters = 5, random_state=42).fit(points)

labels = kmeans.labels_


# + hidden=true
students_df.head()

# + hidden=true
clusters_df = student_profile_dummies.copy()
clusters_df.loc[:, 'converted'] = students_df['has_purchased'].astype(float)
clusters_df.loc[:, 'ltv'] = students_df['ltv']
clusters_df.loc[:, 'cohort'] = students_df['signup_at'].str[:7]
clusters_df.loc[:, 'cluster'] = labels
clusters_df.loc[:, 'student_id'] = students_df.index

# + hidden=true
metrics_to_calc = {c:'mean' for c in student_profile_dummies.columns.tolist()}
metrics_to_calc['converted'] = 'mean'
metrics_to_calc['ltv'] = 'mean'
metrics_to_calc['student_id'] = 'count'

clusters_mean = clusters_df\
                    .groupby(['cluster'])\
                    .agg(metrics_to_calc)

clusters_melt = clusters_mean\
                    .stack().to_frame().reset_index()

clusters_melt.columns = ['cluster', 'variable_dummy', 'value']

# + hidden=true
# clusters_melt.loc[:, 'variable'] = clusters_melt['variable_dummy'].apply(lambda x: '_'.join(x.split('_')[:-1]))
# clusters_melt.loc[:, 'category'] = clusters_melt['variable_dummy'].apply(lambda x: x.split('_')[-1])

# + hidden=true
clusters_melt.loc[:, 'var_max'] = clusters_melt.groupby(['variable_dummy'])['value'].rank(ascending=False, method='first')
clusters_melt.loc[:, 'max_val'] = clusters_melt.groupby(['variable_dummy'])['value'].transform('max')
clusters_melt.loc[:, 'min_val'] = clusters_melt.groupby(['variable_dummy'])['value'].transform('min')

clusters_melt.loc[:, 'cluster_highlight'] = clusters_melt.apply(lambda x: x['var_max'] <= 1 or x['value'] == x['min_val'], axis=1)


# + hidden=true
clusters_mean

# + hidden=true
for c in clusters_melt['cluster'].unique().tolist():
    df = clusters_melt.loc[(clusters_melt['cluster'] == c) & (clusters_melt['cluster_highlight'])]
    display(df)

# + hidden=true
clusters_df.head()

# + hidden=true
clusters_cohorts = clusters_df\
                    .groupby(['cluster', 'cohort'], as_index=True)\
                    .agg({'ltv':'mean', 'converted':['mean', 'sum', 'count']})
clusters_cohorts.columns = ['ltv', 'conversion_rate', 'converted', 'students']

clusters_cohorts.loc[:, 'students'] = clusters_cohorts['students'].astype(float)
clusters_cohorts = clusters_cohorts.reset_index()

# + hidden=true
fig = px.line(clusters_cohorts, x='cohort', y=['students', 'converted', 'ltv', 'conversion_rate',], 
              color='cluster', facet_col='variable', facet_row='cluster')
fig.update_yaxes(matches=None, showticklabels=True)
# -

# # User Behaviour

# ## Data prep

# +
user_behaviour = pd.read_csv('./data/prep/user_activity_summary.csv')#.set_index(['student_id', 'month'])

# Create relative metrics
activity_cols = ['question_events', 'subject_events', 'subject_events']
for c in activity_cols:
    user_behaviour.loc[:, c + '_percent'] = user_behaviour[c]/user_behaviour['total_activities']

events_cols = [c for c in user_behaviour.columns.tolist() if 'events_' in c]
for c in events_cols:
    user_behaviour.loc[:, c + '_percent'] = user_behaviour[c]/user_behaviour['total_events']

days_cols = [c for c in user_behaviour.columns.tolist() if 'days' in c]
days_cols.remove('total_days')
for c in days_cols:
    user_behaviour.loc[:, c + '_percent'] = user_behaviour[c]/user_behaviour['total_days']

user_behaviour.loc[:, 'weekend_use_percent'] = user_behaviour['days_on_weekend']/user_behaviour['total_days']
user_behaviour.loc[:, 'week_use_percent'] = 1 - user_behaviour['weekend_use_percent']

user_behaviour.head()

# +
# Remove outliers

activity_df = user_behaviour.drop(columns=['first_event', 'last_event', 'student_id', 'month']).fillna(0)

activity_df.loc[:, 'outlier'] = False

outlier_dict = activity_df.quantile(0.999).to_dict()

for k, v in outlier_dict.items():
    activity_df.loc[activity_df[k] > v, 'outlier'] = True

cluster_activity = activity_df[~activity_df['outlier']].iloc[:, :-1]

cluster_activity.head()
# -

# ## Find cluster size

metrics = find_cluster_size(cluster_activity.values, 20, 0.8, False)

fig = px.line(metrics, x='cluster_size', y='wss')
fig.show()

# Vamos focar em 5 clusters.

# ## Find Clusters

# +
scaler = StandardScaler()
data_scaled = scaler.fit_transform(cluster_activity.values)
pca = PCA(n_components=0.8)
points = pca.fit_transform(data_scaled)
kmeans = KMeans(n_clusters = 5, random_state=42).fit(points)

data_scaled = scaler.transform(activity_df.iloc[:, :-1].values)
points = pca.transform(data_scaled)

labels_behav = kmeans.predict(points)
# -

behav_clusters = activity_df.copy()
behav_clusters.loc[:, 'cluster'] = labels_behav
behav_clusters.loc[:, 'student_id'] = user_behaviour['student_id']
behav_clusters.loc[:, 'month'] = user_behaviour['month']

# +
metrics_to_calc = {c:'mean' for c in cluster_activity.columns.tolist()}
# metrics_to_calc['converted'] = 'mean'
# metrics_to_calc['ltv'] = 'mean'
metrics_to_calc['student_id'] = 'count'

clusters_mean = behav_clusters\
                    .groupby(['cluster'])\
                    .agg(metrics_to_calc)

clusters_melt = behav_clusters\
                    .stack().to_frame().reset_index()

clusters_melt.columns = ['cluster', 'variable_dummy', 'value']
# -

clusters_mean.sort_values(by=['total_events', 'total_days'])

# 5 tipos de uso:
# * Mobile fileview
# * Web fileview
# * Subject following
# * Casual studying
# * Super fileview

# +
cluster_labels = {0:'mobile_fileview', 2:'web_fileview', 1:'subject_following', 4:'casual_studying', 3:'super_fileview'}

behav_clusters.loc[:, 'cluster_name'] = behav_clusters['cluster'].apply(lambda x: cluster_labels[x])

behav_clusters.loc[:, 'first_month'] = behav_clusters.groupby(['student_id'])['month'].transform('min')

# +
first_behav = behav_clusters\
                .loc[behav_clusters['month'] == behav_clusters['first_month'], ['cluster_name', 'student_id']]\
                .set_index('student_id')
    
first_behav.columns = ['first_cluster']

first_behav.head()

# +
user_behav_cluster = behav_clusters\
                        .groupby(['student_id', 'cluster_name'], as_index=False)\
                        .agg({'month':['count', 'min']})\

user_behav_cluster.columns = ['student_id', 'main_cluster', 'months_as_main', 'main_first_month']

user_behav_cluster = user_behav_cluster.sort_values(by=['student_id', 'months_as_main', 'main_first_month'])

user_behav_cluster.loc[:, 'max_behav'] = user_behav_cluster.groupby(['student_id'])['months_as_main'].rank(method='first', ascending=False)

main_behaviour = user_behav_cluster.loc[user_behav_cluster['max_behav'] == 1].iloc[:, :-1]
main_behaviour.head()

# +
user_behav = behav_clusters\
                .groupby(['student_id'], as_index=True)\
                .agg({'month':['count', 'min', 'max'], 'cluster_name':'nunique'})\

user_behav.columns = ['total_time', 'first_month', 'last_month', 'unique_clusters']

user_behav = pd.merge(left=user_behav.reset_index(), right=main_behaviour, on=['student_id']).set_index('student_id')

user_behav.loc[:, 'first_cluster'] = first_behav['first_cluster']

user_behav.head()
# -

user_behav.to_csv('./data/prep/user_activity_cluster_summary.csv', index=True)


# ## Data Analysis

def plot(fig):
    fig.update_layout(
        font=dict(family="Roboto", size=16),
        template='plotly_white'
    )
    colors = ['#250541', '#F93D55']
    fig.show()


students_df = pd.read_csv('./data/prep/user_infos.csv').set_index('student_id')
students_df.head()

user_behav = pd.read_csv('./data/prep/user_activity_cluster_summary.csv').set_index('student_id')
user_behav.head()

# +
summ_df = students_df.loc[:, ['signup_at', 'origin', 'course_area', 'on_top_20_university', 'total_plans', 'signup_source',
                              'region', 'ltv', 'first_purchase', 'has_purchased', 'revenue_first_purchase']]

summ_df = pd.concat([summ_df, user_behav], axis=1).reset_index()

summ_df.loc[:, 'signup_month'] = summ_df['signup_at'].str[:7]
summ_df.loc[:, 'converted'] = summ_df['has_purchased'].astype(int)
summ_df.loc[:, 'students'] = 1
summ_df.head()

# + [markdown] heading_collapsed=true
# ### User Features 

# + hidden=true
df = summ_df\
        .groupby(['course_area', 'has_purchased'], as_index=False)\
        .agg({'student_id':'count'})\
        .sort_values(by='student_id', ascending=False)

fig = px.bar(df, y='course_area', x='student_id', facet_col='has_purchased', color='course_area')
fig.update_traces(texttemplate='%{x}', textposition='inside')
fig.update_xaxes(matches=None, title='Total de estudantes')

fig.update_layout(title='Conversão dos usuários, por área', 
                  yaxis_title='Área de interesse',
                  showlegend=False
                 )
plot(fig)

# + hidden=true
df = summ_df\
        .groupby(['origin', 'has_purchased'], as_index=False)\
        .agg({'student_id':'count'})\
        .sort_values(by='student_id', ascending=False)

df.loc[:, 'total'] = df.groupby(['has_purchased'])['student_id'].transform('sum')
df.loc[:, 'percent'] = 100*df['student_id']/df['total']

fig = px.bar(df, y='origin', x='percent', facet_col='has_purchased', color='origin')
fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
fig.update_xaxes(matches=None, title='Percentual de estudantes (%)', range=[0,100])

fig.update_layout(title='Conversão de usuários, por dispositivo', 
                  yaxis_title='Dispositivo do cadastro',
                  showlegend=False
                 )
plot(fig)

# + hidden=true
df = summ_df\
        .groupby(['signup_source', 'has_purchased'], as_index=False)\
        .agg({'student_id':'count'})\
        .sort_values(by='student_id', ascending=False)

df.loc[:, 'total'] = df.groupby(['has_purchased'])['student_id'].transform('sum')
df.loc[:, 'percent'] = 100*df['student_id']/df['total']

fig = px.bar(df, y='signup_source', x='percent', facet_col='has_purchased', color='signup_source')
fig.update_traces(texttemplate='%{x:.1f}%', textposition='auto')
fig.update_xaxes(matches=None, title='Percentual dos estudantes (%)')

fig.update_layout(title='Conversão de usuários, por origem do cadastro', 
                  yaxis_title='Origem do cadastro',
                  showlegend=False
                 )
plot(fig)

# + hidden=true
df = summ_df\
        .groupby(['region', 'has_purchased'], as_index=False)\
        .agg({'student_id':'count'})\
        .sort_values(by='student_id', ascending=False)

df.loc[:, 'total'] = df.groupby(['has_purchased'])['student_id'].transform('sum')
df.loc[:, 'percent'] = 100*df['student_id']/df['total']

fig = px.bar(df, y='region', x='percent', facet_col='has_purchased', color='region')
fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
fig.update_xaxes(matches=None, title='Percentual dos estudantes (%)', range=[0,80])

fig.update_layout(title='Conversão dos usuários, por região', 
                  yaxis_title='Região do usuário',
                  showlegend=False
                 )
plot(fig)
# -

# ### User behaviour

# +
df = summ_df\
        .groupby(['first_cluster'], as_index=False)\
        .agg({'student_id':'count'})\
        .sort_values(by='student_id', ascending=True)

fig = px.bar(df, y='first_cluster', x='student_id')
fig.update_traces(texttemplate='%{x}', textposition='auto')
fig.update_layout(title='Padrão de comportamento no 1º mês', 
                  xaxis_title='Quantidade de usuários',
                  yaxis_title=' Comportamento no 1º mês'
                 )
plot(fig)

# +
df = summ_df\
        .groupby(['first_cluster', 'has_purchased'], as_index=False)\
        .agg({'student_id':'count'})\
        .sort_values(by='student_id', ascending=False)

df.loc[:, 'total'] = df.groupby(['has_purchased'])['student_id'].transform('sum')
df.loc[:, 'percent'] = 100*df['student_id']/df['total']

fig = px.bar(df, y='first_cluster', x='percent', facet_col='has_purchased', color='first_cluster')
fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
fig.update_xaxes(matches=None, showticklabels=True, title='Porcentagem de usuários (%)',  range=[0,80])
fig.update_layout(title='Padrão de comportamento por usuário, no 1o mês', 
                  yaxis_title='Perfil de uso',
                  showlegend=False
                 )
plot(fig)

# +
df = summ_df\
        .groupby(['first_month', 'first_cluster'], as_index=False)\
        .agg({'student_id':'count'})\
        .sort_values(by='first_month', ascending=False)

fig = px.line(df, x='first_month', y='student_id', color='first_cluster')
fig.update_layout(title='Evolução do comportamento do 1º mês de atividade, por safra', 
                  yaxis_title='Quantidade de usuários',
                  xaxis_title='Mês do usuário',
                  legend_orientation='h',
                  height=700
                 )
plot(fig)

# +
df = summ_df\
        .groupby(['first_cluster', 'course_area'], as_index=False)\
        .agg({'student_id':'count'})\
        .sort_values(by='student_id', ascending=False)

df.loc[:, 'total'] = df.groupby(['first_cluster'])['student_id'].transform('sum')
df.loc[:, 'percent'] = 100*df['student_id']/df['total']

fig = px.bar(df, facet_col='first_cluster', x='student_id', y='course_area', 
                   color='course_area', text='percent')

fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
fig.update_xaxes(matches=None, title='')

fig.update_layout(title='Distribuição do comportamento do 1o mês, por curso de interesse', 
                  yaxis_title='Curso de interesse',
                  showlegend=False
                 )

fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1]))
plot(fig)
# -

# ### LTV

# +
df = summ_df\
        .loc[summ_df['has_purchased']]\
        .groupby(['main_cluster'], as_index=False)\
        .agg({'student_id':'count', 'revenue_first_purchase':'mean', 
              'ltv':'mean', 'total_plans':'mean'})\
        .sort_values(by='ltv', ascending=True)


fig = px.bar(df, y='main_cluster', x=['revenue_first_purchase', 'ltv'], barmode='group')

fig.update_traces(texttemplate='R$%{x:.1f}', textposition='inside')

fig.update_layout(title='LTV e Receita de primeira compra, por tipo de comportamento mais comum do usuário', 
                  yaxis_title='Comportamento mais comum',
                  xaxis_title='Valor em receita (R$)',
                  showlegend=True
                 )
plot(fig)

# +
df = summ_df\
        .loc[summ_df['has_purchased']]\
        .groupby(['first_cluster'], as_index=False)\
        .agg({'student_id':'count', 'revenue_first_purchase':'mean', 
              'ltv':'mean', 'total_plans':'mean'})\
        .sort_values(by='ltv', ascending=True)


fig = px.bar(df, y='first_cluster', x=['revenue_first_purchase', 'ltv'], barmode='group')

fig.update_traces(texttemplate='R$%{x:.1f}', textposition='inside')

fig.update_layout(title='LTV e Receita de primeira compra, por primeiro comportamento do usuário', 
                  yaxis_title='Comportamento mais comum',
                  xaxis_title='Valor em receita (R$)',
                  showlegend=True
                 )

plot(fig)

# +
df = summ_df\
        .groupby(['has_purchased', 'total_time'], as_index=False)\
        .agg({'student_id':'count'})\
        .sort_values(by=['has_purchased', 'total_time'], ascending=True)

df.loc[:, 'cumsum'] = df.groupby(['has_purchased'])['student_id'].cumsum()
df.loc[:, 'total'] = df.groupby(['has_purchased'])['student_id'].transform('sum')

df.loc[:, 'students_over'] = df['total'] - df['cumsum']

fig = px.histogram(df, x='total_time', y='student_id', histfunc='sum', nbins=50,
                   facet_col='has_purchased')

fig.update_yaxes(matches=None, showticklabels=True)
fig.update_xaxes(title='Meses com atividade')
# fig.update_traces(texttemplate='R$%{x:.1f}', textposition='inside')

fig.update_layout(title='Conversão x Tempo de atividade dos usuários', 
                  yaxis_title='Total de usuários',
                  showlegend=True
                 )

plot(fig)
df.head()
