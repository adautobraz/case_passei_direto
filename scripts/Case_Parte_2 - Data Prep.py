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

# +
import pandas as pd
import plotly.express as px
from IPython.display import display
import numpy as np

pd.set_option('max_rows', 100)
pd.set_option('max_columns', None)

# + [markdown] heading_collapsed=true
# ## Events

# + hidden=true
activity_columns = ['student_id', 'datetime', 'student_client']

# + [markdown] heading_collapsed=true hidden=true
# ### File Views

# + hidden=true
# File Views
fileviews_raw = pd.read_json('./data/BASE B/fileViews.json')
display(fileviews_raw.head())

# + hidden=true
fileviews = fileviews_raw.loc[:, ['StudentId', 'ViewDate', 'Studentclient']]
fileviews.columns = activity_columns
fileviews.loc[:, 'event_type'] = 'file_view'
del fileviews_raw

# + hidden=true
fileviews.head()

# + [markdown] heading_collapsed=true hidden=true
# ### Questions

# + hidden=true
# Questions
questions_raw = pd.read_json('./data/BASE B/questions.json')
display(questions_raw.head())

# + hidden=true
questions = questions_raw.loc[:, ['StudentId', 'QuestionDate', 'StudentClient']]
questions.columns = activity_columns
questions.loc[:, 'event_type'] = 'question_created'
del questions_raw

# + hidden=true
questions.head()

# + [markdown] heading_collapsed=true hidden=true
# ### Subjects

# + hidden=true
subjects_raw = pd.read_json('./data/BASE B/subjects.json')
subjects_raw.head()

# + hidden=true
subjects = subjects_raw.loc[:, ['StudentId', 'FollowDate']]
subjects.loc[:, 'StudentClient'] = ''
subjects.columns = activity_columns
subjects.loc[:, 'event_type'] = 'subject_followed'
del subjects_raw

# + hidden=true
subjects.loc[subjects['student_id'] == 2774]

# + [markdown] heading_collapsed=true hidden=true
# ### Sessions

# + hidden=true
# Load Data
sessions_raw = pd.read_json('./data/BASE B/sessions.json')
sessions_raw.head()

# + hidden=true
sessions = sessions_raw.loc[:, ['StudentId', 'SessionStartTime', 'StudentClient']]
sessions.columns = activity_columns
sessions.loc[:, 'event_type'] = 'session_started'
del sessions_raw

# + [markdown] heading_collapsed=true hidden=true
# ### All Events

# + hidden=true
activity_df = pd.concat([subjects, questions, fileviews, sessions])
activity_df.columns = ['student_id', 'event_time', 'user_client', 'event_type']
activity_df.head()

# + hidden=true
activity_df.to_csv('./data/prep/user_activity_raw.csv', index=False)

# + [markdown] heading_collapsed=true hidden=true
# ### Enrich Activity Data

# + hidden=true
activity_df = pd.read_csv('./data/prep/user_activity_raw.csv').sort_values(by=['student_id', 'event_time'])
activity_df.head()

# + hidden=true
# Data Prep

# Date/Time columns
activity_df.loc[:, 'event_time'] = pd.to_datetime(activity_df['event_time'], infer_datetime_format=True)
activity_df.loc[:, 'month'] = activity_df['event_time'].astype(str).str[:7]
activity_df.loc[:, 'date'] = activity_df['event_time'].astype(str).str[:10]

activity_df.loc[:, 'day_of_week'] = activity_df['event_time'].dt.weekday
activity_df.loc[:, 'weekday'] = activity_df['day_of_week'].between(1,5)
activity_df.loc[~activity_df['weekday'], 'weekend_date'] = activity_df['date']


activity_df.loc[:, 'event_hour'] = activity_df['event_time'].dt.hour
activity_df.loc[:, 'hour_disc'] = 6*(activity_df['event_hour']/6).astype(int)
activity_df.loc[:, 'period_of_day'] = activity_df['hour_disc'].apply(lambda x: '{}-{}'.format(x, x+5))

# User origin and device
activity_df.loc[:, 'user_origin'] = activity_df['user_client']\
                                        .str.split('|', expand=True).iloc[:, 0]\
                                        .str.strip()\
                                        .str.lower()

activity_df.loc[activity_df['user_origin'].isin(['website']), 'device'] = 'web'
activity_df.loc[~activity_df['user_origin'].isin(['website']), 'device'] = 'mobile'
activity_df.loc[activity_df['user_origin'].isnull(), 'device'] = np.nan

# Window functions
activity_df.loc[:, 'last_event'] = activity_df.groupby(['student_id'])['event_time'].shift(1)
activity_df.loc[:, 'last_event_same_type'] = activity_df.groupby(['student_id', 'event_type'])['event_time'].shift(1)

activity_df.loc[:, 'time_since_last_event'] = ((activity_df['event_time'] - activity_df['last_event']).dt.seconds/3600).fillna(0)
activity_df.loc[:, 'time_since_last_event_type'] = ((activity_df['event_time'] - activity_df['last_event_same_type']).dt.seconds/3600).fillna(0)


# + hidden=true
activity_df['device'].value_counts(dropna=False)

# + hidden=true
# Save data
activity_df.to_csv('./data/prep/user_activity_prep.csv', index=False)

# + hidden=true
del activity_df

# + [markdown] heading_collapsed=true hidden=true
# ### User Event Summary

# + hidden=true
user_activity = pd.read_csv('./data/prep/user_activity_prep.csv')

user_activity.head()

# + hidden=true
user_activity['event_type'].value_counts(dropna=False)

# + hidden=true
user_activity.loc[:, 'is_activity'] = 1
user_activity.loc[user_activity['event_type'] == 'session_started', 'is_activity'] = 0

user_activity.loc[:, 'is_session'] = 0
user_activity.loc[user_activity['event_type'] == 'session_started', 'is_session'] = 1

# + hidden=true
# Summarize Data - month
activity_month = user_activity\
                    .groupby(['student_id','month'], as_index=True)\
                    .agg({'event_time':['count', 'min', 'max'],
                          'date':'nunique', 
                          'device': 'nunique',
                          'weekend_date':'nunique',
                          'is_activity':'sum'
                         })
activity_month.columns = ['total_events', 'first_event', 'last_event',\
                              'total_days', 'unique_origins', 'days_on_weekend', 'total_activities']
activity_month.head()

# + hidden=true
# Summarize Data - month, type
activity_type = user_activity\
                    .groupby(['student_id','month', 'event_type'], as_index=True)\
                    .agg({'event_time':'count',
                          'date':'nunique'   
                         })

activity_type.columns = ['events', 'days_used'] 

activity_type = activity_type.unstack().fillna(0)

activity_type.columns = ['{}_{}'.format(c[1].split('_')[0], c[0])
                                                      for c in activity_type.columns.tolist()]

activity_type.head()

# + hidden=true
# Summarize Data - month, device

activity_device = user_activity\
                    .groupby(['student_id','month', 'device'], as_index=True)\
                    .agg({'event_time':'count',
                          'date':'nunique'   
                         })

activity_device.columns = ['events_on', 'used_days_on'] 

activity_device = activity_device.unstack().fillna(0)

activity_device.columns = ['{}_{}'.format(c[0], c[1]) for c in activity_device.columns.tolist()]

activity_device.head()

# + hidden=true
# Summarize Data - month, period
activity_period = user_activity\
                    .groupby(['student_id','month', 'period_of_day'], as_index=True)\
                    .agg({'event_time':'count'})

activity_period.columns = ['events_usage__']

activity_period = activity_period.unstack().fillna(0)

activity_period.columns = [c[0] + c[1] for c in activity_period.columns.tolist()]

activity_period.head()

# + hidden=true
user_monthly_summary = activity_month\
                        .join(activity_type, how='left')\
                        .join(activity_device, how='left')\
                        .join(activity_period, how='left')\
                        .fillna(0)\
                        .reset_index()

user_monthly_summary.head()

# + hidden=true
user_monthly_summary.to_csv('./data/prep/user_activity_summary.csv', index=False)

# + [markdown] heading_collapsed=true
# ## Payments

# + hidden=true
premium_payments = pd.read_json('./data/BASE B/premium_payments.json')

premium_payments.columns = ['student_id', 'created_at', 'plan_type']
premium_payments.loc[:, 'plan_id'] = premium_payments['student_id'].astype(str) + '_' +\
                                    premium_payments['plan_type'] + '_' +\
                                    premium_payments['created_at'].str[:10]

premium_payments.loc[:, 'event_type'] = 'payment'
premium_payments.head()

# + hidden=true
premium_payments['created_at'].max()

# + hidden=true
premium_cancellations = pd.read_json('./data/BASE B/premium_cancellations.json')

premium_cancellations.columns = ['student_id', 'created_at']

premium_cancellations.loc[:, 'plan_type'] = ''
premium_cancellations.loc[:, 'plan_id'] = ''
premium_cancellations.loc[:, 'event_type'] = 'cancelation'

# + hidden=true
premium_cancellations.shape

# + hidden=true
plan_events = pd\
                .concat([premium_cancellations, premium_payments], axis=0)\
                .sort_values(by=['student_id', 'created_at'])\
                .reset_index().drop(columns=['index'])

plan_events.loc[:, 'last_plan_id'] = plan_events\
                                        .groupby(['student_id'])['plan_id'].shift(1)


plan_events.loc[plan_events['event_type'] == 'cancelation', 'plan_id'] =  plan_events['last_plan_id']

cancellations_adj = plan_events\
                        .loc[plan_events['event_type'] == 'cancelation']\
                        .loc[:, ['plan_id', 'created_at']].set_index('plan_id')

cancellations_adj.columns = ['cancelled_at']
    
plans = premium_payments.set_index('plan_id')\
            .join(cancellations_adj, how='left', rsuffix='_cancel')\
            .reset_index()\
            .iloc[:, [0, 1, 2, 3, 5]]

plans.loc[plans['plan_type'] == 'Mensal', 'cost'] = 29.9
plans.loc[plans['plan_type'] == 'Anual', 'cost'] = 286.8
                                                        
(~plans.isnull()).sum()

# + hidden=true
# Check inconsistency
plan_events.loc[plan_events['event_type'] == 'cancelation', 'last_plan_id'].isnull().sum()

# + [markdown] hidden=true
# There are 46 plans that do not appear on the premium plans table.

# + hidden=true
plans.head()

# + hidden=true
# Save data
plans.to_csv('./data/prep/premium_plans_info.csv', index=False)

# + hidden=true
# Load data
plans = pd.read_csv('./data/prep/premium_plans_info.csv')
# -

# ## Students

# + code_folding=[]
# Load Data
students_raw = pd.read_json('./data/BASE B/students.json')
students_raw.head()
# -

students_raw.isnull().mean()

# +
# Data Prep
students_raw.loc[:, 'user_origin'] = students_raw['StudentClient']\
                            .str.split('|', expand=True).iloc[:, 0]\
                            .str.strip().str.lower()

students_raw.loc[~students_raw['user_origin'].isnull(), 'origin'] = 'other'
students_raw.loc[students_raw['user_origin'].isin(['website', 'ios', 'android']), 'origin'] = students_raw['user_origin']
# -

students_raw['origin'].value_counts()

# +
fig = px.histogram(students_raw, x=['RegisteredDate', 'SignupSource', 'State'], facet_col='variable',
                   title='Distribuição de variáveis do usuário')

fig.update_yaxes(matches=None, showticklabels=True)
fig.update_layout(showlegend=False)
fig.show()

# + [markdown] heading_collapsed=true
# ### Course Area

# + hidden=true
# Exploring course distribution
df = students_raw\
        .groupby(['CourseName'], as_index=False)\
        .agg({'Id':'count'})\
        .sort_values(by='Id', ascending=False)

df.columns = ['course', 'students']
df['cumul_students'] = df['students'].cumsum()
df['total_students'] = df['students'].sum()
df['rank'] = df['students'].rank(ascending=False)

df.loc[:, 'cumul_percentage'] = 100*df['cumul_students']/df['total_students']
df.loc[:, 'percentage'] = 100*df['students']/df['total_students']

df.head()

fig = px.area(df, x='rank', y='cumul_percentage', hover_name='course', hover_data=['students', 'percentage'])
fig.show()

df.head(40)

# + hidden=true
course_map = {
    'Direito': ['Direito'],
    'Administração': ['Administração', 'Gestão'],
    'Engenharia': ['Engenharia'],
    'Biológicas': ['Enfermagem', 'Medicina', 'Fisioterapia', 'Nutrição', 
                      'Farmácia', 'Educação Física', 'Biologia', 'Saúde',
                      'Odontologia', 'Veterinária', 'Biomedicina'],
    'Humanas': ['Psicologia', 'Pedagogia', 'Arquitetura', 'História', 'Arte',
                'Geografia', 'Letras', 'Marketing', 'Comunicação'],
    'Exatas': ['Contabilidade', 'Sistemas', 'Química', 'Economia', 'Agronomia', 'Logística', 
               'Matemática', 'Física', 'Tecnologia', 'Informação']
}

def get_course_classification(name):
    for macro_class, courses in course_map.items():
        for c in courses:
            if c.lower() in name.lower():
                return macro_class
            
    if name:
        return 'Outros'

students_area = students_raw.copy()

students_area.loc[:, 'course_area'] = students_area['CourseName'].apply(lambda x: get_course_classification(x))

students_area.groupby(['course_area', 'CourseName']).agg({'Id':'count'})

students_area = students_area.set_index('Id').loc[:, ['course_area']]

# + hidden=true
students_area['course_area'].value_counts(dropna=False)

# + [markdown] heading_collapsed=true
# ### University

# + hidden=true
# Exploring university distribution
df = students_raw\
        .groupby(['UniversityName'], as_index=False)\
        .agg({'Id':'count'})\
        .sort_values(by='Id', ascending=False)

df.columns = ['university', 'students']
df['cumul_students'] = df['students'].cumsum()
df['total_students'] = df['students'].sum()
df['rank'] = df['students'].rank(ascending=False)

df.loc[:, 'cumul_percentage'] = 100*df['cumul_students']/df['total_students']
df.loc[:, 'percentage'] = 100*df['students']/df['total_students']

fig = px.area(df, x='rank', y='cumul_percentage', hover_name='university', hover_data=['students', 'percentage'])
fig.show()

df.head(20)

# + hidden=true
top_20_uni = df.loc[df['rank'] <= 20, 'university'].tolist()

# + hidden=true
student_university = students_raw.copy()
student_university.loc[:, 'on_top_20_university'] = student_university['UniversityName']\
                                                        .apply(lambda x: x in top_20_uni)

student_university = student_university.set_index('Id').loc[:, ['on_top_20_university']]
student_university.head()

# + [markdown] heading_collapsed=true
# ### Region

# + hidden=true
states = pd.read_csv('./data/estados.csv')
regions = pd.read_csv('./data/regioes.csv')

states_infos = pd.merge(left=states, right=regions, left_on='Regiao', right_on='Id', suffixes=('_state', '_region'))

states_infos.loc[:, 'state_name'] = states_infos['Nome_state']\
                                        .str.lower()\
                                        .str.replace(' ', '_')\
                                        .str.normalize('NFKD')\
                                        .str.encode('ascii', errors='ignore')\
                                        .str.decode('utf-8')

states_infos.loc[:, 'region'] = states_infos['Nome_region']

state_region_dict = states_infos.set_index('state_name')['region'].to_dict()

state_region_dict

# + hidden=true
students_region = students_raw.copy()
students_region.loc[:, 'state_name'] = students_region['State']\
                                                .str.lower()\
                                                .str.replace(' ', '_')\
                                                .str.normalize('NFKD')\
                                                .str.encode('ascii', errors='ignore')\
                                                .str.decode('utf-8')\
                                                .fillna('')

students_region.loc[:, 'region'] = students_region['state_name'].apply(lambda x: state_region_dict[x] if x else np.nan)

students_region = students_region.set_index('Id').loc[:, ['region']]

# + hidden=true
students_region['region'].value_counts()
# -

# ### Payment

# +
plans = pd.read_csv('./data/prep/premium_plans_info.csv')

plans.head()

# +
not_cancelled_plans = plans.loc[plans['cancelled_at'].isnull()].copy()

not_cancelled_plans.loc[:, 'first'] = not_cancelled_plans.groupby(['student_id'])['created_at'].transform('min')
not_cancelled_plans.loc[:, 'revenue_first_plan'] = 0
not_cancelled_plans.loc[not_cancelled_plans['created_at'] == not_cancelled_plans['first'], 'revenue_first_plan'] = not_cancelled_plans['cost']


plans_per_user = not_cancelled_plans\
                    .groupby(['student_id'], as_index=True)\
                    .agg({'plan_id':'count', 'cost':'sum', 'created_at':['min', 'max'], 'revenue_first_plan':'sum'})

plans_per_user.columns = ['total_plans', 'ltv', 'first_purchase', 'last_purchase', 'revenue_first_purchase']
plans_per_user.head()

not_cancelled_plans
# -

# ### Final Dataset

students_raw.head()

# +
students_renamed = students_raw.copy().set_index('Id')
students_renamed.columns = ['signup_at', 'university_name', 'course_name', 'state', 'signup_source', 'city', 'user_client', 'user_origin', 'origin']

students = pd.concat([students_renamed, students_area, student_university, students_region, plans_per_user], axis=1)

students.loc[:, 'has_purchased'] = ~students['first_purchase'].isnull()

students.loc[:, 'on_top_20_university'] = students['on_top_20_university'].fillna(False)

students.index.name = 'student_id'
students.head()
# -

students['origin'].value_counts()

students.isnull().mean()

students.to_csv('./data/prep/user_infos.csv', index=True)


