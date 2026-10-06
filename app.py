"""Run with: streamlit run app.py"""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from churn.pipeline import load_data, FEATURES

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Telco Retention Lab',layout='wide')
st.title('Telco Retention Lab')
st.caption('Educational snapshot analysis • historical labels • no live monitoring')
file=st.file_uploader('Upload the supplied Telco Excel dataset',type=['xlsx'])
path=file or ROOT/'data/raw/Telco_customer_churn.xlsx'
if not file and not path.exists():
    st.info('Place the mentor dataset in data/raw/ or upload it here.'); st.stop()
try:
    df=load_data(path)
except ValueError as exc:
    st.error(str(exc));st.stop()
contracts=st.sidebar.multiselect('Contract',sorted(df.Contract.unique()),default=sorted(df.Contract.unique()))
filtered=df[df.Contract.isin(contracts)]
if filtered.empty: st.info('Select at least one contract.');st.stop()
a,b,c=st.columns(3)
a.metric('Customers',f'{len(filtered):,}')
b.metric('Historical churn rate',f'{filtered["Churn Value"].mean():.1%}')
c.metric('Monthly charges: retained customers',f'{filtered.loc[filtered["Churn Value"].eq(0),"Monthly Charges"].sum():,.2f}')
st.caption('Charges are reported in source units; this total is a snapshot billing proxy, not recognized revenue or a time trend.')
for col in ['Contract','Internet Service','Payment Method']:
    grouped=filtered.groupby(col)['Churn Value'].agg(['size','mean']).reset_index()
    st.plotly_chart(px.bar(grouped,x=col,y='mean',hover_data=['size'],title=f'Historical churn rate by {col}',labels={'mean':'Churn rate'}),use_container_width=True)
st.plotly_chart(px.histogram(filtered,x='Tenure Months',color='Churn Label',barmode='overlay',nbins=24),use_container_width=True)
if (ROOT/'reports/metrics.json').exists():
    results=json.loads((ROOT/'reports/metrics.json').read_text())
    st.subheader('Frozen holdout evaluation')
    st.json({k:results[k] for k in ['selected_model','threshold','test']})
    st.dataframe(pd.read_csv(ROOT/'reports/model_comparison.csv'),hide_index=True)
    st.subheader('Validation permutation importance')
    st.dataframe(pd.read_csv(ROOT/'reports/validation_permutation_importance.csv'),hide_index=True)
    st.caption('Importance describes predictive association; it does not establish that changing a feature will prevent churn.')
st.subheader('Score an unlabelled batch')
st.caption('Only upload customer records you are permitted to process. Required predictor columns: '+', '.join(FEATURES))
batch=st.file_uploader('Predictor CSV',type=['csv'])
if batch:
    import joblib
    model_path=ROOT/'models/churn.joblib'
    if not model_path.exists(): st.warning('Run training first.')
    else:
        x=pd.read_csv(batch)
        missing=set(FEATURES)-set(x.columns)
        if missing: st.error(f'Missing predictors: {sorted(missing)}')
        else:
            for col in ['Tenure Months','Monthly Charges','Total Charges']:
                x[col]=pd.to_numeric(x[col],errors='coerce')
            artifact=joblib.load(model_path) # Only our locally trained artifact is loaded.
            p=artifact['pipeline'].predict_proba(x[FEATURES])[:,1]
            output=pd.DataFrame({'row':range(len(x)),'churn_probability':p,'review_flag':p>=artifact['threshold']})
            st.dataframe(output.head(20));st.download_button('Download scores',output.to_csv(index=False),'churn_scores.csv','text/csv')
            st.warning('The future churn horizon and decision-time feature availability are unverified. These scores are for demonstration, not automatic customer actions.')
