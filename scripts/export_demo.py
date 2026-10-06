"""Export aggregate data and the fitted gradient boosting model for local JS scoring."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from churn.pipeline import load_data,NUMERIC,CATEGORICAL
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'demo';out.mkdir(exist_ok=True)
a=joblib.load(ROOT/'models/churn.joblib');p=a['pipeline'];pre=p.named_steps['preprocess'];gb=p.named_steps['model']
num=pre.named_transformers_['numeric'];cat=pre.named_transformers_['categorical'];df=load_data(ROOT/'data/raw/Telco_customer_churn.xlsx')
data={'numeric':NUMERIC,'categorical':CATEGORICAL,'medians':num.named_steps['impute'].statistics_.tolist(),'indicators':num.named_steps['impute'].indicator_.features_.tolist(),'mean':num.named_steps['scale'].mean_.tolist(),'scale':num.named_steps['scale'].scale_.tolist(),'categories':[x.tolist() for x in cat.named_steps['encode'].categories_],'prior':float(gb.init_.class_prior_[1]),'learning_rate':gb.learning_rate,'threshold':a['threshold'],'trees':[]}
for est in gb.estimators_.ravel():
 t=est.tree_;data['trees'].append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'value':t.value[:,0,0].tolist()})
(out/'model.json').write_text(json.dumps(data,separators=(',',':')))
segments={}
for col in ['Contract','Internet Service','Payment Method','Tech Support','Online Security']:
 segments[col]=df.groupby(col)['Churn Value'].agg(customers='size',churned='sum',rate='mean').reset_index().rename(columns={col:'label'}).to_dict('records')
summary={'customers':len(df),'churned':int(df['Churn Value'].sum()),'rate':df['Churn Value'].mean(),'retainedMonthlyCharges':float(df.loc[df['Churn Value'].eq(0),'Monthly Charges'].sum()),'segments':segments,'metrics':json.loads((ROOT/'reports/metrics.json').read_text()),'comparison':__import__('pandas').read_csv(ROOT/'reports/model_comparison.csv').to_dict('records'),'importance':__import__('pandas').read_csv(ROOT/'reports/validation_permutation_importance.csv').to_dict('records')}
(out/'summary.json').write_text(json.dumps(summary,separators=(',',':')))
# A private parity fixture is never committed: only aggregate verification is public.
rows=df.sample(100,random_state=87)
fixture={'records':json.loads(rows[NUMERIC+CATEGORICAL].to_json(orient='records')),'expected':p.predict_proba(rows[NUMERIC+CATEGORICAL])[:,1].tolist()}
Path('/tmp/churn-parity.json').write_text(json.dumps(fixture))
print('Exported model and aggregate dashboard data')
