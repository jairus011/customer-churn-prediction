"""Reproducible educational churn experiment; no outcome-derived predictors."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
 recall_score, f1_score, accuracy_score, brier_score_loss, confusion_matrix,
 ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)

NUMERIC = ['Tenure Months', 'Monthly Charges', 'Total Charges']
CATEGORICAL = ['Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
 'Internet Service', 'Online Security', 'Online Backup', 'Device Protection',
 'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract',
 'Paperless Billing', 'Payment Method']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'Churn Value'
SEED = 42

def load_data(path):
    df = pd.read_excel(path)
    missing = set(FEATURES + [TARGET, 'CustomerID']) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if df.CustomerID.isna().any() or df.CustomerID.duplicated().any():
        raise ValueError('Customer IDs must be present and unique')
    if df[TARGET].isna().any() or not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError('Target must contain only 0 and 1')
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if (df[col].dropna() < 0).any():
            raise ValueError(f'Negative values in {col}')
    return df

def preprocessing():
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('numeric', numeric, NUMERIC), ('categorical', categorical, CATEGORICAL)])

def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return dict(average_precision=float(average_precision_score(y,p)),
        roc_auc=float(roc_auc_score(y,p)), accuracy=float(accuracy_score(y,pred)),
        precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred,zero_division=0)),
        f1=float(f1_score(y,pred,zero_division=0)), brier=float(brier_score_loss(y,p)),
        confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())

def run(path, output):
    root = Path(output); reports = root/'reports'; figures = reports/'figures'
    figures.mkdir(parents=True,exist_ok=True); (root/'models').mkdir(exist_ok=True)
    df = load_data(path)
    train_val, test = train_test_split(df, test_size=.2, stratify=df[TARGET], random_state=SEED)
    train, val = train_test_split(train_val, test_size=.25, stratify=train_val[TARGET],random_state=SEED)
    pd.concat([part[['CustomerID']].assign(split=name) for name,part in [('train',train),('validation',val),('test',test)]]).to_csv(reports/'split_manifest.csv',index=False)
    audit = dict(rows=len(df), columns=len(df.columns), churned=int(df[TARGET].sum()),
        churn_rate=float(df[TARGET].mean()), numeric_missing=df[NUMERIC].isna().sum().to_dict(),
        split_sizes={k:len(v) for k,v in [('train',train),('validation',val),('test',test)]})
    (reports/'data_audit.json').write_text(json.dumps(audit,indent=2))
    # Explore training data only before model selection.
    segments=[]
    for col in ['Contract','Internet Service','Payment Method']:
        table=train.groupby(col)[TARGET].agg(['size','mean']).reset_index()
        table.columns=['segment','customers','churn_rate'];table.insert(0,'dimension',col)
        segments.append(table)
        ax=table.plot.bar(x='segment',y='churn_rate',legend=False,color='#287e9b',title=f'Training-set churn rate by {col}')
        ax.set_ylabel('Churn rate'); ax.set_ylim(0,1);plt.tight_layout()
        plt.savefig(figures/f'{col.lower().replace(" ","_")}.png',dpi=160);plt.close()
    pd.concat(segments).to_csv(reports/'training_segments.csv',index=False)
    candidates={'dummy':DummyClassifier(strategy='prior'),
        'logistic':LogisticRegression(max_iter=2000,C=1),
        'random_forest':RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=10,n_jobs=2,random_state=SEED),
        'gradient_boosting':GradientBoostingClassifier(n_estimators=100,max_depth=2,random_state=SEED)}
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    rows=[]; fitted={}
    for name,model in candidates.items():
        pipe=Pipeline([('preprocess',preprocessing()),('model',model)])
        scores=cross_val_score(pipe,train[FEATURES],train[TARGET],cv=cv,scoring='average_precision',n_jobs=1)
        rows.append(dict(model=name,cv_average_precision=float(scores.mean()),cv_std=float(scores.std())))
        pipe.fit(train[FEATURES],train[TARGET]);fitted[name]=pipe
    comparison=pd.DataFrame(rows).sort_values('cv_average_precision',ascending=False)
    comparison.to_csv(reports/'model_comparison.csv',index=False)
    winner=comparison.iloc[0]['model']; model=fitted[winner]
    vp=model.predict_proba(val[FEATURES])[:,1]
    thresholds=np.arange(.05,.96,.01)
    # F1 is a transparent academic objective; real intervention costs are unknown.
    threshold=float(max(thresholds,key=lambda t:f1_score(val[TARGET],vp>=t,zero_division=0)))
    pd.DataFrame([dict(threshold=float(t),f1=float(f1_score(val[TARGET],vp>=t,zero_division=0)),recall=float(recall_score(val[TARGET],vp>=t)),precision=float(precision_score(val[TARGET],vp>=t,zero_division=0))) for t in thresholds]).to_csv(reports/'validation_thresholds.csv',index=False)
    importance=permutation_importance(model,val[FEATURES],val[TARGET],scoring='average_precision',n_repeats=5,random_state=SEED,n_jobs=1)
    pd.DataFrame(dict(feature=FEATURES,importance_mean=importance.importances_mean,importance_std=importance.importances_std)).sort_values('importance_mean',ascending=False).to_csv(reports/'validation_permutation_importance.csv',index=False)
    # Model and threshold are frozen before this one final test evaluation.
    tp=model.predict_proba(test[FEATURES])[:,1]
    result=dict(selected_model=winner,threshold=threshold,threshold_objective='validation F1',
        validation=metrics(val[TARGET],vp,threshold),test=metrics(test[TARGET],tp,threshold),
        features=FEATURES,seed=SEED)
    (reports/'metrics.json').write_text(json.dumps(result,indent=2))
    predictions=test[['CustomerID',TARGET]].copy();predictions['churn_probability']=tp
    predictions['predicted_churn']=(tp>=threshold).astype(int)
    predictions.to_csv(reports/'test_predictions.csv',index=False)
    for column in ['Gender','Senior Citizen']:
        groups=[]
        for group,part in test.groupby(column):
            p=model.predict_proba(part[FEATURES])[:,1]
            groups.append(dict(group=group,n=len(part),positives=int(part[TARGET].sum()),recall=float(recall_score(part[TARGET],p>=threshold,zero_division=0)),precision=float(precision_score(part[TARGET],p>=threshold,zero_division=0))))
        pd.DataFrame(groups).to_csv(reports/f'subgroup_{column.lower().replace(" ","_")}.csv',index=False)
    ConfusionMatrixDisplay.from_predictions(test[TARGET],tp>=threshold);plt.tight_layout();plt.savefig(figures/'test_confusion_matrix.png',dpi=160);plt.close()
    RocCurveDisplay.from_predictions(test[TARGET],tp);plt.savefig(figures/'test_roc.png',dpi=160);plt.close()
    PrecisionRecallDisplay.from_predictions(test[TARGET],tp);plt.savefig(figures/'test_precision_recall.png',dpi=160);plt.close()
    joblib.dump(dict(pipeline=model,threshold=threshold,features=FEATURES),root/'models/churn.joblib')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data/raw/Telco_customer_churn.xlsx');parser.add_argument('--output',default='.')
    args=parser.parse_args();run(args.data,args.output)
