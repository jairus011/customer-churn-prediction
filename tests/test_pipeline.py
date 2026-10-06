import numpy as np
import pandas as pd
from churn.pipeline import FEATURES, NUMERIC, CATEGORICAL, preprocessing, metrics

def test_outcome_fields_never_enter_predictors():
    forbidden={'Churn Value','Churn Label','Churn Score','Churn Reason','CLTV','CustomerID'}
    assert forbidden.isdisjoint(FEATURES)

def test_transform_accepts_missing_numeric_and_unseen_category():
    train=pd.DataFrame({c:[1.,2.,3.] for c in NUMERIC}|{c:['Yes','No','Yes'] for c in CATEGORICAL})
    pre=preprocessing();pre.fit(train)
    new=train.iloc[[0]].copy();new['Total Charges']=np.nan;new['Contract']='unseen'
    result=pre.transform(new)
    assert result.shape[0]==1 and np.isfinite(result).all()

def test_metric_confusion_uses_zero_then_one():
    result=metrics([0,0,1,1],np.array([.1,.7,.2,.9]),.5)
    assert result['confusion_matrix']==[[1,1],[1,1]]
