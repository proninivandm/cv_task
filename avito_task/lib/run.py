import numpy as np 
import pandas as pd 
import catboost
import re
import pickle
words = [r'\bviber', r'\bwhatsap', r'\bв[оа]тсап', r'\bвибер', r'\bвайбер', r'\bтелег', r'\btelegram', r'\bзвонит', r'\bсвяз', r'\btg\W', r'\bтг\W']
if __name__ == '__main__':
    test = pd.read_csv('/task-for-hiring-data/test_data.csv')
    test.description=test.description.str.lower()
    test.description = test.description.apply(lambda x: re.sub(r'\b(([a-z])+\.(([a-z])+)/(\S*))|(@[\w_]{5,32})\b', ' NOTTEL ', x))
    test.description = test.description.apply(lambda x: re.sub(r'\b\+?[78]*[-\( ]?\d{3}[\) ]?[- ]?\d{3}[- ]?\d{2}[- ]?\d{2}\b', ' TELNUM ', x))
    for regu in words:
        test.description=test.description.apply(lambda x: re.sub(regu, ' ANYWORD ', x))
    test.description = test.description.apply(lambda x: re.sub(r'\d?(\D?|\s*|[А-Яа-я]*)(\d(\D?|\s*|[А-Яа-я]*)){10}(\D?|\s*|[А-Яа-я]*)', ' MAYBETEL ', x))
    
    test=test.drop(['price','datetime_submitted'], axis=1)
    pool = catboost.Pool(test, cat_features = test.drop(['title', 'description'], axis=1).columns, text_features=['title', 'description'])
    
    with open('lib/mmodel_last-2.pickle', 'rb') as f:
        model1=pickle.load(f)
    
    target_prediction = pd.DataFrame()
    target_prediction['index'] = range(test.shape[0])
    target_prediction['prediction'] = model1.predict_proba(pool)[:,1]

    target_prediction.to_csv('/task-for-hiring-data/target_prediction.csv', index=False)
