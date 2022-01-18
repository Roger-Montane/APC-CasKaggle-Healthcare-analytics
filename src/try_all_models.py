import import_libraries as il
import load_datasets as ld

il.import_libraries()

def select_K_best(X_data, y_data, k):
    # Create and fit selector
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X_data, y_data)
    # Get columns to keep and create new dataframe with those only
    cols = selector.get_support(indices=True)
    data_aux = X_data.iloc[:,cols]
    
    return data_aux

def labelEncode(enc_data, data, test=False):
    label_encoder = preprocessing.LabelEncoder()

    enc_data['Type of Admission'] = label_encoder.fit_transform(data['Type of Admission'])
    enc_data['Severity of Illness'] = label_encoder.fit_transform(data['Severity of Illness'])
    enc_data['Age'] = label_encoder.fit_transform(data['Age'])
    enc_data['Ward_Type'] = label_encoder.fit_transform(data['Ward_Type'])
    enc_data['Hospital_type_code'] = label_encoder.fit_transform(data['Hospital_type_code'])
    if not test:
        enc_data['Stay'] = label_encoder.fit_transform(data['Stay'])
    
    return enc_data

def oneHotEncode(enc_data, data, cat_cols):
    for label in cat_cols[:5]:
        one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, drop = 'first', categories = 'auto')
        one_hot_encoder.fit(data[cat_cols[:5]])
        one_hot_df = one_hot_encoder.transform(data[cat_cols[:5]])
    
        for i in range(one_hot_df.shape[1]):
            enc_data[label + '_' + str(i)] = one_hot_df[:,i]
            
    # Drop unnecessary columns
    enc_data.drop(['Hospital_region_code', 'Department', 'Ward_Facility_Code']
           , axis = 1, inplace = True)
            
    return enc_data

def standarize(data, st_num_cols):
    sc = StandardScaler()

    sc.fit(data[st_num_cols])
    data[st_num_cols] = sc.transform(data[st_num_cols])
    
    return data

# Or we could create a function to get X,y (preprocessed thorugh the process above) from a given dataset
def get_X_y(data, cat_cols, num_cols, st_num_cols):
    data = data.dropna()
    data.drop(['case_id', 'patientid', 'Hospital_code', 'City_Code_Hospital', 'City_Code_Patient', 'Visitors with Patient']
               , axis = 1, inplace = True)
    enc_data = data.copy(deep=True)
    enc_data = labelEncode(enc_data, data)
    enc_data = oneHotEncode(enc_data, data, cat_cols)
    enc_data = standarize(enc_data, st_num_cols)
    
    return enc_data.loc[:, enc_data.columns != 'Stay'], enc_data['Stay']

def model_compare(X, y, models, names, times=10, pctg=0.5, all_results=True, verbose=False, quick=False):
    training_score = dict.fromkeys(names, 0)
    validation_score = dict.fromkeys(names, 0)
    f1score = dict.fromkeys(names, 0)
    exec_time = dict.fromkeys(names, 0)
    
    if quick:
        size = int(np.ceil(X.shape[0] * pctg))
        indices = [randint(0, X.shape[0]-1) for _ in range(size)]
        X = X.iloc[indices]
        y = y.iloc[indices]

    for i in range(times):
        if verbose:
            print("Iteration {}".format(i))
        X_train, X_val, y_train, y_val = train_test_split(X, y)
        for model, title in zip(models, names):
            if verbose:
                print("    -> Running model {}".format(title))
            
            start_time = time.time()
            model.fit(X_train, y_train)
            y_preds = model.predict(X_val)
            training_score[title] += model.score(X_train, y_train)
            validation_score[title] += model.score(X_val, y_val)
            f1score[title] += f1_score(y_preds, y_val, average='micro')
            end_time = time.time()
            exec_time[title] += end_time - start_time
            
        if verbose:
            print("-----------------------------------------------")

    training_score = {k: v / times for k, v in training_score.items()}
    validation_score = {k: v / times for k, v in validation_score.items()}
    f1score = {k: v / times for k, v in f1score.items()}
    exec_time = {k: v / times for k, v in exec_time.items()}
    
    best_model = max(validation_score, key=validation_score.get)
    
    print('\nBest model: ', best_model)
    print('Training score: ', training_score[best_model])
    print('Validation score: ', validation_score[best_model])
    print('f1score: ', f1score[best_model])
    print('Execution time', exec_time[best_model])
    
    data = {'Model':[], 'Training score':[], 'Validation score':[], 'f1 score':[], 'Execution time (s)':[]}
    if all_results:
        for title in names:
            data['Model'].append(title)
            data['Training score'].append(training_score[title])
            data['Validation score'].append(validation_score[title])
            data['f1 score'].append(f1score[title])
            data['Execution time (s)'].append(exec_time[title])
        df = pd.DataFrame(data)
        print(tabulate(df, headers = 'keys', tablefmt = 'psql', showindex=False))



def main():
    classifiers = [
        KNeighborsClassifier(n_neighbors=35, algorithm='auto'),
        DecisionTreeClassifier(splitter='best', criterion='gini', max_depth=None),
        RandomForestClassifier(n_estimators=20),
        CatBoostClassifier(verbose=False),
        XGBClassifier(),
        lgb.LGBMClassifier(),
        OneVsRestClassifier(RandomForestClassifier(n_estimators=20)),
        OneVsOneClassifier(DecisionTreeClassifier(splitter='best', criterion='gini', max_depth=None))
    ]

    names = [
        "K Nearest Neighbors",
        "Decision Tree",
        "Random Forest",
        "Cat boost",
        "XGB",
        "LGBM",
        "One-vs-Rest with Random forest",
        "One-vs-One with Decision tree"
    ]

    # Load train data
    train = ld.load_dataset('../data/train_data.csv')
    train_data = train.copy(deep=True)
    print(train_data.shape)

    # Load test data
    test = ld.load_dataset('../data/test_data.csv')
    test_data = test.copy(deep=True)
    print(test_data.shape)

    cat_cols=[]
    num_cols=[]

    for col in train_data.columns:
        if train_data[col].dtypes == 'object':
            cat_cols.append(col)
            
    for col in train_data.columns:
        if train_data[col].dtypes != 'object':
            num_cols.append(col)

    st_num_cols = ['Available Extra Rooms in Hospital', 'Bed Grade', 'Admission_Deposit', 'Hospital_type_code', 'Ward_Type']

    X, y = get_X_y(train, cat_cols, num_cols, st_num_cols)
    print(X.shape, y.shape)

    sm = SMOTE()
    # For time complexity purposes, we will trim our dataset to less than 100k samples
    size = int(np.ceil(X.shape[0] * 0.3))
    indices = [randint(0, X.shape[0]-1) for _ in range(size)]
    os_X, os_y = sm.fit_resample(X.iloc[indices], y.iloc[indices])
    os_X_best_k, _ = sm.fit_resample(X_best_k.iloc[indices], y.iloc[indices])

    model_compare(os_X, os_y, classifiers, names, times=3, pctg=0.4, verbose=True, quick=True)

if __name__ == "__main__":
    main()
