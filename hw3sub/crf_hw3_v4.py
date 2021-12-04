import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics


# def read_data_n_clean(data):
#     data.columns = ['Sentence','Word','Tag']
#     counter = 1
#     for i, row in data.iterrows():
#         if row['Sentence'] == 1:
#             data.at[i, 'Sentence #'] = 'Sentence: '+ str(counter)
#     #     else:
#     #         df_gen.at[i, 'Sentence#'] =  df_gen.at[i-1, 'Sentence#']
#         counter += 1
#     df = data[['Sentence #','Word','Tag']]
#     df = df.fillna(method='ffill')

#     return df

def read_data_n_clean_test(data):
    '''
    read and clean the test data
    '''
    data.columns = ['Sentence', 'Word']
    counter = 1
    for i, row in data.iterrows():
        if row['Sentence'] == 1:
            data.at[i, 'Sentence #'] = 'Sentence: ' + str(counter)
    #     else:
    #         df_gen.at[i, 'Sentence#'] =  df_gen.at[i-1, 'Sentence#']
        counter += 1
    df = data[['Sentence', 'Sentence #', 'Word']]
    df = df.fillna(method='ffill')

    return df


def word2features(sent, i):
    '''
    get the feather out of word for CRF learning #TODO: add more features
    '''
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def group_data(data):
    def agg_func(s): return [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                          s['POS'].values.tolist(),
                                                          s['Tag'].values.tolist())]
    grouped_data = data.groupby('Sentence #').apply(agg_func)

    return grouped_data


def get_ner(row):
    return row['prediction'][row['Sentence'] - 1]


def get_ner_skip(row):
    '''
     getting the prediction per word, and return O if empty
     '''
    if row['Sentence'] - 1 <= len(row['prediction']):
        return row['prediction'][row['Sentence'] - 1]
    else:
        return 'O'


def add_empty_rows(data):
    '''
    add one last empty row after each sentence
    '''
    df_gen_3 = data
    insert = pd.DataFrame(
        {'Sentence': [' '], 'Word': [' '], 'prediction_ner': [' '], 'ner': [' ']})
    marked = []
    for key, row in df_gen_3.iterrows():
        try:
            if df_gen_3.loc[key, 'Sentence'] < df_gen_3.loc[key-1, 'Sentence']:
                marked.append(key)
        except:
            pass

    for x in marked:
        df_gen_3 = pd.concat([df_gen_3.loc[:x-1], insert, df_gen_3.loc[x:]])
    # finally reset the index and spit out new df
    df = df_gen_3.reset_index(drop=True)
    return df


def test_data_crf_run(data):
    '''
    run the crf model on test set #TODO: clean it up
    '''
    df_gen_test_1 = data
    df_gen_1 = pd.DataFrame(df_gen_test_1.groupby(['Sentence #'])[
                            'Word'].apply(list)).reset_index()
    df_gen_1['word_string'] = df_gen_1['Word'].str.join(
        " ").apply(nltk.word_tokenize).apply(nltk.pos_tag)
    df_gen_1['features'] = df_gen_1['word_string'].apply(
        lambda row: [sent2features(row)])
    df_gen_1['prediction'] = df_gen_1['features'].apply(
        lambda row: list(crf.predict(row)[0]))

    df_gen_2 = df_gen_test_1.merge(
        df_gen_1[['Sentence #', 'prediction']], on='Sentence #', how='left')
    df_gen_2['prediction_ner'] = df_gen_2.apply(
        lambda row: get_ner_skip(row), axis=1)
    df_gen_2['prediction_ner'] = df_gen_2.apply(
        lambda row: get_ner_skip(row), axis=1)
    df_gen_2['ner'] = ', ' + df_gen_2['prediction_ner']

    df = df_gen_2[['Sentence', 'Word', 'prediction_ner', 'ner']]
    return df


# Part 1: start here
# df_gen = pd.read_csv('./pickled_pos.csv', low_memory=False)
df_gen = pd.read_csv('./delete_big_o.csv')

df_gen = df_gen[['Sentence #', 'Word', 'POS', 'Tag']
                ].dropna()  # TODO: what happend to pickle

group_df = group_data(df_gen)


split_ratio = .3
iterations = 100000

sentences = [s for s in group_df]
X = np.array([sent2features(s) for s in sentences],
             dtype=object)
y = np.array([sent2labels(s) for s in sentences],
             dtype=object)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split_ratio, random_state=42)
# X_train.shape, X_test.shape


crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                           c1=0.1,
                           c2=0.1,
                           max_iterations=iterations,
                           all_possible_transitions=True,
                           verbose=True)


crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)
labels = list(crf.classes_)

print(crf_metrics.flat_classification_report(y_test, y_pred, labels=labels))


# testset output #TODO: clean up
df_gen_test = pd.read_csv(
    'F21-gene-test.txt', header=None, delim_whitespace=True)
# clean
df_gen_test_1 = read_data_n_clean_test(df_gen_test)

# run crf
df_gen_3 = test_data_crf_run(df_gen_test_1)

# adding empty rows
df_gen_3 = add_empty_rows(df_gen_3)

# save to txt
df_gen_3[['Sentence', 'Word', 'ner']].to_csv(
    'test4_with_comma.txt', header=False, index=False, sep='\t', mode='a')
df_gen_3[['Sentence', 'Word', 'prediction_ner']].to_csv(
    'test4_without_comma.txt', header=False, index=False, sep='\t', mode='a')
