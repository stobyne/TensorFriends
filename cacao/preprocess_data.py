import numpy as np
import pandas as pd
import re

def preprocess_data(file_path, test_pct):

    cacao = pd.read_csv(file_path)

    original_colnames = cacao.columns
    new_colnames = ['company', 'species', 'REF', 'review_year', 'cocoa_p',
                    'company_location', 'rating', 'bean_typ', 'country']
    cacao = cacao.rename(columns=dict(zip(original_colnames, new_colnames)))
    ## And modify data types
    cacao['cocoa_p'] = cacao['cocoa_p'].str.replace('%','').astype(float)/100
    cacao['country'] = cacao['country'].fillna(cacao['species'])

    def txt_prep(text):
        replacements = [
            ['-', ', '], ['/ ', ', '], ['/', ', '], ['\(', ', '], [' and', ', '], [' &', ', '], ['\)', ''],
            ['Dom Rep|DR|Domin Rep|Dominican Rep,|Domincan Republic', 'Dominican Republic'],
            ['Mad,|Mad$', 'Madagascar, '],
            ['PNG', 'Papua New Guinea, '],
            ['Guat,|Guat$', 'Guatemala, '],
            ['Ven,|Ven$|Venez,|Venez$', 'Venezuela, '],
            ['Ecu,|Ecu$|Ecuad,|Ecuad$', 'Ecuador, '],
            ['Nic,|Nic$', 'Nicaragua, '],
            ['Cost Rica', 'Costa Rica'],
            ['Mex,|Mex$', 'Mexico, '],
            ['Jam,|Jam$', 'Jamaica, '],
            ['Haw,|Haw$', 'Hawaii, '],
            ['Gre,|Gre$', 'Grenada, '],
            ['Tri,|Tri$', 'Trinidad, '],
            ['C Am', 'Central America'],
            ['S America', 'South America'],
            [', $', ''], [',  ', ', '], [', ,', ', '], ['\xa0', ' '],[',\s+', ','],
            [' Bali', ',Bali']
        ]
        for i, j in replacements:
            text = re.sub(i, j, text)
        return text

    cacao['country'] = cacao['country'].str.replace('.', '').apply(txt_prep)

    cacao['company_location'] = cacao['company_location']\
        .str.replace('Amsterdam', 'Holland')\
        .str.replace('U.K.', 'England')\
        .str.replace('Niacragua', 'Nicaragua')\
        .str.replace('Domincan Republic', 'Dominican Republic')
    
    ## Let's define blend feature
    cacao['is_blend'] = np.where(
        np.logical_or(
            np.logical_or(cacao['species'].str.lower().str.contains(',|(blend)|;'),
                        cacao['country'].str.len() == 1),
            cacao['country'].str.lower().str.contains(',')
        )
        , 1
        , 0)
    
    cacao['is_domestic'] = np.where(cacao['country'] == cacao['company_location'], 1, 0)
    cacao = cacao.drop(columns=['bean_typ'])

    cacao = cacao.astype({'REF':object, 'review_year':object})

    cat_columns = ['company', 'species', 'REF', 'review_year', 'company_location', 'country', 'is_blend', 'is_domestic']
    num_columns = ['cocoa_p']
    one_hot_cat_df = pd.get_dummies(cacao[cat_columns])

    cacao_df = pd.concat([one_hot_cat_df, cacao[num_columns]], axis=1)
    y = cacao['rating'].values

    test_size = test_pct
    train_size = 1 - test_size
    split_row = int(train_size * cacao.shape[0])
    x_train = cacao_df.loc[0:split_row,:].values
    x_test = cacao_df.loc[split_row + 1:,:].values

    y_train = y[0:split_row]
    y_test = y[(split_row + 1):]

    return x_train, x_test, y_train, y_test
