import numpy as np
import pandas as pd
import re

def preprocess_data(file_path, test_pct):

    choko = pd.read_csv(file_path)

    original_colnames = choko.columns
    new_colnames = ['company', 'species', 'REF', 'review_year', 'cocoa_p',
                    'company_location', 'rating', 'bean_typ', 'country']
    choko = choko.rename(columns=dict(zip(original_colnames, new_colnames)))
    ## And modify data types
    choko['cocoa_p'] = choko['cocoa_p'].str.replace('%','').astype(float)/100
    choko['country'] = choko['country'].fillna(choko['species'])

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

    choko['country'] = choko['country'].str.replace('.', '').apply(txt_prep)

    choko['company_location'] = choko['company_location']\
        .str.replace('Amsterdam', 'Holland')\
        .str.replace('U.K.', 'England')\
        .str.replace('Niacragua', 'Nicaragua')\
        .str.replace('Domincan Republic', 'Dominican Republic')
    
    ## Let's define blend feature
    choko['is_blend'] = np.where(
        np.logical_or(
            np.logical_or(choko['species'].str.lower().str.contains(',|(blend)|;'),
                        choko['country'].str.len() == 1),
            choko['country'].str.lower().str.contains(',')
        )
        , 1
        , 0)
    
    choko['is_domestic'] = np.where(choko['country'] == choko['company_location'], 1, 0)
    choko = choko.drop(columns=['bean_typ'])

    label = ['rating']
    y = choko[label]

    choko = choko.astype({'REF':object, 'review_year':object})

    cat_columns = ['company', 'species', 'REF', 'review_year', 'company_location', 'country', 'is_blend', 'is_domestic']
    num_columns = ['cocoa_p']
    one_hot_cat_df = pd.get_dummies(choko[cat_columns])

    choko = pd.concat([one_hot_cat_df, choko[num_columns]], axis=1)

    test_size = test_pct
    train_size = 1 - test_size
    split_row = int(train_size * choko.shape[0])
    x_train = choko.loc[0:split_row,:]
    x_test = choko.loc[split_row + 1:,:]

    y_train = y.loc[0:split_row,:]
    y_test = y.loc[split_row + 1,:]

    return x_train, x_test, y_train, y_test
