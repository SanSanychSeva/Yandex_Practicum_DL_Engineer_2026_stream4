'''
Module: dataset.py
        содержит функции для генерации DL-features и сборки 
        батчевого загрузчика тензов торча DataLoader
'''

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# NLP-features generation
#=================================================================

def get_text_embedding(words_list, w2v_model, emb_size=300):
    '''
    функция векторизует список слов, используя предобученную 
    по методу Word2Vec модель парно-семантического эмбеддинга слов.
    Свойством данного типа эмбеддинга является аддитивность смыслов
    складываемых векторов, что позволяет получить семантический эмбеддинг 
    для отдельных фраз.
    Список ингредиентов в нашем случае служит такой фразой, усредненный 
    вектор (центр смысла) которого и возвращает данная функция  

    Параметры:
        - words_list =  список слов-токенов векторизируемого текста. 
                        Предполагаются только широко-употребительные английские слова.
                        В случае незнакомого слова возвращает вектор нулей float32
        - w2v_model  =  модель семантического эмбеддинга слов, предоставляет функцию 
                        self.get_vestor(str), возвращающую вектор эмбеддинга английских слов
        - emb_size   =  размерность возвращаемого вектора эмбеддинга (default 300)
    '''
    vectors = []
    for w in words_list:
        try:
            vectors.append(w2v_model.get_vector(w))
        except:
            return np.zeros(emb_size, dtype=np.float32)
    
    return np.mean(vectors, axis=0)
#-----------------------------------------------------------------

# сборка данных для DataLoader
#=================================================================
def build_data_loaders(np2d_predictors_list, np1d_target, test_header_size, train_batch_size=128, test_batch_size=32):
    '''
    функция собирает загрузчик батчей тензоров торч для стандартного пайплайна обучения torch

    Параметры:
        - np2d_predictors_list  =   либо единичный массив, либо список двумерных массивов numpy, 
                                    представляющих векторизированные фичи объектов предсказания.
                                    NB!: объекты предварительно упорядочены - сперва идут тестовые
        - np1d_target           =   одномерный вектор numpy с целевой переменной - сперва тестовые
        - test_header_size      =   количество тестовых объектов в начале придикторов и целевой
                                    переменной - объекты расположены по нулевому индексу 
        - []_batch_size         =   размеры минибатчей для обучения и валидации - соответственно

    '''

    if type(np2d_predictors_list) is list:
        X_DL = np.concatenate(np2d_predictors_list, axis=1)
    else:
        X_DL = np2d_predictors_list

    X_test  = X_DL[:test_header_size]
    X_train = X_DL[test_header_size:]

    y_test  = np1d_target[:test_header_size]
    y_train = np1d_target[test_header_size:]

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_dataset  = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                                  torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)) 
    
    train_loader = DataLoader( train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader( test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader