'''
Module: utils.py
        содержит функции для обучения моделей и визуализации результатов
        (чтобы не перегружать портянками основной юпитер-ноутбук) 
'''
import numpy as np
import copy
import matplotlib.pyplot as plt
from PIL import Image

# цикл обучения и валидации моделей
#=================================================================
def train(model, criterion, optimizer, train_loader, test_loader, nof_epochs=401):
    '''
    функция реализует обучение модели минибатчами в несколько эпох и
    возвращает историю обучения, метрику лучшей модели на валидации и саму модель
    '''
    
    epochs_history = []
    best_MAE = 250
    if nof_epochs < 21:
        print('WARNING: для сохранения истории обучения кол-во эпох должно быть >> 21')

    for epoch in range(nof_epochs):
        running_train_loss = 0.0
        running_val_loss = 0.0

        # train loop for one epoch
        model.train()
        counter = 0
        for X_batch, y_batch in train_loader:
        
            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss
            counter += 1
    
        avg_train_loss = (running_train_loss / counter).item()

        # validation for one epoch
        model.eval()
        counter = 0
        for X_batch, y_batch in test_loader:
        
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            running_val_loss += loss
            counter += 1
    
        avg_val_loss = (running_val_loss / counter).item()

        # оценка лучшей модели
        if best_MAE > avg_val_loss:
            best_MAE = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())   # сохраняем веса наилучшей модели в памяти

        # сохранение истории и вывод прогресса обучения
        save_freq = nof_epochs // 20
        if epoch % save_freq == 0:                            # не слишком часто!
            epochs_history.append( [epoch, avg_train_loss, avg_val_loss])
            print('Epoch No', str(epoch).rjust(3), 
                  ' |  Train MAE =', round(avg_train_loss,2), 
                  ' |  Validation MAE =', round(avg_val_loss,2))
        
    print('='*70)
    print('Best MAE reached =', best_MAE)
    model.load_state_dict(best_model_state)                # восстанавливаем модель наилучшей эпохи

    return epochs_history, best_MAE, model
#-----------------------------------------------------------------

# визуализация истории обучения моделей
#=================================================================
def two_models_history_graph(epochs_history_dl, epochs_history_ml, best_MAE_dl, best_MAE_ml, target_MAE=50):
    '''
    функция строит сравнительные графики по историям обучения двух моделей
    '''

    loss_history_dl = np.array(epochs_history_dl)
    loss_history_ml = np.array(epochs_history_ml)

    plt.plot(loss_history_dl[:,0], loss_history_dl[:,1], marker='.', color='blue', ls='--', label='DL Train MAE')
    plt.plot(loss_history_dl[:,0], loss_history_dl[:,2], marker='.', color='blue', label='DL Validation MAE')
    plt.axhline(best_MAE_dl, color='blue', ls=':', label='DL best val MAE = '+ str(round(best_MAE_dl,1)))

    plt.plot(loss_history_ml[:,0], loss_history_ml[:,1], marker='.', color='green', ls='--', label='ML Train MAE')
    plt.plot(loss_history_ml[:,0], loss_history_ml[:,2], marker='.', color='green', label='ML Validation MAE')
    plt.axhline(best_MAE_ml, color='green', ls=':', label='ML best val MAE = '+ str(round(best_MAE_ml,1)))

    plt.axhline(target_MAE, color='red', lw=1, label='target val MAE = '+str(target_MAE))

    plt.ylim(0,100)
    plt.grid()
    plt.legend()
    plt.title('MLP regressors train history on table data (ML) and DL-features')
    plt.xlabel('Epoch No')
    plt.ylabel('Loss = MAE')
    plt.show()

    return None
#-----------------------------------------------------------------

def one_model_history_graph(epochs_history, best_MAE, target_MAE=50):
    '''
    функция выводит на график историю обучения модели
    '''

    loss_history = np.array(epochs_history)

    plt.plot(loss_history[:,0], loss_history[:,1], marker='.', label='Train MAE')
    plt.plot(loss_history[:,0], loss_history[:,2], marker='.', label='Validation MAE')
    plt.axhline(best_MAE, color='green', ls=':', label='best val MAE = '+ str(round(best_MAE,1)))

    plt.axhline(target_MAE, color='red', lw=1, label='target val MAE = '+str(target_MAE))

    plt.grid()
    plt.legend()
    plt.title('MLP regressor train history on table data (ML)')
    plt.xlabel('Epoch No')
    plt.ylabel('Loss = MAE')
    plt.show()

    return None
#-----------------------------------------------------------------

# визуализация инференса моделей
#=================================================================
def show_top_six(df, end):
    '''
    функция рисует красиво объекты с самыми лучшими или самыми худшими инференсами
    '''
    
    if end == 'head':
        ddf = df.head(6)
        grade = ' точные '
    elif end == 'tail':
        ddf = df.tail(6)
        grade = ' НЕточные '
    else:
        print('FYI: please, use only values "head" or "tail" for the parameter "end"') 

    fig, axes = plt.subplots(2,3, figsize=(11,8), layout='constrained')
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(Image.open(ddf.iloc[i].path_to_photo).convert('RGB'))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('baseline = ' + str(round(ddf.iloc[i].baseline_pred, 2)) + 
                     ' | multimodal = ' + str(round(ddf.iloc[i].multimodal_pred, 2)))
        ax.set_xlabel(split_long(ddf.iloc[i].descr, 50))
        ax.legend(loc='lower left', 
                  title='вес = '+str(round(ddf.iloc[i].total_mass, 2))+'\nкалорийность = '+str(
                                     round(ddf.iloc[i].total_calories, 2)))

    fig.suptitle('самые' + grade + 'предсказания\n')
    plt.show()

    return None
#-----------------------------------------------------------------

def split_long(text: str, max_len: int) -> str:
    
    words = text.split()
    
    if not words:
        return text

    result = [words[0]]
    current_len = len(words[0])

    for word in words[1:]:
        # +1 за пробел
        if current_len + 1 + len(word) > max_len:
            result.append("\n")
            result.append(word)
            current_len = len(word)
        else:
            result.append(" ")
            result.append(word)
            current_len += 1 + len(word)

    return "".join(result)