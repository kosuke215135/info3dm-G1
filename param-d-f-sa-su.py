import pandas as pd
import numpy as np
'''
パラメータサーチを実行するコード
このコードとemo_clf.pyで出力された4つのファイルをrsyncしてこのコードを実行する
'''
def load_data(train_vectors_path, train_labels_path, test_vectors_path, test_labels_path):
    train_data = []
    train_labels = [] 
    test_data = []
    test_labels = []

    # numpy配列を読み込む
    train_vectors = np.load(train_vectors_path)
    train_labels_array = np.load(train_labels_path)

    test_vectors = np.load(test_vectors_path)
    test_labels_array = np.load(test_labels_path)

    # 訓練データのベクトルデータとラベルをリストに追加
    for i in range(len(train_vectors)):
        train_data.append(train_vectors[i])
        train_labels.append(train_labels_array[i])

    # テストデータのベクトルデータとラベルをリストに追加
    for i in range(len(test_vectors)):
        test_data.append(test_vectors[i])
        test_labels.append(test_labels_array[i])

    return train_data, train_labels, test_data, test_labels


emos = ['disgust','fear','sadness','surprise']

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import expon

for emo in emos:
    print(f'emo = {emo}')
    train_data, train_labels, test_data, test_labels = load_data(
    train_vectors_path = f'./{emo}/train_vectors_{emo}.npy',
    train_labels_path = f'./{emo}/train_labels_{emo}.npy',
    test_vectors_path = f'./{emo}/test_vectors_{emo}.npy',
    test_labels_path = f'./{emo}/test_labels_{emo}.npy'
    )
    
    param_grid = {
    'C': np.linspace(max(0.1, best_params['C']-0.3), best_params['C']+0.3, 10),  # 例えば、ベストパラメータの前後1を探索
    'gamma': np.logspace(np.log10(max(0.01, best_params['gamma']/2)), np.log10(best_params['gamma']*2), 10),# 例えば、ベストパラメータの1/10から10倍の範囲を探索
    'kernel':['rbf','sigmoid']
    }

    # ランダムサーチの設定
    random_search = RandomizedSearchCV(
        estimator=SVC(),
        param_distributions=param_grid,
        random_state=1,
        n_iter=6,
        verbose=1,
        n_jobs=-1,
    )

    # ランダムサーチの実行
    random_search.fit(train_data, train_labels)

    # 最適なモデルとハイパーパラメータの取得
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    print("Best Model:", best_model)
    print("Best Parameters:", best_params)
    print("Best Score:", random_search.best_score_)
    print("Grid Scores:", random_search.cv_results_['mean_test_score'])
    
    
    # テストデータでの予測
    pred_labels = best_model.predict(test_data)

    # テストデータでモデルを評価
    test_score = best_model.score(test_data, test_labels)

    # テストデータでモデルを評価
    test_score = best_model.score(test_data, test_labels)

    print("Test Score:", test_score)

    from sklearn.metrics import confusion_matrix

    # 混合行列の作成
    cm = confusion_matrix(test_labels, pred_labels)

    # 混合行列の表示
    print("Confusion Matrix:")
    print(cm)