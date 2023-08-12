import numpy as np
from thundersvm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

def load_data(train_vectors_path, train_labels_path, test_vectors_path, test_labels_path):
    """
    データを読み込む関数

    Args:
        train_vectors_path (str): 訓練データのベクトルのファイルパス
        train_labels_path (str): 訓練データのラベルのファイルパス
        test_vectors_path (str): テストデータのベクトルのファイルパス
        test_labels_path (str): テストデータのラベルのファイルパス

    Returns:
        tuple: 訓練データ、訓練ラベル、テストデータ、テストラベルのタプル
    """
    train_data = np.load(train_vectors_path)
    train_labels = np.load(train_labels_path)
    test_data = np.load(test_vectors_path)
    test_labels = np.load(test_labels_path)
    return train_data, train_labels, test_data, test_labels

def main():
    """
    メイン関数
    """
    # データの読み込み
    train_data, train_labels, test_data, test_labels = load_data(
        train_vectors_path='datasets/感情/train_vectors.npy',
        train_labels_path='datasets/感情/train_labels.npy',
        test_vectors_path='datasets/感情/test_vectors.npy',
        test_labels_path='datasets/感情/test_labels.npy'
    )

    # ハイパーパラメータの探索範囲
    param_grid = {
        'C': [0.1, 1, 5],
        'gamma': [0.1, 1, 10],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    # ランダムサーチの設定
    random_search = RandomizedSearchCV(
        estimator=SVC(),
        param_distributions=param_grid,
        n_iter=20, # 試行回数
        random_state=42, # 乱数シードを固定
        n_jobs=-1, # 全てのcpuを利用
        verbose=2  # 処理の詳細なログ表示
    )

    # ランダムサーチの実行
    random_search.fit(train_data, train_labels)

    # 最適なモデルとハイパーパラメータの取得
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # グリッドサーチの設定
    param_grid = {
        # 前後0.5の範囲で10分割
        'C': np.linspace(max(0.1, best_params['C'] - 0.3), best_params['C'] + 0.3, 10), 
        # 対数スケールの範囲で10分割
        'gamma': np.logspace(np.log10(max(0.01, best_params['gamma'] / 2)),
                            np.log10(best_params['gamma'] * 2), 10),  
        # ランダムサーチで見つけた最適なカーネルを固定
        'kernel': [best_params['kernel']]  
    }

    # グリッドサーチの実行
    grid_search = GridSearchCV(
        estimator=SVC(),
        param_grid=param_grid,
        scoring='accuracy',  # スコアの評価指標を指定
        cv=15,  # StratifiedKFold K=15
        n_jobs=-1  
    )

    # グリッドサーチの実行
    grid_search.fit(train_data, train_labels)

    # 最適なモデルとハイパーパラメータの取得
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # テストデータでの予測
    pred_labels = best_model.predict(test_data)

    # テストデータでモデルを評価
    test_score = best_model.score(test_data, test_labels)

    # 混合行列の作成
    cm = confusion_matrix(test_labels, pred_labels)

    # 結果の表示
    print("Best Model:", best_model)
    print("Best Parameters (Random Search):", best_params)
    print("Best Parameters (Grid Search):", best_params)
    print("Test Score (Grid Search):", test_score)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()
