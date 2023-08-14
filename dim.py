from sklearn.decomposition import PCA
import numpy as np

emos = ['disgust','fear','sadness','surprise']

for emo in emos:
    train_data = f'./{emo}/train_vectors_{emo}.npy'
    test_data = f'./{emo}/test_vectors_{emo}.npy'
    
    
    # PCAのインスタンス化
    pca = PCA(n_components=110)
    vectorized_train_data = np.load(train_data)
    vectorized_test_data = np.load(test_data)


    # トレーニングデータにPCAを適用
    pca.fit(vectorized_train_data)

    # トレーニングデータとテストデータをPCAで変換
    pca_train_data = pca.transform(vectorized_train_data)
    pca_test_data = pca.transform(vectorized_test_data)

    # PCA適用後のデータの次元数を取得
    new_dim = pca_train_data.shape[1]

    # 寄与率の取得
    explained_variance_ratio = pca.explained_variance_ratio_

    print("New dimension after PCA:", new_dim)

    # 寄与率の表示
    print("寄与率:", explained_variance_ratio)

    # 全ての寄与率の合計を計算
    total_variance_ratio = sum(explained_variance_ratio)

    print("全ての寄与率の合計:", total_variance_ratio)

    np.save(f'./{emo}/pca_{emo}_train_data.npy', pca_train_data)
    np.save(f'./{emo}/pca_{emo}_test_data.npy', pca_test_data)
    
