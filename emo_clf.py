import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split
import tqdm
'''
データを訓練用とテスト用に分割して
それぞれのベクトルと指定したラベルをnumpy配列にしてファイルに保存する

以下の4つのファイルが指定したパスに保存される
・訓練用ベクトルデータ
・訓練用のラベル
・テスト用ベクトルデータ
・テスト用のラベル

このコードの実行はローカルでも学科サーバー上でもどっちで実行しても良い.
ローカルの方がおすすめだと思う
'''

def emo_clf(csv_file,emo_label,train_data,test_data,train_labels,test_labels):
    # データの読み込み
    df = pd.read_csv(csv_file)

    # データを訓練データとテストデータに分割
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # 訓練データからベクトルデータと 'writer_joy' を抽出
    train_vector_data = df_train.iloc[:, :300].values
    train_writer_emo = df_train[emo_label].values

    # numpy配列を保存
    np.save(train_data, train_vector_data)
    np.save(train_labels, train_writer_emo)

    # テストデータからベクトルデータと 'writer_joy' を抽出
    test_vector_data = df_test.iloc[:, :300].values
    test_writer_emo = df_test[emo_label].values

    # numpy配列を保存
    np.save(test_data, test_vector_data)
    np.save(test_labels, test_writer_emo)
    print(f'{emo}')
    
emos = ['disgust','fear','sadness','surprise']

for emo in emos:
    emo_clf(
    #csvファイル
    csv_file = f'./{emo}/vec_{emo}.csv',
    #取り出すラベル
    emo_label = f'writer_{emo}',
    #訓練用のベクトルデータとラベル
    train_data = f'./{emo}/train_vectors_{emo}.npy',
    train_labels = f'./{emo}/train_labels_{emo}.npy',
    # テスト用のベクトルデータとラベル
    test_data = f'./{emo}/test_vectors_{emo}.npy',
    test_labels = f'./{emo}/test_labels_{emo}.npy',
    )
