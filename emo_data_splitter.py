import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def emo_clf(csv_file, emo_label, train_data, test_data, train_labels, test_labels):
    """
    指定したCSVファイルからデータを読み取り，訓練データとテストデータに分割して保存
    
    :param csv_file: 読み取り対象のCSVファイルパス
    :param emo_label: 抽出する感情のラベル
    :param train_data: 訓練データの保存先
    :param test_data: テストデータの保存先
    :param train_labels: 訓練ラベルの保存先
    :param test_labels: テストラベルの保存先
    """
    df = pd.read_csv(csv_file)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    # ベクトルデータとラベルの抽出
    for vector_data, writer_emo, save_data, save_labels in [(df_train, emo_label, train_data, train_labels), (df_test, emo_label, test_data, test_labels)]:
        vector_data_values = vector_data.iloc[:, :300].values
        writer_emo_values = vector_data[writer_emo].values
        np.save(save_data, vector_data_values)
        np.save(save_labels, writer_emo_values)

def build_paths(courpus_dir, emotion):
    """
    指定した感情に対するファイルパスを構築
    
    :param courpus_dir: コーパスファイルがあるディレクトリのパス
    :param emotion: 感情の種類（例: 'joy', 'anger'など）
    :return: 各ファイルのパス
    """
    base_path = f'{courpus_dir}/{emotion}'
    csv_file = f'./{emotion}/vec_{emotion}.csv'
    train_data = f'{base_path}/train_vectors.npy'
    train_labels = f'{base_path}/train_labels.npy'
    test_data = f'{base_path}/test_vectors.npy'
    test_labels = f'{base_path}/test_labels.npy'
    return csv_file, emotion, train_data, test_data, train_labels, test_labels

def main():
    """
    メインの実行関数
    定義された感情のリストに対してデータの分割と保存を実行
    """
    courpus_dir = 'datasets'
    emotions = ['joy', 'anger', 'anticipation', 'trust']
    for emotion in emotions:
        csv_file, emo_label, train_data, test_data, train_labels, test_labels = build_paths(courpus_dir, emotion)
        emo_clf(csv_file, f'writer_{emo_label}', train_data, test_data, train_labels, test_labels)

if __name__ == "__main__":
    main()
