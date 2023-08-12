from emo_data_splitter import build_paths


def test_build_paths():
    courpus_dir = '../src/joy'
    emotion = 'joy'
    csv_file, emo_label, train_data, test_data, train_labels, test_labels = build_paths(courpus_dir, emotion)
    
    assert csv_file == f'./{emotion}/vec_{emotion}.csv', "CSVファイルのパスが間違っています"
    assert emo_label == emotion, "感情のラベルが間違っています"
    assert train_data == f'{courpus_dir}/{emotion}/train_vectors.npy', "訓練データのパスが間違っています"
    assert train_labels == f'{courpus_dir}/{emotion}/train_labels.npy', "訓練ラベルのパスが間違っています"
    assert test_data == f'{courpus_dir}/{emotion}/test_vectors.npy', "テストデータのパスが間違っています"
    assert test_labels == f'{courpus_dir}/{emotion}/test_labels.npy', "テストラベルのパスが間違っています"
