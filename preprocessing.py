import sudachipy
from datasets import load_dataset
import neologdn
import re
import os

dataset = load_dataset("shunk031/wrime", name="ver1")
tokenizer = sudachipy.Dictionary().create()
emotions = ['joy','sadness','anticipation','surprise','anger','fear','disgust','trust']
TRAIN_NOM_ROWS = 40000
TEST_NOM_ROWS = 2000

def pull_out_sentence(nom_rows, mode, dataset, text_data):
    for i in range(nom_rows):
        tr = dataset[mode][i]
        text_data.append(tr['sentence'])
    return text_data

def normalize_neologdn(text_data):
    normalized_text_data = []
    for i in text_data:
        normalized_text_data.append(neologdn.normalize(i))
    return normalized_text_data

def replace_to_zero(text_data):
    zero_normalized_text_data = []
    for i in text_data:
        tmp = re.sub(r'(\d)([,.])(\d+)', r'\1\3', i)
        zero_normalized_text_data.append(re.sub(r'\d+', '0', tmp))
    return zero_normalized_text_data

def conversion_to_txt(path, text_data):
    with open(path, mode='w') as f:
        f.write('\n'.join(text_data))

def conversion_to_list(path):
    with open(path) as f:
        l_strip = [s.rstrip() for s in f.readlines()]
    return l_strip

def tokenize(sentence: str, mode: str):
    mode = {
        'A': sudachipy.Tokenizer.SplitMode.A,
        'B': sudachipy.Tokenizer.SplitMode.B,
        'C': sudachipy.Tokenizer.SplitMode.C}[mode]
    tokens = [m.normalized_form() for m in tokenizer.tokenize(sentence, mode)]
    return ' '.join(tokens)
def create_training_corpus(inputpath, outputpath):
    with open(inputpath) as inputfile, open(outputpath, 'w') as outputfile:
        for mode in ('A', 'B', 'C'):
            for line in inputfile:
                line = line.strip()
                if line == '':
                    continue
                outputfile.write(tokenize(line, mode) + '\n')
            inputfile.seek(0)

def remove_stop_words(wakati_list,stop_words):
    data_len = len(wakati_list)
    ans_list = []
    for i in range(data_len):
        tmp_list = []
        word_list = wakati_list[i].split()
        for word in word_list:
            if not(word in stop_words):
                tmp_list.append(word)
        ans_list.append(tmp_list)
    return ans_list

def pull_out_num_every_emotions():
    emotions_data_num = {}
    for emo in emotions:
        emotions_data_num[emo] = [] #初期値に空のリストを設定
        for i in range(TRAIN_NOM_ROWS):
            tr = dataset["train"][i]
            if tr['writer'][emo] > 0:
                emotions_data_num[emo] += [i,42000+i,84000+i]
        for i in range(TEST_NOM_ROWS):
            tr = dataset["test"][i]
            if tr['writer'][emo] > 0:
                emotions_data_num[emo] += [40000+i,82000+i,124000+i]
    return emotions_data_num

def classify_text_every_emotions(emotions_data_num, text_data):
    dir_name = "./datasets/corpus_emotion/"
    os.mkdir(dir_name)
    for k in emotions_data_num.keys():
        emotion_text_list = []
        for emo in emotions_data_num[k]:
            emotion_text_list.append(text_data[emo])
        path = f"{dir_name}{k}_corpus.txt"
        with open(path, mode='w') as f:
            for text in emotion_text_list:
                f.write(text+"\n")



if __name__ == '__main__':
    text_data = []
    text_data = pull_out_sentence(TRAIN_NOM_ROWS, "train", dataset, text_data)
    text_data = pull_out_sentence(TEST_NOM_ROWS, "test", dataset, text_data)

    # 1.neologdnで正規化
    text_data = normalize_neologdn(text_data)

    # 2.数字を0に置き換え
    text_data = replace_to_zero(text_data)

    #ひとまずtxtファイルに保存
    conversion_to_txt("./datasets/tmp_txt_datasets/chiVe_training_corpus.txt", text_data)

    #sudachiPyで分かち書きして保存
    create_training_corpus("./datasets/tmp_txt_datasets/chiVe_training_corpus.txt", "./tmp_txt_datasets/sudachi_wakati_corpus.txt")


    # 3.ストップワードの除去
    #ストップワードの読み込み
    stop_words = conversion_to_list("./datasets/tmp_txt_datasets/Japanese.txt")

    #分かち書きしたファイルをリストで読み込み
    text_data = conversion_to_list("./datasets/tmp_txt_datasets/sudachi_wakati_corpus.txt")

    #ストップワード除去を実行
    text_data = remove_stop_words(text_data, stop_words)

    #最後に保存
    with open("./datasets/tmp_txt_datasets/last_text_corpus.txt", mode='w') as f:
        for i in text_data:
            f.write(' '.join(i)+"\n")


    #各感情ごとに文章の番号を抜き出す
    emotions_data_num = pull_out_num_every_emotions()

    text_data = conversion_to_list("./datasets/tmp_txt_datasets/last_text_corpus.txt")

    #感情ごとにファイルを分ける
    classify_text_every_emotions(emotions_data_num, text_data)