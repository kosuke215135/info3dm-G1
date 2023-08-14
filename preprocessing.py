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
    """
    datasetからmodeに応じたテキストを抜き出してリストに格納する関数

    Args:
        nom_rows (int): 抜き出したいデータの数。ここでは`TRAIN_NOM_ROWS`か`TEST_NOM_ROWS`のいずれかを引数に取る。
        mode (str): データの種類。ここでは`train`か`test`のいずれかの文字列を引数に取る。
        dataset (datasets.dataset_dict.DatasetDict): データセット
        text_data (list[str]): このlistにデータセットから抜き出したテキストを1文ずつ追加していく。
    Returns:
        text_data (list[str]): テキストデータを追加し終わったlistを返す。
    """
    for i in range(nom_rows):
        tr = dataset[mode][i]
        text_data.append(tr['sentence'])
    return text_data


def normalize_neologdn(text_data):
    """
    neologdn(https://github.com/ikegami-yukino/neologdn)を用いてlistに格納されているテキストの正規化を行う。

    Args:
        text_data (list[str]): テキストが1文ずつ格納されたlist
    Returns:
        normalized_text_data (list[str]): 正規化されたテキストを格納したlist
    """
    normalized_text_data = []
    for i in text_data:
        normalized_text_data.append(neologdn.normalize(i))
    return normalized_text_data


def replace_to_zero(text_data):
    """
    正規表現を用いて、テキストに含まれる数字を全て0に置き換える。

    Args:
        text_data (list[str]): テキストが1文ずつ格納されたlist
    Returns:
        zero_normalized_text_data (list[str]): 数字を全て0で置き換えたテキストが1文ずつ格納されているlist
    """
    zero_normalized_text_data = []
    for i in text_data:
        tmp = re.sub(r'(\d)([,.])(\d+)', r'\1\3', i)
        zero_normalized_text_data.append(re.sub(r'\d+', '0', tmp))
    return zero_normalized_text_data


def conversion_to_txt(path, text_data):
    """
    listに格納されているテキストをtxtファイルに書き込む。

    Args:
        path (str): 保存するtxtファイルのパス
        text_data (list[str]): テキストが1文ずつ格納されたlist
    """
    with open(path, mode='w') as f:
        f.write('\n'.join(text_data))


def conversion_to_list(path):
    """
    txtファイルに書かれているテキストを行ごとにlistへ格納する。

    Args:
        path (str): txtファイルのパス
    Returns:
        l_strip (list[str]): テキストが1文ずつ格納されたlist。改行コードは取り除かれている。
    """
    with open(path) as f:
        l_strip = [s.rstrip() for s in f.readlines()]
    return l_strip


def tokenize(sentence: str, mode: str):
    """
    モードに応じた辞書でテキストを分かち書きする。
    以下URLを参考にした。
    https://github.com/WorksApplications/chiVe/blob/master/docs/continue-training.md

    Args:
        sentence (str): 分かち書きの対象となる文章。
        mode (str): 辞書の種類。ここでは`A`, `B`, `C`のいずれかを引数に取る。
    Returns:
        split_tokens (str): 分かち書きした文章を半角スペースで結合したテキスト
    """
    mode = {
        'A': sudachipy.Tokenizer.SplitMode.A,
        'B': sudachipy.Tokenizer.SplitMode.B,
        'C': sudachipy.Tokenizer.SplitMode.C}[mode]
    tokens = [m.normalized_form() for m in tokenizer.tokenize(sentence, mode)]
    split_tokens = ' '.join(tokens)
    return split_tokens
def create_training_corpus(inputpath, outputpath):
    """
    分かち書き後の文章が書き込まれたtxtファイルを作成する。
    以下URLを参考にした。
    https://github.com/WorksApplications/chiVe/blob/master/docs/continue-training.md

    Args:
        inputpath (str): 1行に1文が書かれたtxtファイルのパス
        outputpath (str): 分かち書きされた文章を書き込むためのtxtファイルのパス
    """
    with open(inputpath) as inputfile, open(outputpath, 'w') as outputfile:
        for mode in ('A', 'B', 'C'):
            for line in inputfile:
                line = line.strip()
                if line == '':
                    continue
                outputfile.write(tokenize(line, mode) + '\n')
            inputfile.seek(0)


def remove_stop_words(wakati_list,stop_words):
    """
    テキストからストップワードを除去する。

    Args:
        wakati_list (list[str]): 分かち書きされた文章が格納されているlist
        stop_words (list[str]): ストップワードが格納されているlist
    Returns:
        ans_list (list[str]): ストップワードが除去された文章が格納されたlist
    """
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
    """
    データセットより、各感情のスコアが1以上の文章のデータセット内での番号を抜き出す。

    Returns:
        emotions_data_num (dict{str:list[int]}): 各感情のスコアが1以上の文章のデータセット内での番号を格納しているdict
    """
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
    """
    分かち書きされた文章を感情ごとに分けてtxtファイルに書き込む。

    Args:
        emotions_data_num (dict{str:list[int]}): 各感情のスコアが1以上の文章のデータセット内での番号を格納しているdict
        text_data (list[str]): 分かち書きされた文章(半角スペースで単語を結合したもの)が1文ずつ格納されているlist
    """
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