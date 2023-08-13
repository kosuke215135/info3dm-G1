from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os
import sys

if __name__ == '__main__':
    args = sys.argv
    k = args[1]
    path = f"./datasets/corpus_emotion/{k}_corpus.txt"
    emotion_text_LineSentence = LineSentence(path)
    model = Word2Vec.load('./chive-1.2-mc5_gensim-full/chive-1.2-mc5.bin')
    model.min_count = 3
    model.build_vocab(emotion_text_LineSentence, update=True)
    model.train(emotion_text_LineSentence, total_examples=model.corpus_count, epochs=15)
    save_folder_path = f"./datasets/emotion_word2vec/{k}/"
    os.makedirs(save_folder_path)
    model.wv.save(f"{save_folder_path}/chive-1.2-mc5.finetuned-mc3.kv")  # Save as KeyedVectors
    model.save(f"{save_folder_path}/chive-1.2-mc5.finetuned-mc3.bin")    # Save as Full model