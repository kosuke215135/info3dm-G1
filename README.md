# info3dm-G1
## 環境構築
1. [こちらのページ](https://docs.conda.io/en/latest/miniconda.html)から自分の環境に合わせてMinicondaをインストールしてください
2. condaのbase環境に入ってください
3. `conda env create -f experiment.yaml`を実行し必要なライブラリをインストールしてください。
4. `./setup.sh`を実行し、必要なファイルや事前学習済みモデルをダウンロードしてください。

## プログラムの実行手順
1. `python preprocessing.py`
   - データセットの前処理を行います。
2. `python additional_learning.py 感情`
   - Word2Vecモデルの追加学習を行います。
   - 感情には` joy, sadness, anticipation, surprise, anger, fear, disgust, trust`の8つの感情のうち1つを選んで実行してください。
   - 追加学習済みモデルを新たに出力させるため、1感情ごとに10GBの空き容量が必要になります。
3. `python corpus_to_vector.py 感情`
   - 追加学習モデルを用いて、コーパスをベクトルに変換します。
   - 感情には` joy, sadness, anticipation, surprise, anger, fear, disgust, trust`の8つの感情のうち1つを選んで実行してください。
   - 実行の際には、各感情に対応した追加学習済みモデルが必要になります。
4. `python emo_data_splitter.py `
   - コーパスから訓練データとテストデータを抜き出します。
5. `python svm_param_search.py`
   - ハイパーパラメーターの探索を行います。


# ThunderSVMの実行環境構築
ThunderSVMは基本GPUを使いSVMの処理速度を改善するが、CPUだけを使ってでも実行することができる。
しかしながら、今回の実験ではGPUを使う場合のみ説明する。

## 1.  学科のサーバーから使う場合
sbatchを用いて学科のサーバーから登録順に実行するようにする
*name.sbatch
などのファイルを作成し、以下のようなコードを書く
```

#!/bin/bash
#SBATCH --job-name *job-name
#SBATCH --output ./log/%x-%j.log
#SBATCH --error ./log/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1

source ~/miniconda3/bin/activate
conda activate *env-name

#ここに実行したいコマンドを書く
python3 *name.py
```

## 2. sbatchをサーバーに登録

```
sbatch *name.sbatch
```
をすることで学科サーバーから実行される。

## 3. 実行確認

```
squeue
```

を入力することで実行確認ができる。

## 4. 実行結果アラーム

実行が完了したことをmattermostを用いて利用者に知らせる

   ### 4-1. mattermostの総合機能から内向きウェブフックを選択。<br>
   ### 4-2. タイトル、チャンネル、ユーザー名などを書く。＃今回の場合、チャンネル＝info3,4dm、ユーザー名=testとしている。<br>
   ### 4-3. 取得したURLを以下のコードに入れる
```
curl -X POST -H 'Content-Type: application/json' -d '{"text": "*text", "channel": "*@e2X57XX"}' https://mattermost.ie.u-ryukyu.ac.jp/hooks/*xxxxxxxxxxxxx
```
’*text、*@e2X57XX、https://mattermost.ie.u-ryukyu.ac.jp/hooks/*xxxxxxxxxxxxx’は変更すること。

   ### 4-4. 4-3から作成したコードを*name.sbatchの語尾に書き込む。

最終的な*name.sbatchの中身を確認
```
#!/bin/bash
#SBATCH --job-name *job-name
#SBATCH --output ./log/%x-%j.log
#SBATCH --error ./log/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1

source ~/miniconda3/bin/activate
conda activate *env-name

#ここに実行したいコマンドを書く
python3 *name.py
curl -X POST -H 'Content-Type: application/json' -d '{"text": "*text", "channel": "*@e2X57XX"}' https://mattermost.ie.u-ryukyu.ac.jp/hooks/*xxxxxxxxxxxxx
```