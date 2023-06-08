# DRL_SPS_FightingICE

まず以下のリンクからFightingICEをダウンロードしてください。
https://www.ice.ci.ritsumei.ac.jp/~ftgaic/Downloadfiles/FTG4.50.zip

ファイルを解答後、Dueling_DDQNHRA.pyが入っているフォルダを上記でダウンロードしたFTG4.50フォルダに移動させます。
（FTG4.50/Gym-FightingICE_RL_spring/Dueling_DDQNHRA.pyとなるように）

DRL_SPS_FightingICE内にymlファイルがあるので、Minicondaを使って環境構築を行った後、FTG4.50をメインディレクトリとした状態でpython DRL_SPS_FightingICE/Dueling_DDQNHRA.pyを実行するとHRA Dueling DDQN AIの学習が始まります。ymlファイル内のPytorchのverは、お使いの機器に合わせるよう、内容を書き換えてください


Dueling_DDQNHRA.pyコード内のモデルの保存場所を自環境のフォルダに設定してから実行してください。

学習が完了すると以下のP1のエージェントができあがります。

<object width="425" height="350">
  <param name="movie" value="[https://youtu.be/0fH4kWfbQaY]" />
  <param name="wmode" value="transparent" />
  <embed src="http://www.youtube.com/user/wwwLoveWatercom?v=BTRN1YETpyg"
         type="application/x-shockwave-flash"
         wmode="transparent" width="425" height="350" />
</object>
