# @Author: yancz1989
# @Date:   2017-01-15 01:37:50
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-01-15 12:00:47
if [ "$1" == "train" ]; then
  echo "Training"
  python train.py --hypes hypes/lstm_rezoom_luna.json --gpu 1 --logdir data/output --subset 0
  python train.py --hypes hypes/lstm_rezoom_luna.json --gpu 1 --logdir data/output --subset 1
  python train.py --hypes hypes/lstm_rezoom_luna.json --gpu 1 --logdir data/output --subset 2
  python train.py --hypes hypes/lstm_rezoom_luna.json --gpu 1 --logdir data/output --subset 3
  python train.py --hypes hypes/lstm_rezoom_luna.json --gpu 1 --logdir data/output --subset 4
  python train.py --hypes hypes/lstm_rezoom_luna.json --gpu 1 --logdir data/output --subset 5
  python train.py --hypes hypes/lstm_rezoom_luna.json --gpu 1 --logdir data/output --subset 6
  python train.py --hypes hypes/lstm_rezoom_luna.json --gpu 1 --logdir data/output --subset 7
  python train.py --hypes hypes/lstm_rezoom_luna.json --gpu 1 --logdir data/output --subset 8
  python train.py --hypes hypes/lstm_rezoom_luna.json --gpu 1 --logdir data/output --subset 9
else
  echo "Evaluate"
  python evaluate.py --gpu 3 --subset 0
  python evaluate.py --gpu 3 --subset 1
  python evaluate.py --gpu 3 --subset 2
  python evaluate.py --gpu 3 --subset 3
  python evaluate.py --gpu 3 --subset 4
  python evaluate.py --gpu 3 --subset 5
  python evaluate.py --gpu 3 --subset 6
  python evaluate.py --gpu 3 --subset 7
  python evaluate.py --gpu 3 --subset 8
  python evaluate.py --gpu 3 --subset 9
fi