# @Author: yancz1989
# @Date:   2017-01-15 01:37:50
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-01-18 00:24:10

for i in 0 1 2 3 4 5 6 7 8 9
do
python train.py --hypes hypes/luna.json --gpu 0 --logdir data/output --subset $i
python evaluate.py --gpu 0 --subset $i --left 84000 --right 120010 --sep 12000
done
# python train.py --hypes hypes/luna.json --gpu 0 --logdir data/output --subset 1
# python evaluate.py --gpu 0 --subset 1 --left 84000 --right 120010 --sep 12000
# python train.py --hypes hypes/luna.json --gpu 0 --logdir data/output --subset 2
# python evaluate.py --gpu 0 --subset 2 --left 84000 --right 120010 --sep 12000
# python train.py --hypes hypes/luna.json --gpu 0 --logdir data/output --subset 3
# python evaluate.py --gpu 0 --subset 3 --left 84000 --right 120010 --sep 12000
# python train.py --hypes hypes/luna.json --gpu 0 --logdir data/output --subset 4
# python evaluate.py --gpu 0 --subset 4 --left 84000 --right 120010 --sep 12000
# python train.py --hypes hypes/luna.json --gpu 0 --logdir data/output --subset 5
# python evaluate.py --gpu 0 --subset 5 --left 84000 --right 120010 --sep 12000
# python train.py --hypes hypes/luna.json --gpu 0 --logdir data/output --subset 6
# python evaluate.py --gpu 0 --subset 6 --left 84000 --right 120010 --sep 12000
# python train.py --hypes hypes/luna.json --gpu 0 --logdir data/output --subset 7
# python evaluate.py --gpu 0 --subset 7 --left 84000 --right 120010 --sep 12000
# python train.py --hypes hypes/luna.json --gpu 0 --logdir data/output --subset 8
# python evaluate.py --gpu 0 --subset 8
# python train.py --hypes hypes/luna.json --gpu 0 --logdir data/output --subset 9
# python evaluate.py --gpu 0 --subset 9