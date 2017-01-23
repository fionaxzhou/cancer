# @Author: yancz1989
# @Date:   2017-01-15 01:37:50
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-01-23 01:32:36

# for i in 0 1 2 3 4 5 6 7 8 9
# do
#   python scan.py gen subset$i
# done

for i in 0 1 2 3 4 5 6 7 8 9
do
# python train.py --hypes hypes/luna.json --gpu 0 --logdir data/output --subset $i
python evaluate.py --gpu 0 --subset $i --left 84000 --right 120010 --sep 12000
done