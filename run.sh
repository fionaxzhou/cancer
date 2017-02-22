# @Author: yancz1989
# @Date:   2017-01-15 01:37:50
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-02-22 01:23:43

# for i in 0 1 2 3 4 5 6 7 8 9
# do
#   python scan.py gen subset$i
# done

gpu=$1
shift
args="$@"

# for i in $args
# do
# python tensorbox.py --hypes hypes/luna.json --gpu $gpu --logdir data/output.detection --subset $i
# python evaluate.py --gpu $gpu --subset $i --left 84000 --right 120010 --sep 12000
# done
for i in $args
do
  python fp.py --hype hypes/fp.json --gpu $gpu --subset $i
done