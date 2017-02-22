# @Author: yancz1989
# @Date:   2017-02-22 16:57:08
# @Last Modified by:   yancz1989
# @Last Modified time: 2017-02-22 21:59:20

# TARGET=$1
# echo ${TARGET}
# mkdir ${TARGET}
# cp *.py ${TARGET}
# cp -r utils hypes doc ${TARGET}
# mkdir ${TARGET}/data
# mkdir ${TARGET}/data/sample
# mkdir ${TARGET}/data/fp
# cp -r data/output* ${TARGET}/data/
# cp -r data/meta ${TARGET}/data/
# cp -r data/trunk ${TARGET}/data/trunk
# cp data/inception* ${TARGET}/data/
# cd ${TARGET}
# python sampling.py
# for i in 0 1 2 3 4 5 6 7 8 9
# do
#   python scan.py gen subset${i}
# done

python augment.py
python generate_fp_data.py