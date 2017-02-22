ANNOT_DIR="./annotations/"
RESULT_DIR="./exampleFiles/submission/"
ANNOT_FILE="data/trunk/CSVFILES/annotations_eval.csv"
ANNOT_EX_FILE="data/trunk/CSVFILES/annotations_excluded.csv"
OUTPUT_DIR="test/"

# RESULT_FILE=$1


KEY_FILE=test/keys.csv

# echo ${RESULT_FILE}
# echo ${KEY_FILE}

for i in 0
do
  for epoch in 108000
  do
    RESULT_FILE=data/output.detection/subset${i}/${epoch}.csv
    python keys.py ${RESULT_FILE}

    python noduleCADEvaluationLUNA16.py \
      ${ANNOT_FILE} \
      ${ANNOT_EX_FILE} \
      ${KEY_FILE} \
      ${RESULT_FILE} \
      ${OUTPUT_DIR}
    cat test/CADAnalysis.txt
    cat test/CADAnalysis.txt >> test/subset${i}_${epoch}.txt
    cp test/froc_csv_${epoch}.png test/subset${i}_${epoch}.png
  done
done
