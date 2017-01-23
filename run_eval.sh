ANNOT_DIR="./annotations/"
RESULT_DIR="./exampleFiles/submission/"
ANNOT_FILE="data/trunk/CSVFILES/annotations_eval.csv"
ANNOT_EX_FILE="data/trunk/CSVFILES/annotations_excluded.csv"
OUTPUT_DIR="test/"

RESULT_FILE=$1
KEY_FILE=test/keys.csv

echo ${RESULT_FILE}
echo ${KEY_FILE}

python keys.py ${RESULT_FILE}

python noduleCADEvaluationLUNA16.py \
  ${ANNOT_FILE} \
  ${ANNOT_EX_FILE} \
  ${KEY_FILE} \
  ${RESULT_FILE} \
  ${OUTPUT_DIR}

cat test/CADAnalysis.txt
