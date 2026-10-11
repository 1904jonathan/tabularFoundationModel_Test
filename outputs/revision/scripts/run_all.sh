#!/bin/bash
# Runs E1..E4 sequentially; each in its own process so a crash in one does not stop the others.
cd /c/CODE/tabularFoundationModel_Test/outputs/revision/scripts
PY="/c/Users/mehes/anaconda3/python.exe"
LOG=/c/CODE/tabularFoundationModel_Test/outputs/revision/log.txt
export PYTHONUNBUFFERED=1
for s in e1_multiseed e2_tuned e3_lsa_dim e4_20newsgroups; do
  echo "===== START $s $(date) =====" >> $LOG
  "$PY" $s.py >> $LOG 2>&1
  echo "===== END $s exit=$? $(date) =====" >> $LOG
done
echo "ALL_DONE" >> $LOG
