#!/usr/bin/env bash

npw=$(ps aux | grep ProcessWorker.py | grep -vw grep | wc -l)
echo "" && echo "There are ${npw} videos being processed now."

if [[ ${npw} -eq 0 ]]
then
echo ""
echo "To check if all videos have been processed, you can restart the "
echo "analysis in the Batch Processing Multiple Files GUI, selecting"
echo "the tickbox marked 'Only Display Progress Summary'."
echo ""
echo "Tierpsy will only scan the directories and report on the progress status."
echo ""
fi
