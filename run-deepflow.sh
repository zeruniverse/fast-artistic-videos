if [ "$#" -ne 3 ]; then
  echo "This is an auxiliary script for makeOptFlow.sh. No need to call this script directly."
  exit 1
fi
if [ ! -f deepmatching-static ] && [ ! -f run_OF_RGB ]; then
  echo "Place deepflow2-static and deepmatching-static in this directory."
  exit 1
fi

./run_OF_RGB $1 $2 $3 5 1 35 35 0.05 0.95 0 8 0.80 0 1 0 1 10 10 5 4 3 1.6 2
