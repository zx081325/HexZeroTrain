PROCESS=`ps -ef|grep get_batch.py|grep -v grep|grep -v PPID|awk '{ print $2}'`
for i in $PROCESS
do
  echo "Kill the get_batch.py process [ $i ]"
  kill -9 $i
done

sleep 1

PROCESS=`ps -ef|grep train.py|grep -v grep|grep -v PPID|awk '{ print $2}'`
for i in $PROCESS
do
  echo "Kill the train.py process [ $i ]"
  kill -9 $i
done

echo "start PROCESS"

sleep 1

nohup python3 get_batch.py > batch.out 2>&1 &

sleep 1

nohup python3 -u train.py > train.out 2>&1 &

