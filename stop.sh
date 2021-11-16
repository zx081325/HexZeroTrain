PROCESS=`ps -ef|grep get_batch.py|grep -v grep|grep -v PPID|awk '{ print $2}'`
for i in $PROCESS
do
  echo "Kill the get_batch.py process [ $i ]"
  kill -9 $i
done


PROCESS=`ps -ef|grep train.py|grep -v grep|grep -v PPID|awk '{ print $2}'`
for i in $PROCESS
do
  echo "Kill the train.py process [ $i ]"
  kill -9 $i
done
