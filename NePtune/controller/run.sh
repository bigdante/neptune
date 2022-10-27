for i in {0..$1}
do
tmux new -d -s "n${i}" "bash controller/run_single_controller.sh ${i}"
done
