end=$(($1 + $2))
for i in $(seq $1 $end)
do
tmux new -s "n$(($i - $1))" "bash controller/run_single_controller.sh ${i} && exec bash"
done
