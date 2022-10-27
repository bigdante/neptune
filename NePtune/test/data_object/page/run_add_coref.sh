for i in $(seq 0 $1)
do
#tmux new -d -s "n${i}" "python3 -m test.data_object.page.add_coref_to_page ${i} && bash -l"
tmux kill-session -t "n${i}"
done
