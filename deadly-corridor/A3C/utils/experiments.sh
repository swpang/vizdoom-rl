conda activate vizdoom
which python
nvidia-smi

python main.py --name 1102 --scenario deadly_corridor --num_processes -1 --actions all --max_episodes 3000 --no_render
python main.py --name 1102 --scenario deadly_corridor --num_processes -1 --actions all --max_episodes 3000 --use_curiosity --no_render
python main.py --name 1102 --scenario deadly_corridor --num_processes -1 --actions all --max_episodes 3000 --use_ppo --no_render
python main.py --name 1102 --scenario deadly_corridor --num_processes -1 --actions all --max_episodes 3000 --use_ppo --use_curiosity --no_render