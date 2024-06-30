input_file=${1:-"none"}
num_turns=${2:-2}
device=${3:-"0"}
model_path=${4:-"meta-llama/Meta-Llama-3-8B-Instruct"}
tensor_parallel=1
gpu_memory_utilization=0.95
batch_size=128

if [ $input_file == "none" ]; then
    echo "[magpie.sh] Input file not provided!"
    exit 1
fi
if [ ! -f $input_file ]; then
    echo "[magpie.sh] Input file not found!"
    exit 1
fi

# get job path from input file
job_path=$(dirname "$input_file")
exec > >(tee -a "$job_path/tagging.log") 2>&1
echo "[magpie.sh] Job Path: $job_path"
echo "[magpie.sh] Input File: $input_file"
echo "[magpie.sh] Num Turns: $num_turns"
echo "[magpie.sh] Model Name: $model_path"
echo "[magpie.sh] System Config: device=$device, batch_size=$batch_size, tensor_parallel=$tensor_parallel"

echo "[magpie.sh] Start Generating Multi-turn Conversations..."
CUDA_VISIBLE_DEVICES=$device python ../exp/gen_mt.py \
    --device $device \
    --model_path $model_path \
    --input_file $input_file \
    --num_turns $num_turns \
    --tensor_parallel $tensor_parallel \
    --gpu_memory_utilization $gpu_memory_utilization \
    --batch_size $batch_size \

echo "[magpie.sh] Finish Generating Multi-turn Conversations!"