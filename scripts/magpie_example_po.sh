model_path=${1:-"meta-llama/Meta-Llama-3-8B-Instruct"}
input_file=${2:-"../data_po/example_instructions.jsonl"}
num_samples=${3:-5}
res_topp=${4:-1}
res_temp=${5:-0.8}
res_rep=1
device="0"
tensor_parallel=1
gpu_memory_utilization=0.95
n=200
batch_size=200

# Get Current Time
timestamp=$(date +%s)

# Generate Pretty Name
job_name="${input_file##*/}_topp${ins_topp}_temp${ins_temp}_PO"

### Setup Logging
log_dir="data_po"
if [ ! -d "../${log_dir}" ]; then
    mkdir -p "../${log_dir}"
fi

exec > >(tee -a "$job_path/${job_name}.log") 2>&1
echo "[magpie.sh] Model Name: $model_path"
echo "[magpie.sh] Response Generation Config: temp=$res_temp, top_p=$res_topp, rep=$res_rep"
echo "[magpie.sh] System Config: device=$device, n=$n, batch_size=$batch_size, tensor_parallel=$tensor_parallel"
echo "[magpie.sh] Timestamp: $timestamp"
echo "[magpie.sh] Job Name: $job_name"

echo "[magpie.sh] Start Generating Responses..."
CUDA_VISIBLE_DEVICES=$device python ../exp/gen_po_multi_res.py \
    --device $device \
    --input_file $input_file \
    --model_path $model_path \
    --num_samples $num_samples \
    --batch_size $batch_size \
    --top_p $res_topp \
    --temperature $res_temp \
    --repetition_penalty $res_rep \
    --tensor_parallel $tensor_parallel \
    --gpu_memory_utilization $gpu_memory_utilization \
    --offline

CUDA_VISIBLE_DEVICES=$device python ../exp/gen_po_rewards.py \
    --device $device \
    --input_file "${input_file%.jsonl}_${num_samples}res.json" \

echo "[magpie.sh] Finish Generating Responses!"