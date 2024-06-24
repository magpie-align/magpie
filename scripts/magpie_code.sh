model_path=${1:-"meta-llama/Meta-Llama-3-8B-Instruct"}
total_prompts=${2:-1000}
ins_topp=${3:-1}
ins_temp=${4:-1}
res_topp=${5:-1}
res_temp=${6:-0}
res_rep=1
device="0"
tensor_parallel=1
gpu_memory_utilization=0.95
n=200
batch_size=200

# Get Current Time
timestamp=$(date +%s)

# Generate Pretty Name
job_name="${model_path##*/}_topp${ins_topp}_temp${ins_temp}_${timestamp}"

### Setup Logging
log_dir="data"
if [ ! -d "../${log_dir}" ]; then
    mkdir -p "../${log_dir}"
fi

job_path="../${log_dir}/${job_name}"

mkdir -p $job_path
exec > >(tee -a "$job_path/${job_name}.log") 2>&1
echo "[magpie.sh] Model Name: $model_path"
echo "[magpie.sh] Pretty name: $job_name"
echo "[magpie.sh] Total Prompts: $total_prompts"
echo "[magpie.sh] Instruction Generation Config: temp=$ins_temp, top_p=$ins_topp"
echo "[magpie.sh] Response Generation Config: temp=$res_temp, top_p=$res_topp, rep=$res_rep"
echo "[magpie.sh] System Config: device=$device, n=$n, batch_size=$batch_size, tensor_parallel=$tensor_parallel"
echo "[magpie.sh] Timestamp: $timestamp"
echo "[magpie.sh] Job Name: $job_name"

echo "[magpie.sh] Start Generating Instructions..."
CUDA_VISIBLE_DEVICES=$device python ../exp/gen_ins.py \
    --device $device \
    --model_path $model_path \
    --total_prompts $total_prompts \
    --top_p $ins_topp \
    --temp $ins_temp \
    --tensor_parallel $tensor_parallel \
    --gpu_memory_utilization $gpu_memory_utilization \
    --control_tasks code \
    --n $n \
    --job_name $job_name \
    --timestamp $timestamp \

echo "[magpie.sh] Finish Generating Instructions!"

echo "[magpie.sh] Start Generating Responses..."
CUDA_VISIBLE_DEVICES=$device python ../exp/gen_res.py \
    --device $device \
    --model_path $model_path \
    --batch_size $batch_size \
    --top_p $res_topp \
    --temp $res_temp \
    --rep $res_rep \
    --tensor_parallel $tensor_parallel \
    --gpu_memory_utilization $gpu_memory_utilization \
    --input_file $job_path/Magpie_${model_path##*/}_${total_prompts}_${timestamp}_ins.json \
    --offline \

echo "[magpie.sh] Finish Generating Responses!"