input_file=${1:-"none"}
tag_mission=${2:-"all"}
model_path=${2:-"meta-llama/Meta-Llama-3-8B-Instruct"}
res_rep=1
device="0"
tensor_parallel=1
gpu_memory_utilization=0.95
batch_size=1000

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
echo "[magpie.sh] Tagging Mission: $tag_mission"
echo "[magpie.sh] Model Name: $model_path"
echo "[magpie.sh] System Config: device=$device, batch_size=$batch_size, tensor_parallel=$tensor_parallel"

if [ $tag_mission == "difficulty" ] || [ $tag_mission == "all" ]; then
    echo "[magpie.sh] Start Generating Difficulty Tags..."
    CUDA_VISIBLE_DEVICES=$device python ../exp/unitag.py \
        --device $device \
        --model_path $model_path \
        --input_file $input_file \
        --tag_mission "difficulty" \
        --tensor_parallel $tensor_parallel \
        --gpu_memory_utilization $gpu_memory_utilization \
        --batch_size $batch_size \

    echo "[magpie.sh] Finish Generating Difficulty Tags!"

    # Change input file name to difficulty tagged file
    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    difficulty_tag_file="${input_file_dir}/${input_file_name_no_ext}_difficulty.${input_file_ext}"
    input_file=$difficulty_tag_file
    echo "[magpie.sh] Difficulty Tagged File: $input_file"
fi

if [ $tag_mission == "quality" ] || [ $tag_mission == "all" ]; then
    echo "[magpie.sh] Start Generating Quality Tags..."
    CUDA_VISIBLE_DEVICES=$device python ../exp/unitag.py \
        --device $device \
        --model_path $model_path \
        --input_file $input_file \
        --tag_mission "quality" \
        --tensor_parallel $tensor_parallel \
        --gpu_memory_utilization $gpu_memory_utilization \
        --batch_size $batch_size \

    echo "[magpie.sh] Finish Generating Quality Tags!"

    # Change input file name to quality tagged file
    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    quality_tag_file="${input_file_dir}/${input_file_name_no_ext}_quality.${input_file_ext}"
    input_file=$quality_tag_file
    echo "[magpie.sh] Quality Tagged File: $input_file"
fi

if [ $tag_mission == "classification" ] || [ $tag_mission == "all" ]; then
    echo "[magpie.sh] Start Generating Task Tags..."
    CUDA_VISIBLE_DEVICES=$device python ../exp/unitag.py \
        --device $device \
        --model_path $model_path \
        --input_file $input_file \
        --tag_mission "classification" \
        --tensor_parallel $tensor_parallel \
        --gpu_memory_utilization $gpu_memory_utilization \
        --batch_size $batch_size \

    echo "[magpie.sh] Finish Generating Task Tags!"

    # Change input file name to task tagged file
    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    task_tag_file="${input_file_dir}/${input_file_name_no_ext}_category.${input_file_ext}"
    input_file=$task_tag_file
    echo "[magpie.sh] Task Tagged File: $input_file"
fi