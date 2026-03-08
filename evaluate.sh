# Industrial_and_Scientific
# Office_Products
timestamp=$(date +%m%d_%H%M)

for category in "Industrial_and_Scientific"
do
    # your model path
    exp_name="/root/autodl-tmp/runs/Qwen2.5-1.5B/sft_checkpoint"

    exp_name_clean=$(basename "$exp_name")
    echo "Processing category: $category with model: $exp_name_clean (STANDARD MODE)"
    
    train_file=$(ls ./data/Amazon/train/${category}*.csv 2>/dev/null | head -1)
    test_file=$(ls ./data/Amazon/test/${category}*11.csv 2>/dev/null | head -1)
    info_file=$(ls ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1)
    
    if [[ ! -f "$test_file" ]]; then
        echo "Error: Test file not found for category $category"
        continue
    fi
    if [[ ! -f "$info_file" ]]; then
        echo "Error: Info file not found for category $category"
        continue
    fi
    
    temp_dir="/root/autodl-tmp/eval/temp/${timestamp}"
    output_dir="/root/autodl-tmp/eval/results/${timestamp}"

    echo "Creating temp directory: $temp_dir"
    mkdir -p "$temp_dir"
    echo "Creating output directory: $output_dir"
    mkdir -p "$output_dir"
    
    echo "Splitting test data..."
    python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "0"   # 4卡则"0,1,2,3"
    
    if [[ ! -f "$temp_dir/0.csv" ]]; then
        echo "Error: Data splitting failed for category $category"
        continue
    fi
    
    cudalist="0"  # 4卡则"0,1,2,3"
    echo "Starting parallel evaluation (STANDARD MODE)..."
    for i in ${cudalist}
    do
        if [[ -f "$temp_dir/${i}.csv" ]]; then
            echo "Starting evaluation on GPU $i for category ${category}"
            CUDA_VISIBLE_DEVICES=$i python -u ./evaluate.py \
                --base_model "$exp_name" \
                --info_file "$info_file" \
                --category ${category} \
                --test_data_path "$temp_dir/${i}.csv" \
                --result_json_data "$output_dir/${i}.json" \
                --batch_size 64 \
                --num_beams 10 \
                --max_new_tokens 5 \
                --length_penalty 0.0 &
        else
            echo "Warning: Split file $temp_dir/${i}.csv not found, skipping GPU $i"
        fi
    done
    echo "Waiting for all evaluation processes to complete..."
    wait
    
    result_files=$(ls "$output_dir"/*.json 2>/dev/null | wc -l)
    if [[ $result_files -eq 0 ]]; then
        echo "Error: No result files generated for category $category"
        continue
    fi

    actual_cuda_list=$(ls "$output_dir"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')
    echo "Merging results from GPUs: $actual_cuda_list"
    
    python ./merge.py \
        --input_path "$output_dir" \
        --output_path "$output_dir/merged_result_${category}.json" \
        --cuda_list "$actual_cuda_list"
    
    if [[ ! -f "$output_dir/merged_result_${category}.json" ]]; then
        echo "Error: Result merging failed for category $category"
        continue
    fi
    
    echo "Calculating metrics..."
    python ./calc.py \
        --path "$output_dir/merged_result_${category}.json" \
        --item_path "$info_file" \
        --save_txt "$output_dir/metric.txt"
    
    echo "Completed processing for category: $category"
    echo "Results saved to: $output_dir/merged_result_${category}.json"
    echo "----------------------------------------" 
done

echo "All categories processed!"
