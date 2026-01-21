import json
import os

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def convert_format_data(source_dir):
    path = os.path.join(source_dir, "format", "ifeval_single_instr_format.jsonl")
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return []
    
    raw_data = load_jsonl(path)
    converted = []
    for item in raw_data:
        # prompt = with instruction (Position)
        # prompt_without_instruction = base (Negative)
        # instruction_id_list = e.g. ["change_case:english_lowercase"]
        
        subtype = item.get("instruction_id_list", ["unknown"])[0]
        
        converted.append({
            "task_type": "format",
            "sub_type": subtype,
            "prompt": item["prompt"],
            "pos": item["prompt"],
            "neg": item["prompt_without_instruction"],
            "instruction": subtype
        })
    return converted

def convert_keyword_data(source_dir, mode="include"):
    filename = "ifeval_single_keyword_include.jsonl" if mode == "include" else "ifeval_single_keyword_exclude.jsonl"
    path = os.path.join(source_dir, "keywords", filename)
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return []

    raw_data = load_jsonl(path)
    converted = []
    for item in raw_data:
        # keywords are in kwargs
        kw = item.get("kwargs", [{}])[0].get("keywords", [])
        kw_str = ",".join(kw) if kw else "unknown"
        
        converted.append({
            "task_type": "keyword",
            "sub_type": mode,
            "prompt": item["prompt"],
            "pos": item["prompt"], # Prompt asking for keyword
            "neg": item["prompt_without_instruction"], # Prompt without asking
            "instruction": f"{mode} {kw_str}"
        })
    return converted

def main():
    source_root = "llm-steer-instruct/data"
    output_path = "dataset/steering_tasks.jsonl"
    
    all_tasks = []
    
    print("Converting Format tasks...")
    all_tasks.extend(convert_format_data(source_root))
    
    print("Converting Keyword (Include) tasks...")
    all_tasks.extend(convert_keyword_data(source_root, mode="include"))
    
    print("Converting Keyword (Exclude) tasks...")
    all_tasks.extend(convert_keyword_data(source_root, mode="exclude"))
    
    # Write to JSONL
    with open(output_path, "w") as f:
        for task in all_tasks:
            f.write(json.dumps(task) + "\n")
            
    print(f"Successfully converted {len(all_tasks)} tasks to {output_path}")

if __name__ == "__main__":
    main()
