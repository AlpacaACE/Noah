import json

# 文件列表
file_list = ['2020-40_zh_head_0000.jsonl', '2020-40_zh_head_0001.jsonl', '2020-40_zh_head_0002.jsonl', '2020-40_zh_head_0003.jsonl', '2020-40_zh_head_0004.jsonl', '2020-40_zh_middle_0000.jsonl', '2020-40_zh_middle_0001.jsonl', '2020-40_zh_middle_0002.jsonl', '2020-40_zh_middle_0003.jsonl', '2020-40_zh_middle_0004.jsonl' ,'2021-17_zh_head_0000.jsonl', 'mobvoi_seq_monkey_general_open_corpus.jsonl']

# 输出文件名
output_file = 'pretrain_data_total.jsonl'

# 打开输出文件，并指定UTF-8编码
with open(output_file, 'w', encoding='utf-8') as outfile:
    for file in file_list:
        # 依次读取每个文件
        with open(file, 'r', encoding='utf-8') as infile:
            for line in infile:
                # 将每一行写入到输出文件中
                json_data = json.loads(line)
                outfile.write(json.dumps(json_data, ensure_ascii=False) + '\n')

print(f"合并完成，已保存到 {output_file}")

# 第二步：读取合并后的文件并打印前5行
with open(output_file, 'r', encoding='utf-8') as infile:
    for i, line in enumerate(infile):
        if i < 5:
            json_data = json.loads(line)
            print(json_data)
        else:
            break
