from transformers import AutoTokenizer
import jsonlines
import numpy as np

bos_token = "<s>"
eos_token = "</s>"

def data_clean(batch_size=10000):
    batch = []
    data_available = 0
    with jsonlines.open('./pretrain_data_total.jsonl') as reader:
        for idx, obj in enumerate(reader):
            try:
                content = obj.get('text', '')
                if len(content) > 512:
                    # 找到512字符以内的部分
                    truncated_content = content[:512]
                    # 找到最后一个句号的位置
                    last_period_index = truncated_content.rfind('。')
                    if last_period_index != -1:
                        # 截取最后一个句号之前的内容
                        content = truncated_content[:last_period_index + 1]
                    else:
                        # 如果没有句号，直接截取512字符
                        content = truncated_content

                text_id = tokenizer(f'{bos_token}{content}{eos_token}').input_ids
                batch.extend(text_id)
                data_available += 1

                # 当达到批次大小时，写入文件
                if len(batch) >= batch_size:
                    arr = np.array(batch, dtype=np.uint16)
                    with open('./pretrain_data.bin', 'ab') as f:
                        f.write(arr.tobytes())
                    batch.clear()
                    print(f"Processed {idx + 1} records and appended to pretrain_data.bin")

                # 每处理50000条记录打印一次进度
                if idx % 50000 == 0:
                    print(f"seq_monkey: [{idx}]")
            except UnicodeDecodeError as e:
                print(f"Skipping invalid line {idx + 1}: {e}")
                continue

    # 处理最后一批次数据
    if batch:
        arr = np.array(batch, dtype=np.uint16)
        with open('./pretrain_data.bin', 'ab') as f:
            f.write(arr.tobytes())
        print("Wrote remaining data to pretrain_data.bin")

    print(f"data_available: {data_available}")

def pretrain_process():
    data_clean()

    # 合并所有数据文件（如果有多个）
    data_path_list = ['./pretrain_data.bin']
    data_lst = []
    for data_path in data_path_list:
        with open(data_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16)
            data_lst.append(data)
    
    arr = np.concatenate(data_lst)

    with open('./pretrain_data_merge.bin', 'wb') as f:
        f.write(arr.tobytes())

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('../../tokenizer', use_fast=False)
    print('tokenizer_size：', len(tokenizer))

    pretrain_process()