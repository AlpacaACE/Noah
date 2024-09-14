from transformers import AutoTokenizer
import jsonlines
import numpy as np

bos_token = "<s>"
eos_token = "</s>"

def data_clean():
    doc_ids = []
    data_available = 0
    with jsonlines.open('./pretrain_data.json') as reader:
        for idx, obj in enumerate(reader):
            try:
                content = obj.get('text', '')
                if len(content) > 512:
                    continue
                text_id = tokenizer(f'{bos_token}{content}{eos_token}').data['input_ids']
                doc_ids += text_id
                data_available += 1
                if idx % 50000 == 0:
                    print(f"seq_monkey: [{idx}]")
            except UnicodeDecodeError as e:
                print(f"Skipping invalid line {idx + 1}: {e}")
                continue

    print(f"data_available: {data_available}")
    arr = np.array(doc_ids, dtype=np.uint16)

    with open('./pretrain_data_clean.bin', 'wb') as f:
        f.write(arr.tobytes())

def pretrain_process():
    data_clean()

    data_path_list = [
        './pretrain_data_clean.bin'
    ]
    data_lst = []
    for data_path in data_path_list:
        with open(data_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16)
            data_lst.append(data)
    
    arr = np.concatenate(data_lst)

    with open('./pretrain_data.bin', 'wb') as f:
        f.write(arr.tobytes())

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('../tokenizer_mistral', use_fast=False)
    print('tokenizer_size：', len(tokenizer))

    pretrain_process()