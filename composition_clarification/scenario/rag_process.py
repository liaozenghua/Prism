import json
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 加载JSON数据
with open('/home/data/liaozenghua/ReST-SDLG/scenario/data_process_all.json', 'r') as f:
    data = json.load(f)

# 定义需要重命名的键
rename_map = {
    "element_process": "element",
    "example_process": "example",
    "preceding_element_process": "preceding_element",
    "hierarchical_elements_process": "hierarchical_elements"
}

# 处理每个意图
for intent in data:
    # 1. 重命名键
    for old_key, new_key in rename_map.items():
        if old_key in intent:
            intent[new_key] = intent.pop(old_key)

    # 2. 为示例生成嵌入向量
    if "example" in intent:
        examples = intent["example"]
        embeddings = model.encode(examples, convert_to_tensor=False)
        intent["rag_embedding"] = [embed.tolist() for embed in embeddings]  # 转换为列表存储

# 保存修改后的数据
with open('/home/data/liaozenghua/ReST-SDLG/scenario/rag.json', 'w') as f:
    json.dump(data, f, indent=2)

print("处理完成！生成嵌入向量并更新了键名。")