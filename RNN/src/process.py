import pandas as pd
from sklearn.model_selection import train_test_split
import jieba
from tqdm import tqdm
from config import *


def prosecc():
    print("开始处理数据")

    df = pd.read_json(RAW_DATA_DIR / 'synthesized_.jsonl', orient='records', lines=True)
    # print(df.head())
    datas = df['dialog'].to_list()
    # print(datas[0])
    # print(len(datas))
    sentences = []
    for data in datas:
        for sentence in data:
            sentences.append(sentence.split('：')[1])
    # print(sentences[0])
    # print(len(sentences))

    # 划分数据集
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)
    # 构建词表，基于训练集构建，防止数据泄露
    vocab_set = set()
    for sentence in tqdm(train_sentences, desc="构建词表"):
        vocab_set.update(jieba.lcut(sentence))
    # 放入未登录词在第一位
    vocab_list = ['<unk>'] + list(vocab_set)

    # 保存训练集词表
    with open(MODELS_DIR / 'vocab.txt', mode='w', encoding='utf-8') as f:
        f.write('\n'.join(vocab_list))

    
    

    print("数据处理结束")

if __name__ == '__main__':
    prosecc()
