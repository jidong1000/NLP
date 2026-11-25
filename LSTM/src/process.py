import pandas as pd
from config import *
from sklearn.model_selection import train_test_split
from tokenizer import JiebaTokenizer

csv_path = RAW_DATA_DIR / 'online_shopping_10_cats.csv'

def process():
    print("开始处理数据")

    df = pd.read_csv(csv_path, usecols=['label', 'review'], encoding='utf-8').dropna()
    # print(df.head())
    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])
    # 构建词表
    JiebaTokenizer.build_vocab(train_df['review'].tolist(), PROCESSED_DATA_DIR / 'vocab.txt')
    # 构建tokenizer
    tokenizer = JiebaTokenizer.from_vocab(PROCESSED_DATA_DIR / 'vocab.txt')
    # 计算序列长度
    # print(train_df['review'].apply(lambda x : len(tokenizer.tokenize(x))).quantile(0.95))
    # 构建训练集
    train_df['review'] = train_df['review'].apply(lambda x : tokenizer.encode(x, SEN_LEN))
    # 保存训练集
    train_df.to_json(PROCESSED_DATA_DIR / 'train.jsonl', orient='records', lines=True)
    # 构建测试集
    test_df['review'] = test_df['review'].apply(lambda x : tokenizer.encode(x, SEN_LEN))
    # 保存测试集
    test_df.to_json(PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)

    print("数据处理完成")

if __name__ == "__main__":
    process()