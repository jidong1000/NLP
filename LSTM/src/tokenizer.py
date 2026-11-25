import jieba
from tqdm import tqdm

class JiebaTokenizer:
    unk_token = '<unk>'
    pad_token = '<pad>'

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2id = {word:idx for idx, word in enumerate(vocab_list)}
        self.id2word = {idx:word for idx, word in enumerate(vocab_list)}
        self.unk_token_idx = self.word2id[self.unk_token]
        self.pad_token_idx = self.word2id[self.pad_token]

    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    def encode(self, text, seq_len):
        tokens = self.tokenize(text)
        
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens) < seq_len:
            tokens = tokens + [self.pad_token] * (seq_len - len(tokens))

        return [self.word2id.get(token, self.unk_token_idx) for token in tokens]

    @classmethod
    def build_vocab(cls, sentences, save_path):
        vocab_set = set()
        for sentence in tqdm(sentences, desc="构建词表"):
            vocab_set.update(jieba.lcut(sentence))

        vocab_list = [cls.pad_token, cls.unk_token] + [item for item in vocab_set if item.strip() != '']
        print(f'词表大小{len(vocab_list)}')

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)
