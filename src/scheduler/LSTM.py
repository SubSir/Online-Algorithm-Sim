import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .belady import Belady


class LSTM_Attention_Model(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, N):
        super().__init__()
        self.N = N
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim, hidden_size=hidden_size, num_layers=1, batch_first=True
        )
        self.scale = 1.0 / math.sqrt(hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 输入形状处理
        x_emb = self.embedding(x).squeeze(1)  # [batch_size, emb_dim]

        # 生成滑动窗口 [batch_size-N+1, N, emb_dim]
        x_seq = x_emb.unfold(0, self.N, 1).permute(0, 2, 1)

        # LSTM处理
        out, (hn, cn) = self.lstm(x_seq)  # [batch_size-N+1, N, hidden_size]

        # Attention机制
        h_t = out[:, -1, :]  # 最后一个时间步的hidden state
        h_s = out  # 所有时间步的hidden states

        # 计算注意力权重
        attn_scores = torch.bmm(h_s, h_t.unsqueeze(-1)).squeeze(-1) * self.scale
        attn_weights = F.softmax(attn_scores, dim=1)

        # 上下文向量
        context = torch.sum(h_s * attn_weights.unsqueeze(-1), dim=1)

        # 最终预测
        output = torch.sigmoid(self.out(context))
        return output  # [batch_size-N+1, 1]


class LSTM_Cache:
    def __init__(self, cache_size, vocab_size, emb_dim=32, hidden_size=64, N=5):
        self.cache_size = cache_size
        self.vocab_size = vocab_size
        self.N = N
        self.model = LSTM_Attention_Model(vocab_size, emb_dim, hidden_size, N)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train(self, requests):
        X = torch.tensor([i.obj_id % self.vocab_size for i in requests])
        belady = Belady(self.cache_size)
        belady.initial(requests)
        belady.resize(len(X))
        Y = torch.tensor(belady.result)
        for epoch in range(100):
            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(X)

            # 对齐标签（取最后N-1个标签之后的标签）
            valid_labels = Y[self.N - 1 :].float().view(-1, 1)

            # 计算损失
            loss = F.binary_cross_entropy(outputs, valid_labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        self.optimizer.zero_grad()
        return [0] * (self.N - 1) + self.model(X).squeeze().tolist()


# 示例使用
if __name__ == "__main__":
    seq_length = 100  # 总序列长度
    N = 5  # 滑动窗口大小
    vocab_size = 100
    emb_dim = 32
    hidden_size = 64

    # 生成示例数据（模拟长序列处理）
    X = torch.randint(0, vocab_size, (seq_length, 1))  # 输入序列
    Y = torch.rand(seq_length) > 0.5  # 模拟标签

    # 创建模型
    model = LSTM_Attention_Model(vocab_size, emb_dim, hidden_size, N)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(100):
        optimizer.zero_grad()

        # 前向传播
        outputs = model(X)

        # 对齐标签（取最后N-1个标签之后的标签）
        valid_labels = Y[N - 1 :].float().view(-1, 1)

        # 计算损失
        loss = F.binary_cross_entropy(outputs, valid_labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
