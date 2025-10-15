import torch  
import torch.nn as nn

class LightweightTransformerModel(nn.Module):
    def __init__(self, item_dim, location_dim, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super(LightweightTransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 嵌入层
        self.item_embedding = nn.Embedding(item_dim, 16)
        self.location_embedding = nn.Embedding(location_dim, 8)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, 20, d_model))
        
        # 数值特征的线性投影
        self.numeric_proj = nn.Linear(5, d_model - 16 - 8)  # 5个数值特征
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 当前特征处理
        current_input_dim = 16 + 8 + 4  # item_emb + location_emb + numeric features
        self.current_proj = nn.Linear(current_input_dim, d_model)
        
        # 输出层
        self.fc1 = nn.Linear(d_model * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, seq_data, current_data):
        batch_size, seq_len = seq_data.size(0), seq_data.size(1)
        
        # 处理序列数据
        item_seq = seq_data[:, :, 0].long()
        location_seq = seq_data[:, :, 1].long()
        numeric_seq = seq_data[:, :, 2:7]
        
        # 嵌入和投影
        item_emb = self.item_embedding(item_seq)
        location_emb = self.location_embedding(location_seq)
        numeric_proj = self.numeric_proj(numeric_seq)
        
        # 拼接所有序列特征
        seq_combined = torch.cat([item_emb, location_emb, numeric_proj], dim=2)
        
        # 添加位置编码
        seq_combined = seq_combined + self.pos_encoder[:, :seq_len, :]
        
        # Transformer编码
        transformer_out = self.transformer_encoder(seq_combined)
        seq_representation = transformer_out[:, -1, :]  # 取最后一个时间步
        
        # 处理当前特征
        current_item = current_data[:, 0].long()
        current_location = current_data[:, 1].long()
        current_numeric = current_data[:, 2:]
        
        current_item_emb = self.item_embedding(current_item)
        current_location_emb = self.location_embedding(current_location)
        current_combined = torch.cat([current_item_emb, current_location_emb, current_numeric], dim=1)
        current_representation = self.current_proj(current_combined)
        
        # 融合特征
        combined = torch.cat([seq_representation, current_representation], dim=1)
        
        # 输出层
        x = self.dropout(torch.relu(self.fc1(combined)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return self.sigmoid(x).squeeze()