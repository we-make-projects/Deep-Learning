import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Example sequences (encoded as numbers)
# A=1, C=2, D=3, etc (just for demo)
X = [
    [1,2,3,4,5,0,0],   # padded
    [2,3,4,5,6,7,0],
    [1,3,5,7,2,4,6]
]

# Labels (0 = non-epitope, 1 = epitope)
y = [1, 0, 1]

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.float32)
class CNN_BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, lstm_hidden):
        super(CNN_BiLSTM, self).__init__()
        
        # 🔹 Embedding layer (turn numbers → vectors)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 🔹 CNN layer (pattern detection)
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=3,
            padding=1
        )
        
        # 🔹 BiLSTM (context understanding)
        self.bilstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )
        
        # 🔹 Final classification
        self.fc = nn.Linear(lstm_hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        print("Input:", x.shape)
        
        # Step 1: Embedding
        x = self.embedding(x)
        print("After Embedding:", x.shape)
        
        # Step 2: CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        print("After CNN:", x.shape)
        
        # Step 3: Back to (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # Step 4: BiLSTM
        x, _ = self.bilstm(x)
        print("After BiLSTM:", x.shape)
        
        # Take last time step
        x = x[:, -1, :]
        
        # Step 5: Dense
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x

model = CNN_BiLSTM(
    vocab_size=20,     # number of amino acids
    embed_dim=8,
    num_filters=16,
    lstm_hidden=32
)
output = model(X)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    output = model(X).squeeze()
    loss = criterion(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Epoch:", epoch, "Loss:", loss.item())
print("Final Output:", output)