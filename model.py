import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class FinanceDataset(Dataset):
    def __init__(self, entries, task_type='binary', target='ret5'):
        # 确保数据仍留在 CPU
        images = np.stack([x[0] for x in entries], axis=0).astype(np.float32) / 255.0
        self.images = torch.tensor(images) 
        
        if task_type == 'binary':
            self.labels = torch.tensor([1 if x[1 if target=='ret5' else 2] == 1 else 0 for x in entries], dtype=torch.float32)
        else:
            self.labels = torch.tensor([x[1 if target=='ret5' else 2] + 1 for x in entries], dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 在 DataLoader 里，动态转换到 GPU
        return self.images[idx].unsqueeze(0), self.labels[idx]

def create_dataloaders(entries, batch_size=128, target='ret5', num_workers=4):
    """
    创建训练和验证集的DataLoader，用于模型训练
    return: train_loader, val_loader
    """
    dataset = FinanceDataset(entries, task_type='binary', target=target)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    # 优化 DataLoader，不要预加载进 GPU
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

def create_cnn_dataloaders(entries, batch_size=128, target='ret5', num_workers=4, test_size=0.2):
    """
    创建训练、验证、测试集DataLoader，保持与传统方法相同的划分逻辑
    return: train_loader, val_loader, test_loader
    """
 
    # 获取与传统方法相同的索引划分
    X_indices = np.arange(len(entries))
    y = np.array([1 if entry[1] == 1 else 0 for entry in entries])  # ret5
    
    # 第一次划分：训练+验证 与 测试集
    X_train_val_idx, X_test_idx, _, _ = train_test_split(
        X_indices, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=42
    )
    
    # 第二次划分：训练集与验证集
    X_train_idx, X_val_idx = train_test_split(
        X_train_val_idx,
        test_size=0.25,  # 0.25 * 0.8 = 0.2
        stratify=y[X_train_val_idx],
        random_state=42
    )
    
    # 创建子数据集
    train_dataset = torch.utils.data.Subset(FinanceDataset(entries, target=target), X_train_idx)
    val_dataset = torch.utils.data.Subset(FinanceDataset(entries, target=target), X_val_idx)
    test_dataset = torch.utils.data.Subset(FinanceDataset(entries, target=target), X_test_idx)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


# Model
class BinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1: [1, 41, 15] -> [256, 20, 7]
            nn.Conv2d(1, 256, kernel_size=(6,6), stride=(2,1), padding=(2,2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
            # Layer 2: [256, 20, 7] -> [512, 11, 8]
            nn.Conv2d(256, 512, kernel_size=(2,2), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            
            # Layer 3: [512, 11, 8] -> [1024, 6, 9]
            nn.Conv2d(512, 1024, kernel_size=(2,2), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            nn.Dropout(0.5)
        )
        
        # 自动计算全连接层输入尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 41, 15)  # 输入尺寸假设
            dummy = self.features(dummy)
            self.fc_in = dummy.view(1, -1).size(1)
        
            
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_in, 40960),
            nn.ReLU(),
            nn.Linear(40960, 1)  # 移除 nn.Sigmoid(), 因为损失函数nn.BCEWithLogitsLoss()
        )
        
        # print the output size
        print(f"Flattened size: {self.fc_in}")  # 27648 = 1024*3*9
        
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Initializer
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

def cnn_train(model, train_loader, val_loader, lr=1e-5, epochs=20, target='ret5'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # CUDA out of memory
    # device = torch.device('cpu')
    print(f"Using device: {device}")
    model.to(device)

    # Choose loss function
    if isinstance(model, BinaryCNN):
        criterion = nn.BCEWithLogitsLoss()  # 应用类别权重
    else:
        criterion = nn.CrossEntropyLoss(device=device)


    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loss_list = []
    val_loss_list = []
    best_acc = 0.0
    
    # 早停机制
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5  # 连续patience个epoch验证loss不下降则停止

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        total_train_samples = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            if isinstance(model, BinaryCNN):
                labels = labels.unsqueeze(1)  # Ensure proper shape for BCEWithLogitsLoss

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            total_train_samples += batch_size

            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_loss_list.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="batch")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                if isinstance(model, BinaryCNN):
                    labels = labels.unsqueeze(1)  # 让 labels 变成 [batch_size, 1]
                
                outputs = model(inputs)
                
                loss = criterion(outputs, labels if isinstance(model, BinaryCNN) else labels)
                val_loss += loss.item() * inputs.size(0)

                if isinstance(model, BinaryCNN):
                    preds = (torch.sigmoid(outputs) > 0.5).long()
                else:
                    preds = torch.argmax(outputs, dim=1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                val_pbar.set_postfix(
                    acc=f"{correct/total:.2%}",
                    loss=f"{loss.item():.4f}"
                )
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_loss_list.append(avg_val_loss)
        
        # 早停判断
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_{target}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发，第{epoch+1}轮停止训练")
                break

        # Save Best Model
        epoch_acc = correct / total
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), f"best_model_{target}.pth")

    print(f"Best Validation Accuracy: {best_acc:.2%}")
    
    # 绘制loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def cnn_predict(model, test_loader, device='cuda'):
    """在测试集上进行预测，返回概率和标签"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    return np.array(all_probs), np.array(all_labels)

def evaluate_cnn(y_true, y_prob, threshold=0.5):
    """计算评估指标，与传统机器学习结果可比"""
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred)
    }
