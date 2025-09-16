from MyFunc import *
from MyModels import *
from MyConfig import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- CNN 训练 + 保存 + 绘图 -------------------
def train_cnn(model_class, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=128, lr=1e-4, early_stop_cnt = 6, 
              save_path="cnn_model.pt", use_scheduler=False):
    # model = model_class().to(device)
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    model = model_class()
    model.apply(init_weights)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None


    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.long)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.long)),
                            batch_size=batch_size)

    train_losses = []
    val_losses = []
    val_f1s = []
    early_stopper = EarlyStopping(patience=early_stop_cnt, delta=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        # === 验证集 F1 ===
        model.eval()
        val_total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_total_loss += loss.item()
                
                pred_class = torch.argmax(pred, dim=1).cpu().numpy()
                all_preds.extend(pred_class)
                all_labels.extend(yb.cpu().numpy())

        val_loss = val_total_loss / len(val_loader)
        val_losses.append(val_loss)
        
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_f1s.append(val_f1)

        print(f"Epoch {epoch+1}: Loss={train_losses[-1]:.4f} | Val Loss={val_loss:.4f} | Val F1={val_f1:.4f}")

        
        if scheduler:
            scheduler.step()

        # === EarlyStopping 检查 ===
        early_stopper(val_f1)
        # early_stopper(-val_loss)
        if early_stopper.early_stop:
            print("触发 EarlyStopping，停止训练")
            break

    # === 保存模型 ===
    save_folder = os.path.dirname(save_path[:-3])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    model_save_path = f'{save_folder}/{save_path}'
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存到: {model_save_path}")

    # === 绘图 ===
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title('Training Loss & Validation F1')
    plt.grid()
    plt.tight_layout()
    
    train_plot_path = f"{save_folder}/{save_path[:-3]}_training_plot.png"
    plt.savefig(train_plot_path)
    print("训练曲线已保存为: ", train_plot_path)

    return model


# ---------------- 主程序 -------------------
if __name__ == "__main__":
    
### CSI2000
    data_args = data_args_CSI2000

    downsample_args = {
        "strategy": "fraction",
        "value": 0.5,
        "random_state": 42
    }
    
    train_df = load_data(
        data_folder_list=data_args['train_data_folder_list'],
        code_list=data_args['code_list'],
        downsample_args=downsample_args,
        is_Nov=True  # 训练、验证用11月的数据，测试用12月
    )
    val_df = load_data(
        data_folder_list=data_args['val_data_folder_list'],
        code_list=data_args['code_list'],
        downsample_args=downsample_args,
        is_Nov=True
    )

    X_train, X_val, y_train, y_val  = preprocess_data_cnn(train_df, balanced=False)    
    
    print(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")
    print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"验证集标签分布: {np.unique(y_val, return_counts=True)}")
    
    print("开始训练 CNN...")
    
    #### 1 ####
    model_class = CNN3L_Transformer
    epochs = 20
    batch_size = 128
    lr = 5e-4
    early_stop_cnt = 4
    use_scheduler = True
    
    print(f"数据: {data_args['name']}")
    print(f"模型: {model_class.__name__}")
    print(f"超参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    cur_time = datetime.now().strftime("%m%d-%H%M")
    name_str = f"M_{model_class.__name__}_e{epochs}_b{batch_size}_l{lr}_D_{data_args['name']}_T_{cur_time}"
    print(f"保存路径: {name_str}")
    save_path = name_str + ".pt"
    cnn_model = train_cnn(model_class, X_train, y_train, X_val, y_val, 
                          epochs, batch_size, lr, early_stop_cnt, save_path,
                          use_scheduler=use_scheduler)
    
    

    
    #### 2 ####
    model_class = ViT_SmallImage
    epochs = 20
    batch_size = 128
    lr = 5e-4
    early_stop_cnt = 4
    use_scheduler = True
    
    print(f"数据: {data_args['name']}")
    print(f"模型: {model_class.__name__}")
    print(f"超参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    cur_time = datetime.now().strftime("%m%d-%H%M")
    name_str = f"M_{model_class.__name__}_e{epochs}_b{batch_size}_l{lr}_D_{data_args['name']}_T_{cur_time}"
    print(f"保存路径: {name_str}")
    save_path = name_str + ".pt"
    cnn_model = train_cnn(model_class, X_train, y_train, X_val, y_val, 
                          epochs, batch_size, lr, early_stop_cnt, save_path,
                          use_scheduler=use_scheduler)
    
    
