from MyFunc import *
from MyModels import *
from MyConfig import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------- 使用训练好的模型预测并可视化对比 -------------
def evaluate_and_save(y_true, y_pred, y_prob, model_path, tag=""):

    print(f"评估 {tag} ...")
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{model_path[:-3]}_{tag}_classification_report.csv", index=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: {tag}")
    plt.savefig(f"{model_path[:-3]}_{tag}_confusion_matrix.png")
    plt.close()

    # save y_true, y_pred, y_prob 
    np.savez(f"{model_path[:-3]}_{tag}_predictions.npz", 
             y_true=y_true, 
             y_pred=y_pred, 
             y_prob=y_prob)

def predict_and_collect(model_class, model_path, X_test, y_test, week_idx):
    
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                           torch.tensor(y_test, dtype=torch.long)),
                             batch_size=128)

    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item()

            pred_labels = torch.argmax(preds, dim=1).cpu().numpy()
            all_preds.extend(pred_labels)
            all_labels.extend(yb.cpu().numpy())
            
            probs = torch.softmax(preds, dim=1)[:, 1]  # 概率是softmax后类别1的概率
            all_probs.extend(probs.cpu().numpy()) 

    # 计算平均 loss
    avg_loss = total_loss / len(y_test)
    print(f"第 {week_idx+1} 周平均 Cross-Entropy Loss: {avg_loss:.4f}")
    with open(f"{model_path[:-3]}_week{week_idx+1}_loss.txt", "w") as f:
        f.write(f"Average CrossEntropy Loss: {avg_loss:.4f}\n")

    # 每周单独保存指标
    evaluate_and_save(np.array(all_labels), np.array(all_preds), np.array(all_probs), model_path=model_path, tag=f"week{week_idx+1}")

    return all_preds, all_probs, all_labels


# ---------------- 主程序 -------------------
if __name__ == "__main__":
    
### CSI50
    data_args = data_args_CSI50

    #### 7 ####
    model_class = CNN3L_CBAM
    # 训练好的模型的存储路径
    model_name = "M_CNN3L_CBAM_e20_b128_l0.001_D_CSI50_T_0516-2111"
    save_path = f'model_res_for_backtest/{model_name}/{model_name}.pt'
    print(f"数据: {data_args['name']}")
    print(f"模型: {model_class.__name__}")
    print(f"保存路径: {save_path}")

    print("评估 CNN...")
    all_preds = []
    all_probs = []
    all_labels = []
    for week_idx, week_folder in enumerate(data_args['test_data_folder_list']):
        print(f"加载第 {week_idx+1} 周的数据...")
        test_df = load_data(
            data_folder_list=[week_folder],  # 注意这里是 [week_folder]，每次一个
            code_list=data_args['code_list'],
        )
        X_test, y_test = preprocess_data_cnn(test_df, is_train=False)
        print(f"本周测试集大小: {X_test.shape}")
        print(f"本周标签分布: {np.unique(y_test, return_counts=True)}")

        preds, probs, labels = predict_and_collect(model_class, save_path, X_test, y_test, week_idx)
        all_preds.extend(preds)
        all_labels.extend(labels)

    print("整体评估（全月数据）...")
    evaluate_and_save(np.array(all_labels), np.array(all_preds),np.array(all_probs), model_path=save_path, tag="overall")

    #### 8 ####
    model_class = CNN1L
    # 训练好的模型的存储路径
    model_name = "M_CNN1L_e20_b256_l0.0004_D_CSI50_T_0516-1720"
    save_path = f'model_res_for_backtest/{model_name}/{model_name}.pt'
    print(f"数据: {data_args['name']}")
    print(f"模型: {model_class.__name__}")
    print(f"保存路径: {save_path}")

    print("评估 CNN...")
    all_preds = []
    all_probs = []
    all_labels = []
    for week_idx, week_folder in enumerate(data_args['test_data_folder_list']):
        print(f"加载第 {week_idx+1} 周的数据...")
        test_df = load_data(
            data_folder_list=[week_folder],  # 注意这里是 [week_folder]，每次一个
            code_list=data_args['code_list'],
        )
        X_test, y_test = preprocess_data_cnn(test_df, is_train=False)
        print(f"本周测试集大小: {X_test.shape}")
        print(f"本周标签分布: {np.unique(y_test, return_counts=True)}")

        preds, probs, labels = predict_and_collect(model_class, save_path, X_test, y_test, week_idx)
        all_preds.extend(preds)
        all_labels.extend(labels)

    print("整体评估（全月数据）...")
    evaluate_and_save(np.array(all_labels), np.array(all_preds),np.array(all_probs), model_path=save_path, tag="overall")

    #### 9 ####
    model_class = ViT_SmallImage
    # 训练好的模型的存储路径
    model_name = "M_ViT_SmallImage_e20_b128_l0.0005_D_CSI50_T_0516-2232"
    save_path = f'model_res_for_backtest/{model_name}/{model_name}.pt'
    print(f"数据: {data_args['name']}")
    print(f"模型: {model_class.__name__}")
    print(f"保存路径: {save_path}")

    print("评估 CNN...")
    all_preds = []
    all_probs = []
    all_labels = []
    for week_idx, week_folder in enumerate(data_args['test_data_folder_list']):
        print(f"加载第 {week_idx+1} 周的数据...")
        test_df = load_data(
            data_folder_list=[week_folder],  # 注意这里是 [week_folder]，每次一个
            code_list=data_args['code_list'],
        )
        X_test, y_test = preprocess_data_cnn(test_df, is_train=False)
        print(f"本周测试集大小: {X_test.shape}")
        print(f"本周标签分布: {np.unique(y_test, return_counts=True)}")

        preds, probs, labels = predict_and_collect(model_class, save_path, X_test, y_test, week_idx)
        all_preds.extend(preds)
        all_labels.extend(labels)

    print("整体评估（全月数据）...")
    evaluate_and_save(np.array(all_labels), np.array(all_preds),np.array(all_probs), model_path=save_path, tag="overall")


### CSI2000
    data_args = data_args_CSI2000
    
    #### 1 ####
    model_class = ViT_SmallImage
    # 训练好的模型的存储路径
    model_name = "M_ViT_SmallImage_e20_b128_l0.0005_D_CSI2000_T_0517-0127"
    save_path = f'model_res_for_backtest/{model_name}/{model_name}.pt'
    print(f"数据: {data_args['name']}")
    print(f"模型: {model_class.__name__}")
    print(f"保存路径: {save_path}")

    print("评估 CNN...")
    all_preds = []
    all_probs = []
    all_labels = []
    for week_idx, week_folder in enumerate(data_args['test_data_folder_list']):
        print(f"加载第 {week_idx+1} 周的数据...")
        test_df = load_data(
            data_folder_list=[week_folder],  # 注意这里是 [week_folder]，每次一个
            code_list=data_args['code_list'],
        )
        X_test, y_test = preprocess_data_cnn(test_df, is_train=False)
        print(f"本周测试集大小: {X_test.shape}")
        print(f"本周标签分布: {np.unique(y_test, return_counts=True)}")

        preds, probs, labels = predict_and_collect(model_class, save_path, X_test, y_test, week_idx)
        all_preds.extend(preds)
        all_labels.extend(labels)

    print("整体评估（全月数据）...")
    evaluate_and_save(np.array(all_labels), np.array(all_preds),np.array(all_probs), model_path=save_path, tag="overall")

    #### 2 ####
    model_class = CNN3L_Transformer
    # 训练好的模型的存储路径
    model_name = "M_CNN3L_Transformer_e20_b128_l0.0005_D_CSI2000_T_0517-0053"
    save_path = f'model_res_for_backtest/{model_name}/{model_name}.pt'
    print(f"数据: {data_args['name']}")
    print(f"模型: {model_class.__name__}")
    print(f"保存路径: {save_path}")

    print("评估 CNN...")
    all_preds = []
    all_probs = []
    all_labels = []
    for week_idx, week_folder in enumerate(data_args['test_data_folder_list']):
        print(f"加载第 {week_idx+1} 周的数据...")
        test_df = load_data(
            data_folder_list=[week_folder],  # 注意这里是 [week_folder]，每次一个
            code_list=data_args['code_list'],
        )
        X_test, y_test = preprocess_data_cnn(test_df, is_train=False)
        print(f"本周测试集大小: {X_test.shape}")
        print(f"本周标签分布: {np.unique(y_test, return_counts=True)}")

        preds, probs, labels = predict_and_collect(model_class, save_path, X_test, y_test, week_idx)
        all_preds.extend(preds)
        all_labels.extend(labels)

    print("整体评估（全月数据）...")
    evaluate_and_save(np.array(all_labels), np.array(all_preds),np.array(all_probs), model_path=save_path, tag="overall")
