from MyFunc import *
from MyModels import *
from MyConfig import *
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------- 使用训练好的模型预测并可视化对比 -------------
def evaluate_and_save(y_true, y_pred, model_path, tag=""):

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

def predict_and_collect(model_class, model_path, X_test, y_test, week_idx, info_df):
    
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
    evaluate_and_save(np.array(all_labels), np.array(all_preds), model_path=model_path, tag=f"week{week_idx+1}")

    info_df['pred'] = all_preds
    info_df['prob'] = all_probs
    info_df['label'] = all_labels

    return all_preds, all_labels, info_df


# ---------------- 模拟交易回测 -------------------
def simulate_backtest(df, n_sig=5, init_cash=100000, use_prob=False):
    """
    对单只股票进行基于preds的简单择时回测。
    返回：
        df: 包含交易记录和资金曲线的 DataFrame
        metrics: 回测指标需要的中间变量（延后统一计算）
    """
    df = df.copy()
    df = df.sort_values('datetime').reset_index(drop=True)
    df['ret'] = df['mid_price'].pct_change().fillna(0)
    print("df.shape: ", df.shape)

    position = 0
    cash = init_cash
    equity_curve = []
    position_record = []
    buy_times = 0
    sell_times = 0
    win_trades = 0
    total_trades = 0
    entry_price = 0
    trade_record = []
    consecutive_preds = []

    for i in range(len(df)):
        pred = df.loc[i, 'prob'] if use_prob else df.loc[i, 'pred']
        ret = df.loc[i, 'ret']

        consecutive_preds.append(pred)
        if len(consecutive_preds) > n_sig:
            consecutive_preds.pop(0)

        trade = 0
        if len(consecutive_preds) == n_sig:
            avg_pred = np.mean(consecutive_preds)

            if avg_pred > 0.5 and position == 0:
                position = 1
                entry_price = df.loc[i, 'mid_price']
                buy_times += 1
                total_trades += 1
                trade = 1
            elif avg_pred <= 0.5 and position == 1:
                position = 0
                exit_price = df.loc[i, 'mid_price']
                sell_times += 1
                if exit_price > entry_price:
                    win_trades += 1
                trade = 1

        if position == 1:
            cash *= (1 + ret)

        equity_curve.append(cash)
        position_record.append(position)
        trade_record.append(trade)

    # print(equity_curve)
    df['equity'] = equity_curve
    df['position'] = position_record
    df['trade'] = trade_record

    total_return = df['equity'].iloc[-1] / init_cash - 1

    metrics = {
        'init_cash': init_cash,
        'final_cash': df['equity'].iloc[-1],
        'total_return': total_return,
        'win_trades': win_trades,
        'total_trades': total_trades,
        'buy_times': buy_times,
        'sell_times': sell_times,
        'datetime': df['datetime'],
        'equity': df['equity']
    }

    return df, metrics

def run_backtest(backtest_df, code_list, save_path, n_sig=3, use_prob=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    results = {}
    df_bt_list = []
    equity_df_list = []
    print("backtest_df.shape:", backtest_df.shape)
    # print(backtest_df.head(5))
    for code in code_list:
        print(code[:-2])
        df_code = backtest_df[backtest_df['Code'] == int(code[:-2])].copy()
        df_bt, stats = simulate_backtest(df_code, n_sig=n_sig, use_prob=use_prob)
        df_bt_list.append(df_bt)
        results[code] = stats

        equity_df = pd.DataFrame({'datetime': stats['datetime'], code: stats['equity']})
        equity_df_list.append(equity_df)

    # === 对齐时间戳 ===
    def align_timestamp(df, ts_col='datetime', freq=3):
        df = df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col])
        df[ts_col] = df[ts_col].dt.floor(f'{freq}S')  # 向下取整到3秒
        return df
    
    equity_df_list = [align_timestamp(df, ts_col='datetime', freq=3) for df in equity_df_list]

    # === 合并 equity 曲线
    combined_df = equity_df_list[0]
    for edf in equity_df_list[1:]:
        combined_df = pd.merge(combined_df, edf, on='datetime', how='outer')

    combined_df = combined_df.sort_values('datetime').fillna(method='ffill').fillna(method='bfill')
    combined_df['combined_equity'] = combined_df[code_list].mean(axis=1)
    # combined_df.to_csv('combined_df.csv', index=False)

    combined_equity = combined_df.set_index('datetime')['combined_equity']
    # combined_equity.to_csv('combined_equity.csv', index=True)
    combined_return_all = combined_equity.iloc[-1] / combined_equity.iloc[0] - 1

    # 构造一条初始资金数据，拼接到 combined_daily 前面
    init_cash = 100000  # 之前取的是mean，这里不用乘以10了
    init_index = pd.to_datetime("2021-12-05")  # 比实际回测第一天早
    initial_series = pd.Series([init_cash], index=[init_index])
    combined_equity = pd.concat([initial_series, combined_equity])
    combined_returns = combined_equity.pct_change().dropna()

    # === 计算指标 ===
    combined_daily = combined_equity.groupby(pd.to_datetime(combined_equity.index).date).last()
    combined_daily = pd.concat([initial_series, combined_daily])
    # combined_daily = combined_daily.sort_index()
    print("combined_daily:\n", combined_daily)
    daily_returns = combined_daily.pct_change().dropna()
    print("daily_returns:\n", daily_returns)

    ann_factor = 252 / 20  # 回测用了20个交易日
    summary = {
        'total_return': round(combined_return_all, 4),
        'annual_return': round((1 + combined_return_all) ** ann_factor - 1, 4),
        'sharpe_ratio': round(combined_returns.mean() / combined_returns.std(), 4) if combined_returns.std() > 0 else 0,
        'sharpe_ratio_daily': round(daily_returns.mean() / daily_returns.std(), 4) if daily_returns.std() > 0 else 0,
        'max_drawdown': round((combined_equity.cummax() - combined_equity).max() / combined_equity.cummax().max(), 4),
        'win_rate': round(np.sum([r['win_trades'] for r in results.values()]) /
                          (np.sum([r['total_trades'] for r in results.values()]) + 1e-6), 4),
        'trade_freq': round(np.sum([r['total_trades'] for r in results.values()]) / len(combined_equity), 6),
        'turnover': round(np.sum([r['buy_times'] + r['sell_times'] for r in results.values()]) / len(combined_equity), 6)
    }
    print("回测指标：")
    print(json.dumps(summary, indent=4))

    # === 保存净值图 ===
    plt.figure(figsize=(10, 4))
    plt.plot(combined_equity, label='Combined Equity')
    plt.title('Combined Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}_equity_curve.png')
    plt.close()

    # === 保存 summary ===
    with open(f'{save_path}_backtest_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

    # === 保存每只股票指标 ===
    result_df = pd.DataFrame(results).T
    result_df.index.name = 'code'
    result_df.to_csv(f'{save_path}_backtest_metrics.csv')



# 重复部分代码
def full_test(data_args, model_class, model_name):
    # 以下可重复
    code_list=data_args['code_list']
    save_path = f'model_res_for_backtest/{model_name}/{model_name}.pt'
    print(f"数据: {data_args['name']}")
    print(f"模型: {model_class.__name__}")
    print(f"保存路径: {save_path}")

    print("评估 CNN...")
    all_preds = []
    all_labels = []
    info_df_list = []

    # for week_idx, week_folder in enumerate(data_args['backtest_data_folder_list']):
    #     print(f"加载第 {week_idx+1} 周的数据...")
    #     test_df = load_test_data(
    #         data_folder_list=[week_folder],  # 注意这里是 [week_folder]，每次一个
    #         code_list=code_list,
    #     )
    #     X_test, y_test, info_df = preprocess_test_data_cnn(test_df)
    #     print(f"本周测试集大小: {X_test.shape}")
    #     print(f"本周标签分布: {np.unique(y_test, return_counts=True)}")

    #     preds, labels, info_df = predict_and_collect(model_class, save_path, X_test, y_test, week_idx, info_df)
    #     all_preds.extend(preds)
    #     all_labels.extend(labels)
    #     info_df_list.append(info_df)

    # print("整体评估（全月数据）...")
    # evaluate_and_save(np.array(all_labels), np.array(all_preds), model_path=save_path, tag="overall")
    
    # backtest_df = pd.concat(info_df_list, axis=0)
    # backtest_df.to_feather(f"{save_path[:-3]}_backtest.feather")

    # 回测入口
    backtest_df = pd.read_feather(f"{save_path[:-3]}_backtest.feather")
    try:
        run_backtest(backtest_df, code_list, save_path[:-3])
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        with open(f"{save_path}_error_log.txt", "w") as f:
            f.write(str(e))
        # continue
        

# ---------------- 主程序 -------------------
if __name__ == "__main__":
    
    data_args = data_args_CSI50

    #### 1 ####
    model_class = CNN3L
    model_name = "M_CNN3L_e20_b128_l0.0004_D_CSI50_T_0427-0252"
    full_test(data_args, model_class, model_name)

    # #### 2 ####
    # model_class = CNN3L_SE
    # model_name = "M_BinaryCNN_3_e20_b128_l0.001_D_CSI50_T_0424-1716"
    # full_test(data_args, model_class, model_name)

    # #### 3 ####
    # model_class = CNN5L
    # model_name = "M_CNN5L_e20_b128_l0.001_D_CSI50_T_0427-0203"
    # full_test(data_args, model_class, model_name)

    # #### 4 ####
    # model_class = BinaryVGG11
    # model_name = "M_BinaryVGG11_e20_b64_l0.0004_D_CSI50_T_0424-1924"
    # full_test(data_args, model_class, model_name)

    # #### 5 ####
    # model_class = ResNetCNN
    # model_name = "M_ResNetCNN_e20_b128_l0.001_D_CSI50_T_0427-0204"
    # full_test(data_args, model_class, model_name)

    # #### 6 ####
    # model_class = CNN1L
    # model_name = "M_CNN1L_e20_b256_l0.0004_D_CSI50_T_0516-1720"
    # full_test(data_args, model_class, model_name)

    # #### 7 ####
    # model_class = CNN1L_Mod
    # model_name = "M_CNN1L_Mod_e20_b256_l0.0004_D_CSI50_T_0516-1747"
    # full_test(data_args, model_class, model_name)

    # #### 8 ####
    # model_class = CNN2L
    # model_name = "M_CNN2L_e20_b256_l0.0001_D_CSI50_T_0516-1829"
    # full_test(data_args, model_class, model_name)

    # #### 9 ####
    # model_class = CNN2L_Mod
    # model_name = "M_CNN2L_Mod_e20_b256_l0.0001_D_CSI50_T_0516-1856"
    # full_test(data_args, model_class, model_name)

    # #### 10 ####
    # model_class = CNN3L_Mod
    # model_name = "M_CNN3L_Mod_e20_b128_l0.0004_D_CSI50_T_0516-2017"
    # full_test(data_args, model_class, model_name)

    # #### 11 ####
    # model_class = CNN4L
    # model_name = "M_CNN4L_e20_b128_l0.001_D_CSI50_T_0516-2302"
    # full_test(data_args, model_class, model_name)

    # #### 12 ####
    # model_class = CNN3L_CBAM
    # model_name = "M_CNN3L_CBAM_e20_b128_l0.001_D_CSI50_T_0516-2111"
    # full_test(data_args, model_class, model_name)

    # #### 13 ####
    # model_class = CNN3L_Transformer
    # model_name = "M_CNN3L_Transformer_e20_b128_l0.0005_D_CSI50_T_0516-2008"
    # full_test(data_args, model_class, model_name)

    # #### 14 ####
    # model_class = ViT_SmallImage
    # model_name = "M_ViT_SmallImage_e20_b128_l0.0005_D_CSI50_T_0516-2232"
    # full_test(data_args, model_class, model_name)

    # # data_args = data_args_CSI2000

    # # #### 1 ####
    # # model_class = CNN3L
    # # model_name = "M_CNN3L_e20_b128_l0.0004_D_CSI2000_T_0427-1355"
    # # full_test(data_args, model_class, model_name)

    # # #### 2 ####
    # # model_class = CNN3L_SE
    # # model_name = "M_CNN3L_SE_e20_b128_l0.001_D_CSI2000_T_0427-1429"
    # # full_test(data_args, model_class, model_name)

    # # #### 3 ####
    # # model_class = CNN4L
    # # model_name = "M_CNN4L_e20_b128_l0.001_D_CSI2000_T_0515-1724"
    # # full_test(data_args, model_class, model_name)

    # # #### 4 ####
    # # model_class = CNN5L
    # # model_name = "M_CNN5L_e20_b128_l0.001_D_CSI2000_T_0427-0439"
    # # full_test(data_args, model_class, model_name)

    # # #### 5 ####
    # # model_class = BinaryVGG11
    # # model_name = "M_BinaryVGG11_e20_b64_l0.0004_D_CSI2000_T_0427-1527"
    # # full_test(data_args, model_class, model_name)

    # # #### 6 ####
    # # model_class = ResNetCNN
    # # model_name = "M_ResNetCNN_e20_b128_l0.001_D_CSI2000_T_0427-1637"
    # # full_test(data_args, model_class, model_name)

    # # #### 7 ####
    # # model_class = CNN3L_CBAM
    # # model_name = "M_CNN3L_CBAM_e20_b128_l0.001_D_CSI2000_T_0515-1411"
    # # full_test(data_args, model_class, model_name)

    # # #### 8 ####
    # # model_class = CNN1L
    # # model_name = "M_CNN1L_e20_b256_l0.0001_D_CSI2000_T_0516-1422"
    # # full_test(data_args, model_class, model_name)

    # # #### 9 ####
    # # model_class = CNN1L_Mod
    # # model_name = "M_CNN1L_Mod_e20_b256_l0.0001_D_CSI2000_T_0516-1513"
    # # full_test(data_args, model_class, model_name)

    # # #### 10 ####
    # # model_class = CNN2L
    # # model_name = "M_CNN2L_e20_b256_l0.0001_D_CSI2000_T_0516-1433"
    # # full_test(data_args, model_class, model_name)

    # # #### 11 ####
    # # model_class = CNN2L_Mod
    # # model_name = "M_CNN2L_Mod_e20_b256_l0.0001_D_CSI2000_T_0516-1524"
    # # full_test(data_args, model_class, model_name)

    # # #### 12 ####
    # # model_class = CNN3L_Mod
    # # model_name = "M_CNN3L_Mod_e20_b128_l0.0004_D_CSI2000_T_0516-1640"
    # # full_test(data_args, model_class, model_name)

    # # #### 13 ####
    # # model_class = CNN3L_Transformer
    # # model_name = "M_CNN3L_Transformer_e20_b128_l0.0005_D_CSI2000_T_0517-0053"
    # # full_test(data_args, model_class, model_name)

    # # #### 14 ####
    # # model_class = ViT_SmallImage
    # # model_name = "M_ViT_SmallImage_e20_b128_l0.0005_D_CSI2000_T_0517-0127"
    # # full_test(data_args, model_class, model_name)
