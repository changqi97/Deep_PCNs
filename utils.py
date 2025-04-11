import matplotlib.pyplot as plt
import numpy as np

def plot_line_chart_with_bar_chart(epochs, dataT, accuracy_train, accuracy_val, savenpath=None):
    data = list(map(list, zip(*dataT)))
    
    # 创建图形和坐标轴，调整figsize使图例有位置显示
    fig, ax1 = plt.subplots(figsize=(12, 8))  # 加宽图形以容纳图例

    # 创建双轴，并绘制柱状图
    ax2 = ax1.twinx()  # 创建共享 x 轴的第二个 y 轴
    bar_width = 0.4  # 柱状图宽度
    x = np.arange(len(epochs))

    # 绘制柱状图
    bars_train = ax1.bar(x - bar_width/2, accuracy_train, width=bar_width, color='skyblue', label='Train Accuracy', alpha=0.3)
    bars_val = ax1.bar(x + bar_width/2, accuracy_val, width=bar_width, color='orange', label='Val Accuracy', alpha=0.3)

    # 设置第二个y轴为对数刻度
    ax2.set_yscale('log')

    # 绘制折线图
    lines = []
    for i, line in enumerate(data):
        line_plot, = ax2.plot(np.arange(len(epochs)), line, label=f'Layer {i+1}', marker='o')
        lines.append(line_plot)

    # 设置折线图属性
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Energy of Each Layer (log scale)")
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 设置柱状图属性
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)

    # 设置横轴刻度
    ax2.set_xticks(x)
    ax2.set_xticklabels(epochs)

    # 在柱状图的最高值上标注数值
    max_train_idx = np.argmax(accuracy_train)
    max_val_idx = np.argmax(accuracy_val)

    # 标注 Train Accuracy 的最大值
    bar_train = bars_train[max_train_idx]
    height_train = bar_train.get_height()
    ax1.text(bar_train.get_x() + bar_train.get_width() / 2, height_train + 0.02, 
            f'{height_train:.4f}', ha='center', va='bottom', fontsize=14, color='blue')

    # 标注 Val Accuracy 的最大值
    bar_val = bars_val[max_val_idx]
    height_val = bar_val.get_height()
    ax1.text(bar_val.get_x() + bar_val.get_width() / 2, height_val + 0.02, 
            f'{height_val:.4f}', ha='center', va='bottom', fontsize=14, color='orange')

    # 获取所有图例句柄和标签
    bars_legend = ax1.get_legend_handles_labels()
    lines_legend = ax2.get_legend_handles_labels()
    
    # 将图例放在图形外部右侧
    fig.legend(bars_legend[0] + lines_legend[0], 
              bars_legend[1] + lines_legend[1],
              loc='center left', 
              bbox_to_anchor=(1.00, 0.7))

    # 调整布局以确保图例不被裁切
    plt.tight_layout()
    
    # 保存数据和图像
    if savenpath:
        # 创建数据字典
        csv_data = {
            'Epoch': epochs,
            'Train_Accuracy': accuracy_train,
            'Val_Accuracy': accuracy_val
        }
        # 添加每层的数据
        for i, layer_data in enumerate(data):
            csv_data[f'Layer_{i+1}'] = layer_data
            
        # 转换为DataFrame并保存
        import pandas as pd
        df = pd.DataFrame(csv_data)
        csv_path = savenpath.rsplit('.', 1)[0] + '.csv'
        df.to_csv(csv_path, index=False)

        # 保存图像
        title = savenpath.split("/")[-1].split(".")[0]   
        plt.title(title)
        # 保存时确保图例不被裁切
        plt.savefig(savenpath, bbox_inches='tight')
    else:
        plt.show()


def generate_sequence_Log(x, k=1.0):
    """
    Generates a sequence of length x whose sum is 1, 
    with the curve changing rapidly at the beginning and flattening out later.
    
    Parameters:
        x: Length of the sequence
        k: Decay rate parameter; larger values result in faster changes in the curve (default is 1.0)
    
    Returns:
        A sequence of length x with a sum of 1
    """
    # Generate an exponentially decaying sequence
    indices = np.arange(x)  # Create indices from 0 to x-1
    raw_sequence = np.exp(-k * indices)  # Apply exponential decay
    # Normalize the sequence so that the sum is 1
    normalized_sequence = raw_sequence / np.sum(raw_sequence)
    return normalized_sequence

def generate_sequence_Exp(x):
    """
    Generates a sequence of length x whose sum is 1, satisfying the following pattern:
    The first value is 1, the second value is 1/2, the third value is 1/4, and so on.
    
    Parameters:
        x: Length of the sequence
    
    Returns:
        A sequence of length x with a sum of 1
    """
    # Generate the sequence 1, 1/2, 1/4, ..., 1/(2**(x-1))
    raw_sequence = np.array([1 / (2**i) for i in range(x)])
    # Normalize the sequence so that the sum is 1
    # normalized_sequence = raw_sequence / np.sum(raw_sequence)
    return raw_sequence# normalized_sequence

def get_sequences(N, T, k, decay_fn="PC"):
    
    if decay_fn == "PC":
        sequences = np.ones((N, T))
    elif decay_fn == "log":
        sequences = np.zeros((N, T))
        for i in range(N):
            sequences[N-i-1, i:T] = generate_sequence_Log(T-i, k)
    elif decay_fn == "logD":
        sequences = np.zeros((N, T))
        for i in range(N):
            sequences[N-i-1, i:T] = np.insert(0.1 * generate_sequence_Log(T-i-1, k), 0, 1.0)
        print(sequences.T.tolist())
    elif decay_fn == "logN": # logDeacy, but the trained times is the same for all the vodes
        sequences = np.zeros((N, T))
        logseq = generate_sequence_Log(T-N+1, k)
        for i in range(N):
            sequences[N-i-1, i:T-N+1+i] = logseq
    elif decay_fn == "exp":
        sequences = np.zeros((N, T))
        for i in range(N):
            sequences[N-i-1, i:T] = generate_sequence_Exp(T-i)
    elif decay_fn == "expN":
        sequences = np.zeros((N, T))
        expseq = generate_sequence_Exp(T-N+1)
        for i in range(N):
            sequences[N-i-1, i:T-N+1+i] = expseq
    elif decay_fn == "BP":
        sequences = np.zeros((N, T))
        for i in range(N):
            sequences[N-i-1, i] = 1
        # print(sequences)
    # sequencesW = np.zeros((N, T))
    # for i in range(N):
    #     sequencesW[i, i] = 1
    
    # print(sequences)

    # print(sequencesW)
    # return [sequences.T.tolist(), sequencesW.T.tolist()]
    return sequences.T.tolist()

# for i in get_sequences(5, 5, 1.0, "exp"):
#     print('i',i)