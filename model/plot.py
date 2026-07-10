import torch
import os

from matplotlib import pyplot as plt


def plot():
    """绘制中心服务器准确率曲线图并保存准确率到txt文件"""
    # 加载准确率数据
    accuracy = torch.load('./cache/accuracy.pkl')
    
    # 将准确率保存到txt文件（如果文件不存在或需要更新）
    accuracy_file = './cache/center_server_accuracy.txt'
    if not os.path.exists(accuracy_file):
        # 如果文件不存在，创建并写入所有数据
        with open(accuracy_file, 'w', encoding='utf-8') as f:
            f.write("Iteration\tAccuracy\n")
            for i, acc in enumerate(accuracy, start=1):
                f.write(f"{i}\t{acc:.6f}\n")
        print(f'中心服务器准确率已保存到: {accuracy_file}')
    else:
        # 如果文件已存在（训练过程中已实时保存），则更新最后一行的准确率（如果有变化）
        # 读取现有文件内容
        with open(accuracy_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 检查是否需要更新（比较最后一行）
        if len(lines) > 1:  # 至少有表头和一行数据
            last_line = lines[-1].strip()
            last_iteration = len(accuracy)
            if last_line.startswith(f"{last_iteration}\t"):
                # 最后一行已是最新的，不需要更新
                pass
            else:
                # 需要更新或追加
                with open(accuracy_file, 'a', encoding='utf-8') as f:
                    for i in range(len(lines) - 1, len(accuracy)):  # 从已有数据的下一行开始
                        f.write(f"{i + 1}\t{accuracy[i]:.6f}\n")
        else:
            # 只有表头，写入所有数据
            with open(accuracy_file, 'a', encoding='utf-8') as f:
                for i, acc in enumerate(accuracy, start=1):
                    f.write(f"{i}\t{acc:.6f}\n")
        print(f'中心服务器准确率文件已更新: {accuracy_file}')
    
    # 绘制准确率曲线图
    iterations = list(range(1, len(accuracy) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracy, label='Center Server', linewidth=2, marker='o', markersize=3)
    
    plt.title("Center Server Accuracy", fontsize=14, fontweight='bold')
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    
    plt.ylim(0, 1)
    if len(accuracy) > 0:
        plt.xlim(1, len(accuracy))
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=11)
    
    # 保存图片
    plot_file = './cache/center_server_accuracy_curve.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f'准确率曲线图已保存到: {plot_file}')
    
    plt.show()
