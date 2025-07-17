import os
import pandas as pd
import matplotlib.pyplot as plt

csv_dir = "logs_csv"
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

# 想画哪些 CSV -> PNG 图
plots = {
    "loss_overall_loss.csv": "loss_curve.png",
    "stage1_objectness_acc.csv": "accuracy_curve.png"
}

for csv_name, png_name in plots.items():
    csv_path = os.path.join(csv_dir, csv_name)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        tag = df.columns[-1]  # 自动获取值的列名
        plt.figure()
        plt.plot(df["step"].values, df[tag].values, label=tag)
        plt.xlabel("Step")
        plt.ylabel(tag)
        plt.title(tag)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, png_name))
        plt.close()
        print(f"✅ 已保存图像 {png_name}")
    else:
        print(f"⚠️ 未找到文件 {csv_path}")

