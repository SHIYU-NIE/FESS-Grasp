import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

log_dir = os.path.expanduser('~/FESS_refreshed')  # 修改为你的日志目录
output_dir = 'logs_csv'
os.makedirs(output_dir, exist_ok=True)

def export_scalars_to_csv(log_path):
    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        df = pd.DataFrame(events)
        df.rename(columns={"wall_time": "time", "step": "step", "value": tag}, inplace=True)
        csv_name = tag.replace("/", "_") + ".csv"
        df.to_csv(os.path.join(output_dir, csv_name), index=False)
        print(f"✅ 导出 {csv_name} ({len(df)} 条记录)")

for root, _, files in os.walk(log_dir):
    for fname in files:
        if fname.startswith("events.out.tfevents"):
            export_scalars_to_csv(os.path.join(root, fname))
