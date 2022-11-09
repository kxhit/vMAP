import numpy as np

exp_list=["h4", "h8", "h16", "h32", "h64", "h128", "h256"]
metrics = []
for exp in exp_list:
    metric_avg = []
    metric = np.load("./logs/ablation/bmap_"+exp+"/eval_mesh/metrics_3D.npy")
    metric_avg.append(metric.shape[1])  # succ num
    metric_avg.extend(metric.mean(1).squeeze(-1))    # avg acc, rec, ratio1cm, ratio5cm

    metrics.append(np.stack(metric_avg))

metrics = np.stack(metrics)

print("metrics ", metrics.shape)
print(metrics)
np.save("./logs/ablation/obj_metric_ablation.npy", metrics)
