import matplotlib.pyplot as plt
import os 

precision_list = ['INT8', 'FP16', 'FP32']
loading_time = []
inference_time = []
fps = []

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_generator = lambda x: os.path.join(BASE_DIR, "results", x)

for precision in precision_list:
    with open(os.path.join(path_generator(precision), 'stats.txt'), 'r') as f:
        loading_time.append(float(f.readline().split("\n")[0]))
        inference_time.append(float(f.readline().split("\n")[0]))
        fps.append(float(f.readline().split("\n")[0]))

plt.bar(precision_list, loading_time)
plt.xlabel("Precision")
plt.ylabel("Total model loading time(s)")
plt.savefig(os.path.join(BASE_DIR, "results", "loading_time.png"), bbox_inches='tight')

plt.bar(precision_list, inference_time)
plt.xlabel("Precision")
plt.ylabel("Total inference Time(s)")
plt.savefig(os.path.join(BASE_DIR, "results", "inference_time.png"), bbox_inches='tight')

plt.bar(precision_list, fps)
plt.xlabel("Precision")
plt.ylabel("FPS")
plt.savefig(os.path.join(BASE_DIR, "results", "fps.png"), bbox_inches='tight')

