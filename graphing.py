import matplotlib.pyplot as plt
import csv
import math


def plot_throughput():
    plt.subplot()
    with open("/work/ta127/ta127/chrisrae/chris-ml-intern/ML/ResNet50/Torch/results.csv", "r", newline="") as file:
        reader = csv.DictReader(file)
        gbs_data = []
        lbs_data = []
        for row in reader:
            if int(row["Gloabl_Batch_Size"]) == 128*int(row["No.Processes"]):
                lbs_data.append([int(row["No.Processes"]), float(row["Through_Put"])])
            if int(row["Gloabl_Batch_Size"]) == 1024:
                gbs_data.append([int(row["No.Processes"]), float(row["Through_Put"])])
        
    gbs_data = sorted(gbs_data, key=lambda x: x[0])
    lbs_data = sorted(lbs_data, key=lambda x: x[0])

    base = gbs_data[0][1]
    max_np = gbs_data[-1][0]
    desired = max_np*base
    

    for d in gbs_data:
        if d[0] == 8:
            gbs_data.remove(d)
            break
    for d in lbs_data:
        if d[0] == 8:
            lbs_data.remove(d)
            break
    
    logged_gbs_data = gbs_data  # list(map(lambda x: map(math.log2, x), gbs_data))
    logged_lbs_data = lbs_data  # list(map(lambda x: map(math.log2, x), lbs_data))

    fig, ax = plt.subplots()

    #ax.set_xscale('log', base=2)
    #ax.set_yscale('log', base=2)

    gbs_np, gbs_tp = tuple(zip(*logged_gbs_data))
    ax.plot(gbs_np, gbs_tp, "o", color="red")
    ax.plot(gbs_np, gbs_tp, label="Constant Global Batch Size", color="red")
    lbs_np, lbs_tp = tuple(zip(*logged_lbs_data))
    ax.plot(lbs_np, lbs_tp, "o",  color="orange")
    ax.plot(lbs_np, lbs_tp, label="Constant Local Batch Size", color="orange")
    ax.plot([1, max_np], [base, desired], label="Perfect Linear Scaling", color="green")
    ax.set_xlabel("No. Processes")
    ax.set_ylabel("Throughput (images/s)")
    ax.set_xticks(gbs_np)
    ax.set_title("Archer2 Image Throughput")
    ax.legend()
    fig.savefig("_throughput.png", dpi=300)

if __name__ == "__main__":
    plot_throughput()