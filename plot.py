import json
import matplotlib.pyplot as plt
import numpy as np

def generate_grouped_charts(input_file, output_image):
    # 1. 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", [])

    # 2. 数据处理：针对每个 agent 计算 agent_killer, too_hard 和 sum(总计) 的数据
    agent_data = {}
    for r in results:
        agent = r.get("agent_name", "unknown")
        diff = r.get("test_case", {}).get("difficulty", "unknown")
        
        # 初始化字典结构
        if agent not in agent_data:
            agent_data[agent] = {
                "agent_killer": {"success": 0, "rounds": 0, "time": 0.0, "total": 0},
                "too_hard": {"success": 0, "rounds": 0, "time": 0.0, "total": 0},
                "sum": {"success": 0, "rounds": 0, "time": 0.0, "total": 0}
            }
        
        # 累加各个难度的数据
        if diff in ["agent_killer", "too_hard"]:
            agent_data[agent][diff]["total"] += 1
            agent_data[agent][diff]["success"] += r.get("success", 0.0)
            agent_data[agent][diff]["rounds"] += r.get("interaction_rounds", 0)
            agent_data[agent][diff]["time"] += r.get("time_used", 0.0)
            
        # 累加 sum (整体指标) 的数据
        agent_data[agent]["sum"]["total"] += 1
        agent_data[agent]["sum"]["success"] += r.get("success", 0.0)
        agent_data[agent]["sum"]["rounds"] += r.get("interaction_rounds", 0)
        agent_data[agent]["sum"]["time"] += r.get("time_used", 0.0)

    # 3. 计算最终作图需要的坐标轴数据
    metrics = {"success_rate": {}, "avg_rounds": {}, "avg_time": {}}
    difficulties = ["agent_killer", "too_hard", "sum"]

    for m in metrics:
        for d in difficulties:
            metrics[m][d] = []

    # 提取有固定顺序的 Agent 名称
    agents = list(agent_data.keys())

    # 计算均值和成功率并填入列表
    for agent in agents:
        for d in difficulties:
            stats = agent_data[agent][d]
            t = stats["total"]
            metrics["success_rate"][d].append(stats["success"] / t if t > 0 else 0)
            metrics["avg_rounds"][d].append(stats["rounds"] / t if t > 0 else 0)
            metrics["avg_time"][d].append(stats["time"] / t if t > 0 else 0)

    # 4. 绘图准备
    fig, axes = plt.subplots(3, 1, figsize=(10, 16)) # 创建3个纵向排列的子图
    x = np.arange(len(agents))
    width = 0.25 # 柱子宽度，三个柱子刚好占 0.75

    # 定义统一的绘图函数
    def plot_grouped_bar(ax, metric_data, title, ylabel):
        # 绘制三组并排的柱子
        bars1 = ax.bar(x - width, metric_data["agent_killer"], width, label='agent_killer', color='#4C72B0', edgecolor='black', alpha=0.8)
        bars2 = ax.bar(x, metric_data["too_hard"], width, label='too_hard', color='#DD8452', edgecolor='black', alpha=0.8)
        bars3 = ax.bar(x + width, metric_data["sum"], width, label='sum (Overall)', color='#55A868', edgecolor='black', alpha=0.8)
        
        ax.set_title(title, pad=15, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(agents, fontsize=11)
        ax.legend() # 显示图例
        ax.grid(axis='y', linestyle='--', alpha=0.7) # 添加横向网格线以便对齐数值
        
        # 在柱子顶部添加数值标签
        max_y = max(max(metric_data["agent_killer"]), max(metric_data["too_hard"]), max(metric_data["sum"]))
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, 
                        yval + max_y * 0.02, # 在柱子上浮2%的空隙写字
                        f"{yval:.1f}", ha='center', va='bottom', fontsize=10)

    # 5. 依次绘制三张子图
    plot_grouped_bar(axes[0], metrics["success_rate"], "Success Rate by Agent & Difficulty", "Success Rate (%)")
    plot_grouped_bar(axes[1], metrics["avg_rounds"], "Avg Interaction Rounds by Agent & Difficulty", "Rounds")
    plot_grouped_bar(axes[2], metrics["avg_time"], "Avg Time Used by Agent & Difficulty", "Time (s)")

    # 调整布局以防止重叠，并保存图片
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"合并的柱状图生成成功，已保存至: {output_image}")

if __name__ == "__main__":
    generate_grouped_charts("evaluation_results_20260609_182440.json", "20260609_182440.png")