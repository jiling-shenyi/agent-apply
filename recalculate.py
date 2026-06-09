import json
import matplotlib.pyplot as plt

def recalculate(input_file, output_json):
    # 1. 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", [])

    # 2. 初始化统计字典
    agent_stats = {}
    difficulty_stats = {}

    for r in results:
        # Agent 统计累加
        agent = r.get("agent_name", "unknown")
        if agent not in agent_stats:
            agent_stats[agent] = {"total": 0, "success": 0, "rounds": 0, "time": 0.0}
        agent_stats[agent]["total"] += 1
        agent_stats[agent]["success"] += r.get("success", 0.0)
        agent_stats[agent]["rounds"] += r.get("interaction_rounds", 0)
        agent_stats[agent]["time"] += r.get("time_used", 0.0)

        # 难度 统计累加
        diff = r.get("test_case", {}).get("difficulty", "unknown")
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"total": 0, "success": 0, "rounds": 0, "time": 0.0}
        difficulty_stats[diff]["total"] += 1
        difficulty_stats[diff]["success"] += r.get("success", 0.0)
        difficulty_stats[diff]["rounds"] += r.get("interaction_rounds", 0)
        difficulty_stats[diff]["time"] += r.get("time_used", 0.0)

    # 3. 计算各项指标的平均值
    def compute_avgs(stats_dict):
        final_stats = {}
        for k, v in stats_dict.items():
            t = v["total"]
            final_stats[k] = {
                "total_tests": t,
                "grade": v["success"],
                "success_rate": (v["success"] / t) if t > 0 else 0,
                "avg_interaction_rounds": (v["rounds"] / t) if t > 0 else 0,
                "avg_time_used": (v["time"] / t) if t > 0 else 0
            }
        return final_stats

    final_agent_stats = compute_avgs(agent_stats)
    final_diff_stats = compute_avgs(difficulty_stats)

    # 4. 将重新计算的结果更新至原数据结构，并保存为新的 JSON 文件
    data["analysis"]["by_agent"] = final_agent_stats
    data["analysis"]["by_difficulty"] = final_diff_stats

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据统计完成，已保存至: {output_json}")
    """
    # 5. 绘制统计图表
    # 设置图表大小并创建 2x3 的子图 (上排为 Agent，下排为难度)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    def plot_bar(ax, data_dict, metric, title, ylabel, color):
        # 排序以确保图表柱状顺序一致
        keys = sorted(data_dict.keys())
        values = [data_dict[k][metric] for k in keys]
        
        bars = ax.bar(keys, values, color=color, edgecolor='black', alpha=0.8)
        ax.set_title(title, pad=10, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=15) # 倾斜x轴标签，避免文字重叠
        
        # 在每个柱子顶部添加数值标签
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, 
                    yval + (max(values)*0.02), 
                    f"{yval:.1f}", ha='center', va='bottom', fontsize=10)

    # 绘制 Agent 统计
    plot_bar(axes[0, 0], final_agent_stats, "success_rate", "Success Rate by Agent", "Success Rate (%)", '#4C72B0')
    plot_bar(axes[0, 1], final_agent_stats, "avg_interaction_rounds", "Avg Interaction Rounds by Agent", "Rounds", '#55A868')
    plot_bar(axes[0, 2], final_agent_stats, "avg_time_used", "Avg Time Used by Agent", "Time (s)", '#C44E52')

    # 绘制 难度 统计
    plot_bar(axes[1, 0], final_diff_stats, "success_rate", "Success Rate by Difficulty", "Success Rate (%)", '#4C72B0')
    plot_bar(axes[1, 1], final_diff_stats, "avg_interaction_rounds", "Avg Interaction Rounds by Difficulty", "Rounds", '#55A868')
    plot_bar(axes[1, 2], final_diff_stats, "avg_time_used", "Avg Time Used by Difficulty", "Time (s)", '#C44E52')

    # 自动调整间距并保存图表
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"图表绘制完成，已保存至: {output_image}")
    """

if __name__ == "__main__":
    input_file = "evaluation_results_20260609_110912.json"
    output_json = "20260609_110912.json"
    
    recalculate(input_file, output_json)