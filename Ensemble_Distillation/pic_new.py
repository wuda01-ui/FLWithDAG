import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("./alldata.xlsx")

# 设置字体
font = {'family' : 'Times New Roman',
        'size'   : 12}
plt.rc('font', **font)
plt.figure(figsize=(6,6))

# 绘图
# sns.lineplot(x = df["Rounds"].tolist(), y = df["Att+Ce+Adv"].tolist(), label='Att+Ce+Adv', 
#              linewidth=2.5, color="#c35241", linestyle = 'solid')
# sns.lineplot(x = df["Rounds"].tolist(), y = df["Att+Ce"].tolist(), label='Att+Ce', 
#              linewidth=2.5, color="#147dbb", linestyle = 'dashed')
# sns.lineplot(x = df["Rounds"].tolist(), y = df["Att+Adv"].tolist(), label='Att+Adv', 
#              linewidth=2.5, color="#e49440", linestyle = '--')
# sns.lineplot(x = df["Rounds"].tolist(), y = df["Ce+Adv"].tolist(), label='Ce+Adv', 
#              linewidth=2.5, color="#318e5c", linestyle = '-.')
# sns.lineplot(x = df["Rounds"].tolist(), y = df["Ce"].tolist(), label='Ce', 
#              linewidth=2.5, color="#8d8bbc", linestyle = 'dotted')
# sns.lineplot(x = df["Rounds"].tolist(), y = df["Adv"].tolist(), label='Adv', 
#              linewidth=2.5, color="#00ffff", linestyle = 'dotted')
sns.lineplot(x = df["Iterations"].tolist(), y = df["a"].tolist(), label='k=3,m=10%', 
             linewidth=1.5, color="#c35241", linestyle = 'solid')
sns.lineplot(x = df["Iterations"].tolist(), y = df["b"].tolist(), label='k=5,m=10%', 
             linewidth=1.5, color="#147dbb", linestyle = 'solid')
sns.lineplot(x = df["Iterations"].tolist(), y = df["c"].tolist(), label='k=7,m=10%', 
             linewidth=1.5, color="#e49440", linestyle = 'solid')
sns.lineplot(x = df["Iterations"].tolist(), y = df["d"].tolist(), label='k=3,m=40%', 
             linewidth=1.5, color="#318e5c", linestyle = 'dashed')
sns.lineplot(x = df["Iterations"].tolist(), y = df["e"].tolist(), label='k=5,m=40%', 
             linewidth=1.5, color="#8d8bbc", linestyle = 'dashed')
sns.lineplot(x = df["Iterations"].tolist(), y = df["f"].tolist(), label='k=7,m=40%', 
             linewidth=1.5, color="#00ffff", linestyle = 'dashed')

# 设置背景样式
sns.set_style("whitegrid") 

# 添加标题和标签
# plt.title("Title", fontweight='bold', fontsize=14)
plt.xlabel("Iterations", fontsize=18)
plt.ylabel("Accuracy(%)", fontsize=18)


# 添加图例
# plt.legend(loc='lower right', frameon=True, fontsize=10)
plt.legend(loc='right', frameon=True, fontsize=14)
plt.grid()

# 设置刻度字体和范围
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0, 2000)
plt.ylim(0, 90)



# 设置坐标轴样式
for spine in plt.gca().spines.values():
    spine.set_edgecolor("#3d3d3d")
    spine.set_linewidth(1.5)
    

plt.savefig('CIFAR.png', dpi=1500, bbox_inches='tight')

