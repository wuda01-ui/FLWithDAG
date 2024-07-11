import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.rc('font',family='Times New Roman',size=12)
Client= ["0", "1", "2", "3", "4"]
Label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

harvest = np.array(      [[156,  709,  301,  2629,  20,  651,  915,  113,  180,  2133] ,
  [1771,  2695,  1251,  1407,  665,  314,  1419,  3469, 0, 0],
   [236,  15,  1715,  76,  1304,  34,  1773,  75,  3289,  2360] ,
   [2809,  575,  157,  853,  2555,  2557,  203,  1213, 0, 0 ],
   [28,  1006,  1576,  35,  456,  1444,  690,  130,  1531,  507]])

# 使用Seaborn的heatmap函数绘制热力图
sns.heatmap(harvest, annot=True,fmt='d', cmap='GnBu', xticklabels=Label, yticklabels=Client,cbar=False)


plt.xlabel("Label",fontsize=15)
plt.ylabel("Client ID",fontsize=15)
plt.tight_layout()
 
# 保存图像为PNG文件
plt.savefig('seaborn_heatmap0.5.png')


