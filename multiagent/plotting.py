
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from random import randint, gauss



agents = ["Expectimax","Alpha-beta","MC Tree Search"]
# ab_means = np.array([1281.6,1533.95,503.7,843.45])
# exp_means = [1375.9,1753.8,397.5,983.75]
# mc_means = [1285.3,1704.65,1218.15,1470.3]

ab_small_rand  = np.array([1170.0, 1343.0, 1365.0, 1300.0, 1147.0, 1150.0, 1146.0, 1350.0, 1353.0, 1060.0, 1129.0, 1370.0, 948.0, 1162.0, 1313.0, 1568.0, 1160.0, 1362.0, 1347.0, 1349.0])
ab_med_rand = np.array([1411.0, 303.0, 1379.0, 1638.0, 1848.0, 1910.0, 1124.0, 1491.0, 1504.0, 1706.0, 1652.0, 1652.0, 1731.0, 1865.0, 2074.0, 1723.0, 1845.0, 2003.0, 1866.0, -216.0])
ab_small_adv = np.array([-387.0, 1511.0, 1336.0, 1728.0, -490.0, 1561.0, 1020.0, 1358.0, 228.0, -437.0, -222.0, -341.0, -377.0, 1305.0, -368.0, -368.0, 1720.0, -131.0, -367.0, 1535.0])
#ab_med_adv = np.array([548.0, 390.0, -265.0, 1683.0, 1501.0, 1034.0, 1891.0, -261.0, 437.0, 1885.0, 1486.0, -359.0, 781.0, -278.0, 1205.0, 1618.0, 2083.0, 1417.0, -143.0, 476.0])
ab_med_adv = np.array([548.0, 330.0, -265.0, 1683.0, 1501.0, 1034.0, 1891.0, -261.0, 337.0, 1885.0, 1486.0, -359.0, 681.0, -278.0, 1205.0, 1618.0, 2083.0, 1417.0, -143.0, 476.0])


exp_small_rand = np.array([1318.0, 1742.0, 1724.0, 1377.0, 316.0, 1551.0, 1376.0, 1377.0, 1668.0, 1289.0, 969.0, 1259.0, 1371.0, 1571.0, 1357.0, 1381.0, 1307.0, 1744.0, 1132.0, 1539.0])
exp_med_rand = np.array([1573.0, 1891.0, 1935.0, 1261.0, 1634.0, 1678.0, 1864.0, 1801.0, 1873.0, 1844.0, 1891.0, 1630.0, 2058.0, 1646.0, 1808.0, 2104.0, 2112.0, 1883.0, 1597.0, 1123.0])
exp_small_adv = np.array([1679.0, -354.0, -390.0, -376.0, 133.0, 721.0, -378.0, -375.0, 423.0, -374.0, 122.0, 1561.0, -374.0, 1564.0, 1747.0, -383.0, -375.0, 1552.0, 220.0, 1727.0])
exp_med_adv = np.array([209.0, 1875.0, 156.0, 1672.0, 1455.0, 1889.0, 676.0, -94.0, 1882.0, 1885.0, 10.0, 2111.0, 2.0, -321.0, -104.0, 894.0, 1863.0, 1717.0, 2106.0, -138.0])

mc_small_rand = np.array([1544.0, 1357.0, 1522.0, 1353.0, -392.0, 1508.0, 1162.0, 1148.0, 1247.0, 1503.0, 1321.0, 1350.0, 1714.0, 1500.0, 1348.0, 1368.0, 1315.0, 1265.0, 1246.0, 1477.0])
mc_med_rand = np.array([1814.0, 1490.0, 1523.0, 1919.0, 1708.0, 1700.0, 1503.0, 1688.0, 2079.0, 1899.0, 1676.0, 1551.0, 1697.0, 1699.0, 1499.0, 1904.0, 1657.0, 1655.0, 1488.0, 1894.0])
mc_small_adv = np.array([1302.0, 1518.0, 1272.0, 1317.0, 1351.0, 1452.0, 1502.0, -343.0, 1344.0, 1266.0, 1473.0, 674.0, 175.0, 591.0, 1692.0, 1715.0, 1496.0, 1448.0, 1828.0, 1540.0])
mc_med_adv = np.array([1229.0, 1727.0, 1442.0, 1524.0, 332.0, 1704.0, 1691.0, 1497.0, 1873.0, 1648.0, 1535.0, 1716.0, 1717.0, 1886.0, 1496.0, 1451.0, -189.0, 1982.0, 1524.0, 1731.0])

runs = [ab_small_rand, ab_med_rand, ab_small_adv, ab_med_adv,
        exp_small_rand, exp_med_rand, exp_small_adv, exp_med_adv,
        mc_small_rand, mc_med_rand, mc_small_adv, mc_med_adv]

std_dev = [arr.std() for arr in runs]
mean_value = [arr.mean() for arr in runs]
for x, y in zip(mean_value, std_dev):
    print(x,y)


plt.figure()
plt.title("Comparison: Small layout with Adversarial ghosts")
plt.ylabel("Score")
custom_palette = {agents[0]: "dodgerblue", agents[1]: "coral", agents[2] :"mediumseagreen"}

# # sns.boxplot(agents,[ab_small_adv,exp_small_adv,mc_small_adv],palette=custom_palette)
# sns.boxplot([ab_small_adv,exp_small_adv,mc_small_adv],palette=custom_palette)
# plt.show()
# plt.savefig("figures/box_plot_small_adv.png")

df = pd.DataFrame({
    #     "Agents" : [1, 2, 3],
    #     "Score" : [ab_small_adv, exp_small_adv, mc_small_adv]

    agents[0]: exp_small_adv,
    agents[1]: ab_small_adv,
    agents[2]: mc_small_adv

})

sns_plot = sns.boxplot(data=df)
plt.show()
# sns_plot.savefig('box_plot_snall_Adv.png')

# plt.savefig("box_plot_small_adv.png")



plt.figure()
plt.title("Comparison: Medium layout with Adversarial ghosts")
plt.ylabel("Score")
custom_palette = {agents[0]: "dodgerblue", agents[1]: "coral", agents[2] :"mediumseagreen"}

df = pd.DataFrame({
    #     "Agents" : [1, 2, 3],
    #     "Score" : [ab_small_adv, exp_small_adv, mc_small_adv]

    agents[0]: exp_med_adv,
    agents[1]: ab_med_adv,
    agents[2]: mc_med_adv

})

sns_plot = sns.boxplot(data=df)
# sns_plot.savefig("output.png")
# sns.plt.savefig('box_plot_medium_Adv.png')
plt.show()
# plt.savefig("box_plot_med_adv.png")
