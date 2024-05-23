0. Size.Area: int 物体占据的像素数量。
1. Size.MajorAxisLength: float 与物体具有相同归一化第二中心矩的椭圆的主轴长度。
2. Size.MinorAxisLength: float 与区域具有相同归一化第二中心矩的椭圆的次轴长度。
3. Size.Perimeter: float 物体的周长，通过连接边界像素中心的直线近似轮廓，使用4连通性。
4. Shape.Circularity: float 衡量物体形状与圆形的相似度。

5. Shape.Eccentricity: float 用来计算形状的偏心率，即与物体区域具有相同第二矩的椭圆的偏心率。偏心率是焦点距离（焦点之间的距离）与主轴长度的比率。值在 [0, 1) 区间内，当为 0 时，椭圆变为圆形。
6. Shape.EquivalentDiameter: float 与物体面积相同的圆的直径。
7. Shape.Extent: float 物体面积与其轴对齐边界框面积的比率。
8. Shape.MinorMajorAxisRatio: float 形状的长宽比，即与物体区域具有相同第二矩的椭圆的次轴与主轴的比率。

9. Shape.Solidity: float 凸度测量，计算为物体像素数与其凸包像素数的比率。
10. Shape.FSD1 为每个对象计算傅立叶形状描述符。
11. Shape.FSD2 为每个对象计算傅立叶形状描述符。
12. Shape.FSD3 为每个对象计算傅立叶形状描述符。
13. Shape.FSD4 为每个对象计算傅立叶形状描述符。
14. Shape.FSD5 为每个对象计算傅立叶形状描述符。
15. Shape.FSD6 为每个对象计算傅立叶形状描述符。
16. Intensity.Min: float 对象像素的最小强度。
17. Intensity.Max: float 对象像素的最大强度。
18. Intensity.Mean: float 对象像素的平均强度。
19. Intensity.Median: float 对象像素的中位数强度。
20. Intensity.MeanMedianDiff: float 对象像素的平均强度与中位数强度之间的差值。
21. Intensity.Std: float 对象像素强度的标准偏差。
22. Intensity.IQR: float 对象像素强度的四分位距。
23. Intensity.MAD: float 对象像素强度的中位数绝对偏差。
24. Intensity.Skewness: float 对象像素强度的偏度。当所有强度值相等时，该值为0。
25. Intensity.Kurtosis: float 对象像素强度的峰度。当所有值相等时，该值为-3。
26. Intensity.HistEnergy: float 对象像素强度直方图的能量。
27. Intensity.HistEntropy: float 对象像素强度直方图的熵。
28. Gradient.Mag.Mean: float 梯度数据的平均值。
29. Gradient.Mag.Std: float 梯度数据的标准偏差。
30. Gradient.Mag.Skewness: float 梯度数据的偏度。当所有值相等时，该值为0。
31. Gradient.Mag.Kurtosis: float 梯度数据的峰度。当所有值相等时，该值为-3。
32. Gradient.Mag.HistEnergy: float 对象像素的梯度幅度直方图的能量。
33. Gradient.Mag.HistEntropy: float 对象像素的梯度幅度直方图的熵。
34. Gradient.Canny.Sum: float Canny滤波后的梯度数据之和。
35. Gradient.Canny.Mean: float Canny滤波后的梯度数据的平均值。
36. Haralick.ASM.Mean: float
37. Haralick.ASM.Range: float 角二阶矩（ASM）的平均值和范围。它是图像均匀性的度量，计算公式如下：𝐴𝑆𝑀=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝(𝑖,𝑗)²
38. Haralick.Contrast.Mean: float
39. Haralick.Contrast.Range: float 对比度特征的平均值和范围，它度量相邻像素强度的变化量，对于恒定图像为零，随变化增加而增加，计算公式如下：𝐶𝑜𝑛𝑡𝑟𝑎𝑠𝑡=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1(𝑖−𝑗)²𝑝(𝑖,𝑗)
40. Haralick.Correlation.Mean: float
41. Haralick.Correlation.Range: float 相关性特征的平均值和范围，度量相邻像素强度值的相关性，计算公式如下：𝐶𝑜𝑟𝑟𝑒𝑙𝑎𝑡𝑖𝑜𝑛=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝(𝑖,𝑗)[(𝑖−𝜇𝑖)(𝑗−𝜇𝑗)𝜎𝑖𝜎𝑗]
42. Haralick.SumOfSquares.Mean: float
43. Haralick.SumOfSquares.Range: float 平方和特征的平均值和范围，度量方差，计算公式如下： 𝑆𝑢𝑚𝑜𝑓𝑆𝑞𝑢𝑎𝑟𝑒=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1(𝑖−𝜇)²𝑝(𝑖,𝑗)
44. Haralick.IDM.Mean: float
45. Haralick.IDM.Range: float 逆差矩（IDM）的平均值和范围，度量均匀性，计算公式如下：𝐼𝐷𝑀=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−111+(𝑖−𝑗)²𝑝(𝑖,𝑗)
46. Haralick.SumAverage.Mean: float
47. Haralick.SumAverage.Range: float 和平均特征的平均值和范围，计算公式如下：𝑆𝑢𝑚𝐴𝑣𝑒𝑟𝑎𝑔𝑒=∑𝑘=22𝑙𝑒𝑣𝑒𝑙𝑠𝑘𝑝𝑥+𝑦(𝑘),其中𝑝𝑥+𝑦(𝑘)=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝛿𝑖+𝑗,𝑘𝑝((𝑖)𝑝(𝑖,𝑗)
48. Haralick.SumVariance.Mean: float
49. Haralick.SumVariance.Range: float 和方差特征的平均值和范围，计算公式如下：𝑆𝑢𝑚𝑉𝑎𝑟𝑖𝑎𝑛𝑐𝑒=∑𝑘=22𝑙𝑒𝑣𝑒𝑙𝑠(𝑘−𝑆𝑢𝑚𝐸𝑛𝑡𝑟𝑜𝑝𝑦)𝑝𝑥+𝑦(𝑘)
50. Haralick.SumEntropy.Mean: float
51. Haralick.SumEntropy.Range: float 和熵特征的平均值和范围，计算公式如下：𝑆𝑢𝑚𝐸𝑛𝑡𝑟𝑜𝑝𝑦=−∑𝑘=22𝑙𝑒𝑣𝑒𝑙𝑠𝑝𝑥+𝑦(𝑘)log(𝑝𝑥+𝑦(𝑘))
52. Haralick.Entropy.Mean: float
53. Haralick.Entropy.Range: float 熵特征的平均值和范围，计算公式如下：𝐸𝑛𝑡𝑟𝑜𝑝𝑦=−∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝(𝑖,𝑗)log(𝑝(𝑖,𝑗))
54. Haralick.DifferenceVariance.Mean: float
55. Haralick.DifferenceVariance.Range: float 差异方差特征的平均值和范围，计算公式如下：𝐷𝑖𝑓𝑓𝑒𝑟𝑒𝑛𝑐𝑒𝑉𝑎𝑟𝑖𝑎𝑛𝑐𝑒=方差 𝑝𝑥−𝑦, 其中𝑝𝑥−𝑦(𝑘)=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝛿|𝑖−𝑗|,𝑘𝑝(𝑖,𝑗)
56. Haralick.DifferenceEntropy.Mean: float
57. Haralick.DifferenceEntropy.Range: float 差异熵特征的平均值和范围，计算公式如下：𝐷𝑖𝑓𝑓𝑒𝑟𝑒𝑛𝑐𝑒𝐸𝑛𝑡𝑟𝑜𝑝𝑦=差异熵 𝑝𝑥−𝑦
58. Haralick.IMC1.Mean: float
59. Haralick.IMC1.Range: float 第一信息测度相关性特征的平均值和范围，计算公式如下：𝐼𝑀𝐶1=𝐻𝑋𝑌−𝐻𝑋𝑌1/max(𝐻𝑋,𝐻𝑌), 其中𝐻𝑋𝑌, 𝐻𝑋𝑌1, 𝐻𝑋, 和𝐻𝑌 的计算依赖于𝑝𝑥(𝑖), 𝑝𝑦(𝑗)
60. Haralick.IMC2.Mean: float
61. Haralick.IMC2.Range: float 第二信息测度相关性特征的平均值和范围，计算公式如下：
62. Intensity.WholeNucleus: 细胞核内像素强度的平均值
63. Intensity.part05: 将细胞核从内到外按照细胞的轮廓放缩生成5个部分，从内到外01～05，01表示最内层，05表示最外层。第5部分的平均强度。
64. Intensity.part04: 将细胞核从内到外按照细胞的轮廓放缩生成5个部分，从内到外01～05，01表示最内层，05表示最外层。第4部分的平均强度。
65. Intensity.part03: 将细胞核从内到外按照细胞的轮廓放缩生成5个部分，从内到外01～05，01表示最内层，05表示最外层。第3部分的平均强度。
66. Intensity.part02: 将细胞核从内到外按照细胞的轮廓放缩生成5个部分，从内到外01～05，01表示最内层，05表示最外层。第2部分的平均强度。
67. Intensity.part01: 将细胞核从内到外按照细胞的轮廓放缩生成5个部分，从内到外01～05，01表示最内层，05表示最外层。第1部分的平均强度。
68. Intensity.distribution.part05: 将细胞核从内到外按照细胞的轮廓放缩生成5个部分，从内到外01～05，01表示最内层，05表示最外层。第5部分的平均强度占总体平均强度的占比，代表像素点聚集在第5部分。
69. Intensity.distribution.part04: 将细胞核从内到外按照细胞的轮廓放缩生成5个部分，从内到外01～05，01表示最内层，05表示最外层。第4部分的平均强度占总体平均强度的占比，代表像素点聚集在第4部分。
70. Intensity.distribution.part03: 将细胞核从内到外按照细胞的轮廓放缩生成5个部分，从内到外01～05，01表示最内层，05表示最外层。第3部分的平均强度占总体平均强度的占比，代表像素点聚集在第3部分。
71. Intensity.distribution.part02: 将细胞核从内到外按照细胞的轮廓放缩生成5个部分，从内到外01～05，01表示最内层，05表示最外层。第2部分的平均强度占总体平均强度的占比，代表像素点聚集在第2部分。
72. Intensity.distribution.part01: 将细胞核从内到外按照细胞的轮廓放缩生成5个部分，从内到外01～05，01表示最内层，05表示最外层。第1部分的平均强度占总体平均强度的占比，代表像素点聚集在第1部分。