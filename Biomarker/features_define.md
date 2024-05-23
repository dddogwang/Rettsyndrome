0. Size.Area: int Number of pixels the object occupies.

1. Size.MajorAxisLength: float The length of the major axis of the ellipse that has the same normalized second central moments as the object.
2. Size.MinorAxisLength: float The length of the minor axis of the ellipse that has the same normalized second central moments as the region.
3. Size.Perimeter: float Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
4. Shape.Circularity: float A measure of how similar the shape of an object is to the circle
5. Shape.Eccentricity: float A measure of aspect ratio computed to be the eccentricity of the ellipse that has the same second-moments as the object region. Eccentricity of an ellipse is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
6. Shape.EquivalentDiameter: float The diameter of a circle with the same area as the object.
7. Shape.Extent: float Ratio of area of the object to its axis-aligned bounding box.
8. Shape.MinorMajorAxisRatio: float A measure of aspect ratio. Ratio of minor to major axis of the ellipse that has the same second-moments as the object region
9. Shape.Solidity: float A measure of convexity computed as the ratio of the number of pixels in the object to that of its convex hull.
10. Shape.FSD-1 Calculates Fourier shape descriptors for each objects. 
11. Shape.FSD-2 Calculates Fourier shape descriptors for each objects. 
12. Shape.FSD-3 Calculates Fourier shape descriptors for each objects. 
13. Shape.FSD-4 Calculates Fourier shape descriptors for each objects. 
14. Shape.FSD-5 Calculates Fourier shape descriptors for each objects. 
15. Shape.FSD-6 Calculates Fourier shape descriptors for each objects. 
16. Intensity.Min: float Minimum intensity of object pixels.
17. Intensity.Max: float Maximum intensity of object pixels.
18. Intensity.Mean: float Mean intensity of object pixels
19. Intensity.Median: float Median intensity of object pixels
20. Intensity.MeanMedianDiff: float Difference between mean and median intensities of object pixels.
21. Intensity.Std: float Standard deviation of the intensities of object pixels
22. Intensity.IQR: float Inter-quartile range of the intensities of object pixels
23. Intensity.MAD: float Median absolute deviation of the intensities of object pixels
24. Intensity.Skewness: float Skewness of the intensities of object pixels. Value is 0 when all intensity values are equal.
25. Intensity.Kurtosis: float Kurtosis of the intensities of object pixels. Value is -3 when all values are equal.
26. Intensity.HistEnergy: float Energy of the intensity histogram of object pixels
27. Intensity.HistEntropy: float Entropy of the intensity histogram of object pixels.
28. Gradient.Mag.Mean: float Mean of gradient data.
29. Gradient.Mag.Std: float Standard deviation of gradient data.
30. Gradient.Mag.Skewness: float Skewness of gradient data. Value is 0 when all values are equal.
31. Gradient.Mag.Kurtosis: float Kurtosis of gradient data. Value is -3 when all values are equal.
32. Gradient.Mag.HistEnergy: float Energy of the gradient magnitude histogram of object pixels
33. Gradient.Mag.HistEnergy: float Entropy of the gradient magnitude histogram of object pixels.
34. Gradient.Canny.Sum: float Sum of canny filtered gradient data.
35. Gradient.Canny.Mean: float Mean of canny filtered gradient data.
36. Haralick.ASM.Mean: float
37. Haralick.ASM.Range: float Mean and range of the angular second moment (ASM) feature for GLCMs of all offsets. It is a measure of image homogeneity and is computed as follows: 𝐴𝑆𝑀=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝(𝑖,𝑗)2
38. Haralick.Contrast.Mean: float
39. Haralick.Contrast.Range: float Mean and range of the Contrast feature for GLCMs of all offsets. It is a measure of the amount of variation between intensities of neighboiring pixels. It is equal to zero for a constant image and increases as the amount of variation increases. It is computed as follows: 𝐶𝑜𝑛𝑡𝑟𝑎𝑠𝑡=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1(𝑖−𝑗)2𝑝(𝑖,𝑗)
40. Haralick.Correlation.Mean: float
41. Haralick.Correlation.Range: float Mean and range of the Correlation feature for GLCMs of all offsets. It is a measure of correlation between the intensity values of neighboring pixels. It is computed as follows: 𝐶𝑜𝑟𝑟𝑒𝑙𝑎𝑡𝑖𝑜𝑛=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝(𝑖,𝑗)[(𝑖−𝜇𝑖)(𝑗−𝜇𝑗)𝜎𝑖𝜎𝑗]
42. Haralick.SumOfSquares.Mean: float
43. Haralick.SumOfSquares.Range: float Mean and range of the SumOfSquares feature for GLCMs of all offsets. It is a measure of variance and is computed as follows: 𝑆𝑢𝑚𝑜𝑓𝑆𝑞𝑢𝑎𝑟𝑒=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1(𝑖−𝜇)2𝑝(𝑖,𝑗)
44. Haralick.IDM.Mean: float
45. Haralick.IDM.Range: float Mean and range of the inverse difference moment (IDM) feature for GLCMS of all offsets. It is a measure of homogeneity and is computed as follows: 𝐼𝐷𝑀=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−111+(𝑖−𝑗)2𝑝(𝑖,𝑗)
46. Haralick.SumAverage.Mean: float
47. Haralick.SumAverage.Range: float Mean and range of sum average feature for GLCMs of all offsets. It is computed as follows: 𝑆𝑢𝑚𝐴𝑣𝑒𝑟𝑎𝑔𝑒=∑𝑘=22𝑙𝑒𝑣𝑒𝑙𝑠𝑘𝑝𝑥+𝑦(𝑘),𝑤ℎ𝑒𝑟𝑒𝑝𝑥+𝑦(𝑘)=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝛿𝑖+𝑗,𝑘𝑝(𝑖,𝑗)𝛿𝑚,𝑛={1when 𝑚=𝑛0when 𝑚≠𝑛 
48. Haralick.SumVariance.Mean: float
49. Haralick.SumVariance.Range: float Mean and range of sum variance feature for the GLCMS of all offsets. It is computed as follows: 𝑆𝑢𝑚𝑉𝑎𝑟𝑖𝑎𝑛𝑐𝑒=∑𝑘=22𝑙𝑒𝑣𝑒𝑙𝑠(𝑘−𝑆𝑢𝑚𝐸𝑛𝑡𝑟𝑜𝑝𝑦)𝑝𝑥+𝑦(𝑘)
50. Haralick.SumEntropy.Mean: float
51. Haralick.SumEntropy.Range: float Mean and range of the sum entropy features for GLCMS of all offsets. It is computed as follows: 𝑆𝑢𝑚𝐸𝑛𝑡𝑟𝑜𝑝𝑦=−∑𝑘=22𝑙𝑒𝑣𝑒𝑙𝑠𝑝𝑥+𝑦(𝑘)log⁡(𝑝𝑥+𝑦(𝑘))
52. Haralick.Entropy.Mean: float
53. Haralick.Entropy.Range: float Mean and range of the entropy features for GLCMs of all offsets. It is computed as follows: 𝐸𝑛𝑡𝑟𝑜𝑝𝑦=−∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝(𝑖,𝑗)log⁡(𝑝(𝑖,𝑗))
54. Haralick.DifferenceVariance.Mean: float
55. Haralick.DifferenceVariance.Range: float Mean and Range of the difference variance feature of GLCMs of all offsets. It is computed as follows: 𝐷𝑖𝑓𝑓𝑒𝑟𝑒𝑛𝑐𝑒𝑉𝑎𝑟𝑖𝑎𝑛𝑐𝑒=variance of 𝑝𝑥−𝑦,𝑤ℎ𝑒𝑟𝑒𝑝𝑥−𝑦(𝑘)=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝛿|𝑖−𝑗|,𝑘𝑝(𝑖,𝑗)
56. Haralick.DifferenceEntropy.Mean: float
57. Haralick.DifferenceEntropy.Range: float Mean and range of the difference entropy feature for GLCMS of all offsets. It is computed as follows: 𝐷𝑖𝑓𝑓𝑒𝑟𝑒𝑛𝑐𝑒𝐸𝑛𝑡𝑟𝑜𝑝𝑦=entropy of 𝑝𝑥−𝑦
58. Haralick.IMC1.Mean: float
59. Haralick.IMC1.Range: float Mean and range of the first information measure of correlation feature for GLCMs of all offsets. It is computed as follows: 𝐼𝑀𝐶1=𝐻𝑋𝑌−𝐻𝑋𝑌1max(𝐻𝑋,𝐻𝑌),𝑤ℎ𝑒𝑟𝑒𝐻𝑋𝑌=−∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝(𝑖,𝑗)log⁡(𝑝(𝑖,𝑗))𝐻𝑋𝑌1=−∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝(𝑖,𝑗)log⁡(𝑝𝑥(𝑖)𝑝𝑦(𝑗))𝐻𝑋=−∑𝑖=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝𝑥(𝑖)log⁡(𝑝𝑥(𝑖))𝐻𝑌=−∑𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝𝑦(𝑗)log⁡(𝑝𝑦(𝑗))𝑝𝑥(𝑖)=∑𝑗=1𝑙𝑒𝑣𝑒𝑙𝑠𝑝(𝑖,𝑗)𝑝𝑦(𝑗)=∑𝑗=1𝑙𝑒𝑣𝑒𝑙𝑠𝑝(𝑖,𝑗)
60. Haralick.IMC2.Mean: float
61. Haralick.IMC2.Range: float Mean and range of the second information measure of correlation feature for GLCMs of all offsets. It is computed as follows:

62. Intensity.WholeNucleus: Nucleus whole intensity mean.
63. Intensity.part05: Nucleus part 5 intensity mean.
64. Intensity.part04: Nucleus part 4 intensity mean.
65. Intensity.part03: Nucleus part 3 intensity mean.
66. Intensity.part02: Nucleus part 2 intensity mean.
67. Intensity.part01: Nucleus part 1 intensity mean.
68. Intensity.distribution.part05: Nucleus part 5 intensity / Nucleus whole intensity mean.
69. Intensity.distribution.part04: Nucleus part 4 intensity / Nucleus whole intensity mean.
70. Intensity.distribution.part03: Nucleus part 3 intensity / Nucleus whole intensity mean.
71. Intensity.distribution.part02: Nucleus part 2 intensity / Nucleus whole intensity mean.
72. Intensity.distribution.part01: Nucleus part 1 intensity / Nucleus whole intensity mean.