# [Histomicstk.features](https://digitalslidearchive.github.io/HistomicsTK/histomicstk.features.html#module-histomicstk.features)

## Identifier

> Location of the nucleus and its code in the input labeled mask. Columns are prefixed by *Identifier.*. These include …

0. ❌ Identifier.Label (int) - nucleus label in the input labeled mask
1. ❌ Identifier.Xmin (int) - Left bound
2. ❌ Identifier.Ymin (int) - Upper bound
3. ❌ Identifier.Xmax (int) - Right bound
4. ❌ Identifier.Ymax (int) - Lower bound
5. ❌ Identifier.CentroidX (float) - X centroid (columns)
6. ❌ Identifier.CentroidY (float) - Y centroid (rows)
7. ❌ Identifier.WeightedCentroidX (float) - intensity-weighted X centroid
8. ❌ Identifier.WeightedCentroidY (float) - intensity-weighted Y centroid

## Morphometry (size, shape, and orientation) features of the nuclei

> See histomicstk.features.compute_morphometry_features for more details. Feature names prefixed by *Size.*, *Shape.*, or *Orientation.*.

9. ❌ Orientation.Orientation: float

   Angle between the horizontal axis and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.

10. Size.Area: int

    Number of pixels the object occupies.

11. ❌ Size.ConvexHullArea: int

    Number of pixels of convex hull image, which is the smallest convex polygon that encloses the region.

12. Size.MajorAxisLength: float

    The length of the major axis of the ellipse that has the same normalized second central moments as the object.

13. Size.MinorAxisLength: float

    The length of the minor axis of the ellipse that has the same normalized second central moments as the region.

14. Size.Perimeter: float

    Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.

15. Shape.Circularity: float

    A measure of how similar the shape of an object is to the circle

16. Shape.Eccentricity: float

    A measure of aspect ratio computed to be the eccentricity of the ellipse that has the same second-moments as the object region. Eccentricity of an ellipse is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.

17. Shape.EquivalentDiameter: float

    The diameter of a circle with the same area as the object.

18. Shape.Extent: float

    Ratio of area of the object to its axis-aligned bounding box.

19. ❌ Shape.FractalDimension: float

    Minkowski–Bouligand dimension, aka. the box-counting dimension. It is a measure of boundary complexity. See [https://en.wikipedia.org/wiki/Minkowski%E2%80%93Bouligand_dimension](https://en.wikipedia.org/wiki/Minkowski–Bouligand_dimension)

20. Shape.MinorMajorAxisRatio: float

    A measure of aspect ratio. Ratio of minor to major axis of the ellipse that has the same second-moments as the object region

21. Shape.Solidity: float

    A measure of convexity computed as the ratio of the number of pixels in the object to that of its convex hull.

22. ❌ Shape.HuMoments-k: float

    Where k ranges from 1-7 are the 7 Hu moments features. The first six moments are translation, scale and rotation invariant, while the seventh moment flips its sign if the shape is a mirror image. See https://learnopencv.com/shape-matching-using-hu-moments-c-python/


29. ❌ Shape.WeightedHuMoments-k: float

    Same as Hu moments, but instead of using the binary mask, using the intensity image.

## Fourier shape descriptor features

> See histomicstk.features.compute_fsd_features for more details. Feature names are prefixed by *FSD*.

36. Shape.FSD-k

    Calculates Fourier shape descriptors for each objects. see [D. Zhang et al. “A comparative study on shape retrieval using Fourier descriptors with different shape signatures,” In Proc. ICIMADE01, 2001.]()

## Intensity features for the nucleus and cytoplasm channels

> See histomicstk.features.compute_fsd_features for more details. Feature names are prefixed by *Nucleus.Intensity.* for nucleus features and *Cytoplasm.Intensity.* for cytoplasm features.

42. Intensity.Min: float

    Minimum intensity of object pixels.

43. Intensity.Max: float

    Maximum intensity of object pixels.

44. Intensity.Mean: float

    Mean intensity of object pixels

45. Intensity.Median: float

    Median intensity of object pixels

46. Intensity.MeanMedianDiff: float

    Difference between mean and median intensities of object pixels.

47. Intensity.Std: float

    Standard deviation of the intensities of object pixels

48. Intensity.IQR: float

    Inter-quartile range of the intensities of object pixels

49. Intensity.MAD: float

    Median absolute deviation of the intensities of object pixels

50. Intensity.Skewness: float

    Skewness of the intensities of object pixels. Value is 0 when all intensity values are equal.

51. Intensity.Kurtosis: float

    Kurtosis of the intensities of object pixels. Value is -3 when all values are equal.

52. Intensity.HistEnergy: float

    Energy of the intensity histogram of object pixels

53. Intensity.HistEntropy: float

    Entropy of the intensity histogram of object pixels.

## Gradient/edge features for the nucleus and cytoplasm channels

> See histomicstk.features.compute_gradient_features for more details. Feature names are prefixed by *Nucleus.Gradient.* for nucleus features and *Cytoplasm.Gradient.* for cytoplasm features.

54. Gradient.Mag.Mean: float

    Mean of gradient data.

55. Gradient.Mag.Std: float

    Standard deviation of gradient data.

56. Gradient.Mag.Skewness: float

    Skewness of gradient data. Value is 0 when all values are equal.

57. Gradient.Mag.Kurtosis: float

    Kurtosis of gradient data. Value is -3 when all values are equal.

58. Gradient.Mag.HistEnergy: float

    Energy of the gradient magnitude histogram of object pixels

59. Gradient.Mag.HistEnergy: float

    Entropy of the gradient magnitude histogram of object pixels.

60. Gradient.Canny.Sum: float

    Sum of canny filtered gradient data.

61. Gradient.Canny.Mean: float

    Mean of canny filtered gradient data.

## Haralick features for the nucleus and cytoplasm channels

> See histomicstk.features.compute_haralick_features for more details. Feature names are prefixed by *Nucleus.Haralick.* for nucleus features and *Cytoplasm.Haralick.* for cytoplasm features.

62. Haralick.ASM.Mean: float

63. Haralick.ASM.Range: float

    Mean and range of the angular second moment (ASM) feature for GLCMs of all offsets. It is a measure of image homogeneity and is computed as follows:

    𝐴𝑆𝑀=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝(𝑖,𝑗)2

64. Haralick.Contrast.Mean: float

65. Haralick.Contrast.Range: float

    Mean and range of the Contrast feature for GLCMs of all offsets. It is a measure of the amount of variation between intensities of neighboiring pixels. It is equal to zero for a constant image and increases as the amount of variation increases. It is computed as follows:

    𝐶𝑜𝑛𝑡𝑟𝑎𝑠𝑡=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1(𝑖−𝑗)2𝑝(𝑖,𝑗)

66. Haralick.Correlation.Mean: float

67. Haralick.Correlation.Range: float

    Mean and range of the Correlation feature for GLCMs of all offsets. It is a measure of correlation between the intensity values of neighboring pixels. It is computed as follows:

    𝐶𝑜𝑟𝑟𝑒𝑙𝑎𝑡𝑖𝑜𝑛=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝(𝑖,𝑗)[(𝑖−𝜇𝑖)(𝑗−𝜇𝑗)𝜎𝑖𝜎𝑗]

68. Haralick.SumOfSquares.Mean: float

69. Haralick.SumOfSquares.Range: float

    Mean and range of the SumOfSquares feature for GLCMs of all offsets. It is a measure of variance and is computed as follows:

    𝑆𝑢𝑚𝑜𝑓𝑆𝑞𝑢𝑎𝑟𝑒=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1(𝑖−𝜇)2𝑝(𝑖,𝑗)

70. Haralick.IDM.Mean: float

71. Haralick.IDM.Range: float

    Mean and range of the inverse difference moment (IDM) feature for GLCMS of all offsets. It is a measure of homogeneity and is computed as follows:

    𝐼𝐷𝑀=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−111+(𝑖−𝑗)2𝑝(𝑖,𝑗)

72. Haralick.SumAverage.Mean: float

73. Haralick.SumAverage.Range: float

    Mean and range of sum average feature for GLCMs of all offsets. It is computed as follows:

    𝑆𝑢𝑚𝐴𝑣𝑒𝑟𝑎𝑔𝑒=∑𝑘=22𝑙𝑒𝑣𝑒𝑙𝑠𝑘𝑝𝑥+𝑦(𝑘),𝑤ℎ𝑒𝑟𝑒𝑝𝑥+𝑦(𝑘)=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝛿𝑖+𝑗,𝑘𝑝(𝑖,𝑗)𝛿𝑚,𝑛={1when 𝑚=𝑛0when 𝑚≠𝑛

74. Haralick.SumVariance.Mean: float

75. Haralick.SumVariance.Range: float

    Mean and range of sum variance feature for the GLCMS of all offsets. It is computed as follows:

    𝑆𝑢𝑚𝑉𝑎𝑟𝑖𝑎𝑛𝑐𝑒=∑𝑘=22𝑙𝑒𝑣𝑒𝑙𝑠(𝑘−𝑆𝑢𝑚𝐸𝑛𝑡𝑟𝑜𝑝𝑦)𝑝𝑥+𝑦(𝑘)

76. Haralick.SumEntropy.Mean: float

77. Haralick.SumEntropy.Range: float

    Mean and range of the sum entropy features for GLCMS of all offsets. It is computed as follows:

    𝑆𝑢𝑚𝐸𝑛𝑡𝑟𝑜𝑝𝑦=−∑𝑘=22𝑙𝑒𝑣𝑒𝑙𝑠𝑝𝑥+𝑦(𝑘)log⁡(𝑝𝑥+𝑦(𝑘))

78. Haralick.Entropy.Mean: float

79. Haralick.Entropy.Range: float

    Mean and range of the entropy features for GLCMs of all offsets. It is computed as follows:

    𝐸𝑛𝑡𝑟𝑜𝑝𝑦=−∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝(𝑖,𝑗)log⁡(𝑝(𝑖,𝑗))

80. Haralick.DifferenceVariance.Mean: float

81. Haralick.DifferenceVariance.Range: float

    Mean and Range of the difference variance feature of GLCMs of all offsets. It is computed as follows:

    𝐷𝑖𝑓𝑓𝑒𝑟𝑒𝑛𝑐𝑒𝑉𝑎𝑟𝑖𝑎𝑛𝑐𝑒=variance of 𝑝𝑥−𝑦,𝑤ℎ𝑒𝑟𝑒𝑝𝑥−𝑦(𝑘)=∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝛿|𝑖−𝑗|,𝑘𝑝(𝑖,𝑗)

82. Haralick.DifferenceEntropy.Mean: float

83. Haralick.DifferenceEntropy.Range: float

    Mean and range of the difference entropy feature for GLCMS of all offsets. It is computed as follows:

    𝐷𝑖𝑓𝑓𝑒𝑟𝑒𝑛𝑐𝑒𝐸𝑛𝑡𝑟𝑜𝑝𝑦=entropy of 𝑝𝑥−𝑦

84. Haralick.IMC1.Mean: float

85. Haralick.IMC1.Range: float

    Mean and range of the first information measure of correlation feature for GLCMs of all offsets. It is computed as follows:

    𝐼𝑀𝐶1=𝐻𝑋𝑌−𝐻𝑋𝑌1max(𝐻𝑋,𝐻𝑌),𝑤ℎ𝑒𝑟𝑒𝐻𝑋𝑌=−∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝(𝑖,𝑗)log⁡(𝑝(𝑖,𝑗))𝐻𝑋𝑌1=−∑𝑖,𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝(𝑖,𝑗)log⁡(𝑝𝑥(𝑖)𝑝𝑦(𝑗))𝐻𝑋=−∑𝑖=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝𝑥(𝑖)log⁡(𝑝𝑥(𝑖))𝐻𝑌=−∑𝑗=0𝑙𝑒𝑣𝑒𝑙𝑠−1𝑝𝑦(𝑗)log⁡(𝑝𝑦(𝑗))𝑝𝑥(𝑖)=∑𝑗=1𝑙𝑒𝑣𝑒𝑙𝑠𝑝(𝑖,𝑗)𝑝𝑦(𝑗)=∑𝑗=1𝑙𝑒𝑣𝑒𝑙𝑠𝑝(𝑖,𝑗)

86. Haralick.IMC2.Mean: float

87. Haralick.IMC2.Range: float

    Mean and range of the second information measure of correlation feature for GLCMs of all offsets. It is computed as follows: