# [Histomicstk.features](https://digitalslidearchive.github.io/HistomicsTK/histomicstk.features.html#module-histomicstk.features)

## Identifier

> Location of the nucleus and its code in the input labeled mask. Columns are prefixed by *Identifier.*. These include …

00 Label
01 Identifier.Xmin
02 Identifier.Ymin
03 Identifier.Xmax
04 Identifier.Ymax
05 Identifier.CentroidX
06 Identifier.CentroidY
07 Identifier.WeightedCentroidX
08 Identifier.WeightedCentroidY

## Morphometry (size, shape, and orientation) features of the nuclei

> See histomicstk.features.compute_morphometry_features for more details. Feature names prefixed by *Size.*, *Shape.*, or *Orientation.*.

09 Orientation.Orientation
10 Size.Area
11 Size.ConvexHullArea
12 Size.MajorAxisLength
13 Size.MinorAxisLength
14 Size.Perimeter
15 Shape.Circularity
16 Shape.Eccentricity
17 Shape.EquivalentDiameter
18 Shape.Extent
19 Shape.FractalDimension
20 Shape.MinorMajorAxisRatio
21 Shape.Solidity
22 Shape.HuMoments1
23 Shape.HuMoments2
24 Shape.HuMoments3
25 Shape.HuMoments4
26 Shape.HuMoments5
27 Shape.HuMoments6
28 Shape.HuMoments7

29 Shape.WeightedHuMoments1
30 Shape.WeightedHuMoments2
31 Shape.WeightedHuMoments3
32 Shape.WeightedHuMoments4
33 Shape.WeightedHuMoments5
34 Shape.WeightedHuMoments6
35 Shape.WeightedHuMoments7

## Fourier shape descriptor features

> See histomicstk.features.compute_fsd_features for more details. Feature names are prefixed by *FSD*.

36 Shape.FSD1
37 Shape.FSD2
38 Shape.FSD3
39 Shape.FSD4
40 Shape.FSD5
41 Shape.FSD6

## Intensity features for the nucleus and cytoplasm channels

> See histomicstk.features.compute_fsd_features for more details. Feature names are prefixed by *Nucleus.Intensity.* for nucleus features and *Cytoplasm.Intensity.* for cytoplasm features.

42 Nucleus.Intensity.Min
43 Nucleus.Intensity.Max
44 Nucleus.Intensity.Mean
45 Nucleus.Intensity.Median
46 Nucleus.Intensity.MeanMedianDiff
47 Nucleus.Intensity.Std
48 Nucleus.Intensity.IQR
49 Nucleus.Intensity.MAD
50 Nucleus.Intensity.Skewness
51 Nucleus.Intensity.Kurtosis
52 Nucleus.Intensity.HistEnergy
53 Nucleus.Intensity.HistEntropy

## Gradient/edge features for the nucleus and cytoplasm channels

> See histomicstk.features.compute_gradient_features for more details. Feature names are prefixed by *Nucleus.Gradient.* for nucleus features and *Cytoplasm.Gradient.* for cytoplasm features.

54 Nucleus.Gradient.Mag.Mean
55 Nucleus.Gradient.Mag.Std
56 Nucleus.Gradient.Mag.Skewness
57 Nucleus.Gradient.Mag.Kurtosis
58 Nucleus.Gradient.Mag.HistEntropy
59 Nucleus.Gradient.Mag.HistEnergy
60 Nucleus.Gradient.Canny.Sum
61 Nucleus.Gradient.Canny.Mean

## Haralick features for the nucleus and cytoplasm channels

> See histomicstk.features.compute_haralick_features for more details. Feature names are prefixed by *Nucleus.Haralick.* for nucleus features and *Cytoplasm.Haralick.* for cytoplasm features.

62 Nucleus.Haralick.ASM.Mean
63 Nucleus.Haralick.ASM.Range
64 Nucleus.Haralick.Contrast.Mean
65 Nucleus.Haralick.Contrast.Range
66 Nucleus.Haralick.Correlation.Mean
67 Nucleus.Haralick.Correlation.Range
68 Nucleus.Haralick.SumOfSquares.Mean
69 Nucleus.Haralick.SumOfSquares.Range
70 Nucleus.Haralick.IDM.Mean
71 Nucleus.Haralick.IDM.Range
72 Nucleus.Haralick.SumAverage.Mean
73 Nucleus.Haralick.SumAverage.Range
74 Nucleus.Haralick.SumVariance.Mean
75 Nucleus.Haralick.SumVariance.Range
76 Nucleus.Haralick.SumEntropy.Mean
77 Nucleus.Haralick.SumEntropy.Range
78 Nucleus.Haralick.Entropy.Mean
79 Nucleus.Haralick.Entropy.Range
80 Nucleus.Haralick.DifferenceVariance.Mean
81 Nucleus.Haralick.DifferenceVariance.Range
82 Nucleus.Haralick.DifferenceEntropy.Mean
83 Nucleus.Haralick.DifferenceEntropy.Range
84 Nucleus.Haralick.IMC1.Mean
85 Nucleus.Haralick.IMC1.Range
86 Nucleus.Haralick.IMC2.Mean
87 Nucleus.Haralick.IMC2.Range
88 Intensity.distribution.part05
89 Intensity.distribution.part04
90 Intensity.distribution.part03
91 Intensity.distribution.part02
92 Intensity.distribution.part01