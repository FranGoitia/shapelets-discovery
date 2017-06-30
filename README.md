# Shapelets
Shapelets are time series subsequences which are in some sense maximally
representative of a class. Algorithms based on the time series
shapelet primitives can be interpretable, more accurate and
significantly faster than state-of-the-art classifiers.
This packages implements a random strategy for evaluating candidates and
finding the optimum shapelet gaining a lot in speed with little accuracy penalty.
Not all possible shapelet candidates are evaluated but a sample of the set is taken
based on the given probability for considering each candidate.


## Shapelet Classifier
finder function receives training and cross validation sets and the probability of
considering a candidate (0.01 is suggested for long time series) returning
a shapelet that can be used for classifying unseen time series.

```scala
import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{DenseMatrix}

import shapelets.{TimeSeries, Shapelet}
import shapelets.Main.finder
var train_set: ArrayBuffer[Tuple2[DenseMatrix[Double], String]] = ArrayBuffer((DenseMatrix(1.0, 2.0), "0"),
                                                                           (DenseMatrix(1.5, 2.5), "0"),
                                                                           (DenseMatrix(3.0, 5.0), "1"),
                                                                           (DenseMatrix(4.0, 5.0), "1"))
var cross_set: ArrayBuffer[Tuple2[DenseMatrix[Double], String]] = ArrayBuffer((DenseMatrix(1.5, 1.5), "0"),
                                                                           (DenseMatrix(1.3, 2.7), "0"),
                                                                           (DenseMatrix(3.5, 4), "1"),
                                                                           (DenseMatrix(3.8, 4.7), "1"))
var train_set2 = for ((seq, label) <- train_set) yield new TimeSeries(seq, label)
var cross_set2 = for ((seq, label) <- cross_set) yield new TimeSeries(seq, label)
val sample_prob = 1
val shap: Shapelet = finder(train_set2, cross_set2, sample_prob)
shap.predict(new TimeSeries(DenseMatrix(1.0, 3.0))) // returns 0
shap.predict(new TimeSeries(DenseMatrix(3.0, 4.0))) // returns 1
```
