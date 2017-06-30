package shapelets

import scala.collection.mutable.{ArrayBuffer, Map}
import scala.collection.parallel.mutable.{ParArray}
import scala.collection.immutable.{IndexedSeq}
import scala.util.Random.{nextFloat, shuffle}
import scala.math.{pow}

import breeze.linalg.{DenseVector, DenseMatrix, norm}
import breeze.stats.{stddev}


class TimeSeries(val seq: DenseMatrix[Double], val label: String = "") {

  def length(): Int = {
    seq.rows
  }
}


class Shapelet(val seq: DenseMatrix[Double], val splitDist: Double) {

  def predict(otherSeq: TimeSeries, score: Boolean=false): Double = {
    val dist: Double = distanceToSeq(otherSeq.seq)
    if (score == true) dist
    else {if (dist < splitDist) 1.0 else 0.0}
  }

  def distanceToSeq(otherSeq: DenseMatrix[Double]): Double = {
    val subseqs: IndexedSeq[DenseMatrix[Double]] = genSubsets(otherSeq, seq.rows)
    subseqs.par.map(s => norm((seq - s).toDenseVector)).min
  }

  def genSubsets(timeSeries: DenseMatrix[Double], length: Integer):
                 IndexedSeq[DenseMatrix[Double]] = {
    for (i <- (0 to (timeSeries.rows - length))) yield timeSeries(i until i+length, 0 until timeSeries.cols)
  }

}


class ShapeletFinder(val trainSet: ArrayBuffer[TimeSeries], val crossValSet: ArrayBuffer[TimeSeries],
                     val minLen: Int, val maxLen: Int, val randomShap: Double = 0.01) {

  println("training set: %d".format(trainSet.length))
  println("cross validation set: %d".format(crossValSet.length))

  def getOptimal(): Shapelet = {
    val candidates = genCandidates(randomShap)
    val shapsSplitsScores = candidates.par.map(c => testCandidate(c))
    println(candidates.length)
    val bestCandidates = filterCandidates(candidates zip shapsSplitsScores, 1)
    println(bestCandidates.length)
    val bestShapelet = crossValidate(bestCandidates)
    bestShapelet
  }

  def filterCandidates(candidates: ArrayBuffer[(DenseMatrix[Double], Array[Double])],
                       stdDevN: Double): ArrayBuffer[Shapelet] = {
    // returns all candidates that are stdDevN from best score
    val scores: DenseVector[Double] = DenseVector(candidates.map(c => c._2(1)): _*)
    val threshold = scores.max - stdDevN * stddev(scores)
    println(scores.max)
    val bestCandidates: ArrayBuffer[Shapelet] = ArrayBuffer()
    for ((seq, splitScore) <- candidates) {
      val splitDist = splitScore(0)
      val score = splitScore(1)
      if (score >= threshold) bestCandidates.append(new Shapelet(seq, splitDist))
    }
    bestCandidates
  }

  def genCandidates(randomShap: Double): ArrayBuffer[DenseMatrix[Double]] = {
    var allCandidates: ArrayBuffer[DenseMatrix[Double]] = ArrayBuffer()
    var l = maxLen
    while (l > minLen) {
      for (timeSeries <- trainSet) {
        val candidates: IndexedSeq[DenseMatrix[Double]] = genSubsets(timeSeries.seq, l)
        for (c <- candidates if nextFloat() < randomShap) allCandidates.append(c)
      }
      l -= 1
    }
    allCandidates
  }

  def testCandidate(shapelet: DenseMatrix[Double]): Array[Double] = {
    var tSeriesDistances: Map[TimeSeries, Double] = Map()
    for (timeSeries <- trainSet) {
      val distance = seriesShapDist(timeSeries.seq, shapelet)
      tSeriesDistances(timeSeries) = distance
    }

    var splitCandidates: ArrayBuffer[Double] = ArrayBuffer()
    val dst = tSeriesDistances.values.toArray.sorted
    for ((d1, d2) <- (dst.slice(0, dst.length-1) zip dst.takeRight(dst.length-1))) {
      splitCandidates.append((d1+d2)/2)
    }
    val (split, score) = scoreBestSplit(tSeriesDistances, splitCandidates)
    Array(split, score)
  }

  def seriesShapDist(timeSeries: DenseMatrix[Double], shapeletSeq: DenseMatrix[Double]) = {
    val subsets: IndexedSeq [DenseMatrix[Double]] = genSubsets(timeSeries, shapeletSeq.rows)
    subsets.par.map(s => norm((shapeletSeq - s).toDenseVector)).min
  }

  def genSubsets(timeSeries: DenseMatrix[Double], length: Int): IndexedSeq[DenseMatrix[Double]] = {
    for (i <- (0 to (timeSeries.rows - length))) yield timeSeries(i until i+length, 0 until timeSeries.cols)
  }

  def scoreBestSplit(tSeriesDistances: Map[TimeSeries, Double],
                       splitCandidates: ArrayBuffer[Double]): Tuple2[Double, Double] = {
    val (score, split) = splitCandidates.par.map(c => (getSplitScore(tSeriesDistances, c), c)).maxBy(_._1)
    (split, score)
  }

  def getSplitScore(tSeriesDistances: Map[TimeSeries, Double], split: Double): Double = {
    var real: ArrayBuffer[Double] = ArrayBuffer()
    var predictions: ArrayBuffer[Double] = ArrayBuffer()
    for ((timeSeries, distance) <- tSeriesDistances) {
      val pred = if (distance < split) 1.0 else 0.0
      predictions.append(pred)
      real.append(timeSeries.label.toDouble)
    }
    scorePreds(predictions, real, 10)
  }

    def crossValidate(shapelets: ArrayBuffer[Shapelet]): Shapelet = {
    val real: ArrayBuffer[Double] = crossValSet.map(t => t.label.toDouble)
    println("cross validating ...")
    val (score, shapelet) = shapelets.par.map(shap => (crossValidateShapelet(shap, real), shap)).maxBy(_._1)
    println(score)
    shapelet
  }

  def crossValidateShapelet(shap: Shapelet, real: ArrayBuffer[Double]): Double = {
    val predictions = crossValSet.map(t => shap.predict(t))
    scorePreds(predictions, real, 10)
  }

  def scorePreds(predictions: ArrayBuffer[Double], real: ArrayBuffer[Double],
                 penalization: Int=0): Double = {
    var score = 0
    for ( (p, r) <- (predictions zip real)) {
      if (p == r) score += 1
      // penalize false positives
      else if (r == 0 && p == 1)
        score = score - penalization
    }
    score / real.length.toDouble
  }
}


object Main {
  def finder(trainSet: ArrayBuffer[TimeSeries], crossSet: ArrayBuffer[TimeSeries], sampleProb: Double): Shapelet = {
    val lengths = (trainSet ++ crossSet).par.map(t => t.length)
    val minLength: Int = lengths.sum / lengths.length / 10
    val maxLength: Int = lengths.min
    val shapeletFinder = new ShapeletFinder(trainSet, crossSet, minLength, maxLength, sampleProb)
    shapeletFinder.getOptimal()
  }
}
