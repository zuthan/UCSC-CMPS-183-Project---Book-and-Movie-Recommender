package lib.similarity

import collection.JavaConversions._
import org.apache.mahout.cf.taste.model.{DataModel}
import java.lang.{Double => jDouble}


class ScaledDotProductSimilarity(dataModel: DataModel, ratingScaleFactor: Double = 1.0, preSumScaleFactor: Double = 1.0, postSumScaleFactor: Double = 1.0) extends AbstractUnboundedSimilarity(dataModel){
  def getSimilarityAccumulator = new SimilarityAccumulator {
    var sumXY = 0.0

    def processPrefPair(u1Pref: jDouble, u2Pref: jDouble) {
      val xy = u1Pref*u2Pref*ratingScaleFactor*ratingScaleFactor
      val scaledXY = if(preSumScaleFactor!=1.0) math.signum(xy) * math.pow(math.abs(xy),preSumScaleFactor) else xy
      sumXY += scaledXY
    }

    def computeResult() = {
      val result = if(postSumScaleFactor!=1.0) math.signum(sumXY) * math.pow(math.abs(sumXY),postSumScaleFactor) else sumXY
      result
    }
  }
}
