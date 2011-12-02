package lib.eval

import collection.JavaConversions._
import org.apache.mahout.cf.taste.impl.common.{FullRunningAverage, RunningAverage}
import java.util.concurrent.atomic.AtomicInteger
import org.apache.mahout.cf.taste.recommender.Recommender
import collection.mutable.ArrayBuffer
import jsc.datastructures.PairedData
import jsc.correlation.KendallCorrelation
import org.apache.mahout.cf.taste.model.{Preference, PreferenceArray}
import org.apache.mahout.cf.taste.common.{NoSuchItemException, NoSuchUserException}
import org.slf4j.{Logger, LoggerFactory}

class KendallTauRecommenderEvaluator extends AbstractRecommenderEvaluator{
  protected final val log: Logger = LoggerFactory.getLogger(classOf[AbstractRecommenderEvaluator])

  private var average: RunningAverage = null

  protected def reset {
    average = new FullRunningAverage
  }

  protected def evaluateOneUser(recommender: Recommender, testUserID: Long, prefs: PreferenceArray, noEstimateCounter: AtomicInteger) {
    object EstimatedPreference{
      def unapply(pref: Preference): Option[Double] = {
        try {
          val estimatedPreference = recommender.estimatePreference(testUserID, pref.getItemID)
          if(estimatedPreference.isNaN) None
          else Some(estimatedPreference)
        }
        catch {
          case nsue: NoSuchUserException => {
            log.info("User exists in test data but not training data: {}", testUserID)
            None
          }
          case nsie: NoSuchItemException => {
            log.info("Item exists in test data but not training data: {}", pref.getItemID)
            None
          }
        }
      }
    }

    val data: Array[Array[Double]] = prefs.flatMap{
      case p@EstimatedPreference(v) => Some(Array(p.getValue.toDouble, v))
      case _ => {
        noEstimateCounter.incrementAndGet
        None
      }
    }.toArray

    if(data.length > 1){
      val pairedData = new PairedData(data)
      val kendall = new KendallCorrelation(pairedData)
      average.addDatum(kendall.getR)
    }
    //else do not add a data point
  }

  protected def computeFinalEvaluation: Double = {
    average.getAverage
  }

  override def toString = "KendallTauRecommenderEvaluator"
}
