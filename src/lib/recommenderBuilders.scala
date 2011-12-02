package lib

import org.apache.mahout.cf.taste.eval.RecommenderBuilder
import org.apache.mahout.cf.taste.model.DataModel
import org.apache.mahout.cf.taste.recommender.Recommender
import org.apache.mahout.cf.taste.impl.recommender.knn.{KnnItemBasedRecommender, ConjugateGradientOptimizer}
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood
import org.apache.mahout.cf.taste.impl.recommender.{GenericUserBasedRecommender, CachingRecommender}
import org.apache.mahout.cf.taste.impl.similarity.{CachingItemSimilarity, CachingUserSimilarity, PearsonCorrelationSimilarity}
import similarity.ScaledDotProductSimilarity
import org.apache.mahout.cf.taste.common.Weighting
import org.apache.mahout.cf.taste.impl.recommender.slopeone.{MemoryDiffStorage, SlopeOneRecommender}
import org.apache.mahout.cf.taste.impl.recommender.slopeone.jdbc.MySQLJDBCDiffStorage
import datasource.MySqlBXDataModel
import org.apache.mahout.cf.taste.impl.recommender.slopeone.file.FileDiffStorage
import java.io.File
import org.apache.mahout.cf.taste.impl.model.jdbc.AbstractJDBCDataModel

object SlopeOneRecBuilder extends RecommenderBuilder {
  def buildRecommender(model: DataModel): Recommender = {

    new SlopeOneRecommender(model)
  }

  override def toString = "SlopeOneRecBuilder"
}

case class SlopeOneFileDiffRecBuilder(fileDiffStorage: File) extends RecommenderBuilder {
  def buildRecommender(model: DataModel): Recommender = {
    val diffStorage = new FileDiffStorage(fileDiffStorage,1000000)

    new SlopeOneRecommender(model,
      Weighting.WEIGHTED,
      Weighting.WEIGHTED,
      diffStorage
    )
  }

  override def toString = "SlopeOneFileDiffRecBuilder"
}

case class SlopeOneSqlDiffRecBuilder(diffTable: String, jdbcModel:AbstractJDBCDataModel) extends RecommenderBuilder {
  def buildRecommender(model: DataModel): Recommender = {

    val diffStorage = new MySQLJDBCDiffStorage(jdbcModel,
      diffTable,
      "item_id_a",
      "item_id_b",
      "count",
      "average_diff",
      "standard_deviation",
      2
    )

    new SlopeOneRecommender(model,
      Weighting.WEIGHTED,
      Weighting.WEIGHTED,
      diffStorage
      )
  }

  override def toString = "SlopeOneSqlDiffRecBuilder"
}


case class ItemBasedKnnPearsonSimilarityRecBuilder(k: Int) extends RecommenderBuilder() {
  def buildRecommender(model: DataModel): Recommender = {
//    val similarity = new CachingItemSimilarity(new PearsonCorrelationSimilarity(model),model)
    val similarity = new PearsonCorrelationSimilarity(model)
    val optimizer = new ConjugateGradientOptimizer
//    new CachingRecommender(new KnnItemBasedRecommender(model,similarity,optimizer,k))
    new KnnItemBasedRecommender(model,similarity,optimizer,k)
  }
}

case class UserBasedKnnPearsonSimilarityRecBuilder(k: Int) extends RecommenderBuilder {
  def buildRecommender(model: DataModel): Recommender = {
//    val similarity = new CachingUserSimilarity(new PearsonCorrelationSimilarity(model),model)
    val similarity = new PearsonCorrelationSimilarity(model)
    val neighborhood = new NearestNUserNeighborhood(k, similarity, model)
    //new CachingRecommender(new GenericUserBasedRecommender(model,neighborhood,similarity))
    new GenericUserBasedRecommender(model,neighborhood,similarity)
  }
}

case class UserBasedKnnDotProductSimilarityRecBuilder(k: Int, 
                                                      ratingScaling: Double = 1.0,
                                                      preScaling: Double = 1.0,
                                                      postScaling: Double = 1.0) extends RecommenderBuilder() {
    def buildRecommender(model: DataModel): Recommender = {
//      val similarity = new CachingUserSimilarity(new ScaledDotProductSimilarity(model,preScaling,postScaling),model)
      val similarity = new ScaledDotProductSimilarity(model,ratingScaling, preScaling,postScaling)
      val neighborhood = new NearestNUserNeighborhood(k, similarity, model)
      //new CachingRecommender(new GenericUserBasedRecommender(model,neighborhood,similarity))
      new GenericUserBasedRecommender(model,neighborhood,similarity)
    }
  }
