package run

import collection.JavaConversions._
import java.io.File
import org.grouplens.lenskit.data.sql.{BasicSQLStatementFactory, JDBCRatingDAO}
import org.grouplens.lenskit.core.LenskitRecommenderEngineFactory
import org.grouplens.lenskit.RatingPredictor
import org.grouplens.lenskit.knn.item.ItemItemRatingPredictor


object LenskitExperiment extends App{
  val curDir = new File("D:/Programming Workspace/BookRecommenderProject")
  val sqlStatementFactory = new BasicSQLStatementFactory()
  val jdbcURL = "jdbc:mysql://127.0.0.1:3306/books?user=root"
  val daoFactory = new JDBCRatingDAO.Factory(jdbcURL,sqlStatementFactory)
  val engineFactory = new LenskitRecommenderEngineFactory(daoFactory)
  engineFactory.setComponent[RatingPredictor](classOf[RatingPredictor], classOf[ItemItemRatingPredictor])
  val engine = engineFactory.create()
  val rec = engine.open()

  val dao = rec.getRatingDataAccessObject
  val rate = rec.getRatingPredictor
  val itemRec = rec.getItemRecommender

  //divide users into training/validation/test sets
  val userCount = dao.getUserCount
  val trainingCount = 10
  for(u <- dao.getUsers.take(trainingCount)){
    val items = dao.getUserHistory(u).itemSet
    val scores = rate.score(u,items).map(s => s.getKey + " -> " + s.getValue)
    
    println(scores.mkString("(",", ",")"))
  }
}
