package run

import collection.JavaConversions._

import datasource.MySqlBXDataModel
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import java.io._
import lib._
import eval.{MAERecommenderEvaluator, AbstractRecommenderEvaluator, RMSRecommenderEvaluator, KendallTauRecommenderEvaluator}
import org.apache.mahout.cf.taste.eval.{RecommenderBuilder, DataModelBuilder}
import org.apache.mahout.cf.taste.model.DataModel
import org.apache.mahout.cf.taste.impl.model.jdbc.AbstractJDBCDataModel


object BookCrossingEvaluator extends App{
  val curDir = new File("D:\\My Dropbox\\Uni\\Machine Learning and Data Mining\\Project\\BookRecommenderProject")

  /* --------- Evaluation --------- */
    val rmsEvaluator = new RMSRecommenderEvaluator
    val maeEvaluator = new MAERecommenderEvaluator
    val kendallEvaluator = new KendallTauRecommenderEvaluator

//  evalSetSql("Book-Crossing", "books", "ratings", 0.1f, 1.0f)
//  evalSetSql("Book-Crossing-Centered", "books", "ratings_centered", -0.9f, 0.9f)
//  evalSetText("MovieLens", "movieLens", 0.2f, 1.0f, true)
  evalSetText("MovieLens-Centered", "movieLens-centered", -1.0f, 1.0f, true)
//  evalSetText("MovieLens", "movieLens-unNormalized", 1.0f, 5.0f, true)

  /**
   * Text Source Evaluation
   * perform a preset set of evaluations over data from a specified text file
   */
  def evalSetText(datasetName: String,
                  datasetFileName: String,
                  minPref: Float,
                  maxPref: Float,
                  includeKendall: Boolean = false){

  //set min, max preference values
  maeEvaluator.setMinPreference(minPref)
  maeEvaluator.setMaxPreference(maxPref)

      /* --------- Model --------- */
  val dataFile = new File(curDir,"data\\" + datasetFileName + ".dat")
  val model = new FileDataModel(dataFile)

    val slopeOneBuilder = SlopeOneRecBuilder

    val dotProductBuilder = UserBasedKnnDotProductSimilarityRecBuilder(100)
    val pearsonBuilder = UserBasedKnnPearsonSimilarityRecBuilder(100)

    /* --------- Builders --------- */
    val builders: List[RecommenderBuilder] = List(
//      slopeOneBuilder,
//      pearsonBuilder,
//      dotProductBuilder
    )

//    executeEval(datasetName,model,maeEvaluator,0.8,0.2,List(slopeOneBuilder),1)
//    executeEval(datasetName,model,maeEvaluator,0.8,0.2,List(pearsonBuilder),1)
//    executeEval(datasetName,model,maeEvaluator,0.8,0.2,List(dotProductBuilder),1)

    for(trainPercent <- 0.8 to (0.8, 0.1);
        evalPercent <- List(0.2)){
//      executeEval(datasetName,model,maeEvaluator,trainPercent,evalPercent,builders,1)
    }
    if(includeKendall){
      for(trainPercent <- 0.1 to (0.6, 0.1);
          evalPercent <- List(0.2)){
//        executeEval(datasetName,model,kendallEvaluator,trainPercent,evalPercent,builders,5)
      }
    }
    val dotProdBuilders1 = (1 to (10, 1)).map(v =>
      UserBasedKnnDotProductSimilarityRecBuilder(100, v, 1, 1)
    ).toList
    val dotProdBuilders2 = (0.2 to (1.4, 0.2)).map(v =>
      UserBasedKnnDotProductSimilarityRecBuilder(100, 1, 1, v)
    ).toList
    val dotProdBuilders3 = (0.2 to (1.4, 0.2)).map(v =>
      UserBasedKnnDotProductSimilarityRecBuilder(100, 1, v, 1)
    ).toList

    executeEval(datasetName, model, maeEvaluator, 0.8, 0.2, dotProdBuilders1, 5)
    executeEval(datasetName, model, maeEvaluator, 0.8, 0.2, dotProdBuilders2, 5)
    executeEval(datasetName, model, maeEvaluator, 0.8, 0.2, dotProdBuilders3, 5)
  }

  /**
   * JDBC Source Evaluation
   * perform a preset set of evaluations over data from a specified JDBC Source
   */
  def evalSetSql(datasetName: String,
              dbName: String,
              tableName: String,
              minPref: Float,
              maxPref: Float,
              includeKendall: Boolean = false) {

    //set min, max preference values
    List(rmsEvaluator, maeEvaluator).foreach{e =>
      e.setMinPreference(minPref)
      e.setMaxPreference(maxPref)
    }

    val model = MySqlBXDataModel.cached(dbName,tableName)
    val jdbcModel: AbstractJDBCDataModel = MySqlBXDataModel(dbName,tableName)
    val slopeOneBuilder = SlopeOneSqlDiffRecBuilder(tableName + "_diffs", jdbcModel)

    val dotProductBuilder = UserBasedKnnDotProductSimilarityRecBuilder(100,1.0,1)
    val pearsonBuilder = UserBasedKnnPearsonSimilarityRecBuilder(100)

    /* --------- Builders --------- */
    val builders: List[RecommenderBuilder] = List(
      slopeOneBuilder,
  //    ItemBasedKnnPearsonSimilarityRecBuilder(100),
      pearsonBuilder,
      dotProductBuilder
    )

//    executeEval(datasetName,model,maeEvaluator,0.8,0.2,List(slopeOneBuilder),1)
//    executeEval(datasetName,model,maeEvaluator,0.8,0.2,List(pearsonBuilder),1)
//    executeEval(datasetName,model,maeEvaluator,0.8,0.2,List(dotProductBuilder),1)

    for(trainPercent <- 0.1 to (0.9, 0.1);
        evalPercent <- List(0.2)){
      executeEval(datasetName,model,maeEvaluator,trainPercent,evalPercent,builders,5)
    }
    if(includeKendall){
      for(trainPercent <- 0.1 to (0.6, 0.1);
          evalPercent <- List(0.2)){
        executeEval(datasetName,model,kendallEvaluator,trainPercent,evalPercent,builders,5)
      }
    }
  }

  def executeEval(dataset: String,
                  model: DataModel,
                  evaluator: AbstractRecommenderEvaluator,
                  trainingPrefsPercent: Double,
                  evalUsersPercent: Double,
                  builders: List[RecommenderBuilder],
                  repeat: Int
  ){
    val statsFile = new File(curDir,"stats.txt")
    val output = new FileOutputStream(statsFile,true)
    val statsWriter = new PrintWriter(output)

    //print header info
    statsWriter.println()
    statsWriter.println("dataset: " + dataset)
    statsWriter.println("evaluator: " + evaluator)
    statsWriter.println(trainingPrefsPercent*100 + "% of each user's preferences used for training.")
    statsWriter.println(evalUsersPercent*100 + "% of users used for evaluation.")

    builders.foreach{builder =>
      val scores = (1 to repeat).map{_ => evaluator.evaluate(builder,null,model,trainingPrefsPercent,evalUsersPercent)}
      val realScores = scores.filterNot(_.isNaN)
      val avgScore = realScores.size match {
        case 0 => Double.NaN
        case size => realScores.sum[Double] / size
      }

      val msg = builder.toString + ": \t\t" + avgScore + scores.mkString(" (",",",")")
      statsWriter.println(msg)
    }

    statsWriter.close()
  }

}
