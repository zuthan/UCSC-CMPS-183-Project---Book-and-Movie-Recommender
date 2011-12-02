package datasource

import com.mysql.jdbc.jdbc2.optional.MysqlDataSource
import org.apache.mahout.cf.taste.impl.model.jdbc.{ConnectionPoolDataSource, ReloadFromJDBCDataModel, MySQLJDBCDataModel}
import org.apache.mahout.cf.taste.model.DataModel

object MySqlBXDataModel {
  def makeMySqlDataSource(dbName: String) = {
    val datasource = new MysqlDataSource
    datasource.setURL("jdbc:mysql://127.0.0.1:3306/" + dbName + "?user=root")
    new ConnectionPoolDataSource(datasource)
  }

//  def apply(): MySQLJDBCDataModel = apply("books","ratings")

  def apply(dbName: String, tableName: String): MySQLJDBCDataModel = {
    val dataModel = new MySQLJDBCDataModel(
      makeMySqlDataSource(dbName),
      tableName,
      "user",
      "item",
      "rating",
      "timestamp")
    dataModel
  }
  
//  def cached: DataModel = new ReloadFromJDBCDataModel(apply())

  def cached(dbName: String, tableName: String): DataModel = new ReloadFromJDBCDataModel(apply(dbName,tableName))
}

