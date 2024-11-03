import HW2.TokenData
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

class DataLoadingTokenTest extends AnyFunSuite with BeforeAndAfterEach{

  private var sc: SparkContext = _

  override def beforeEach(): Unit = {
    // Initialize SparkContext before each test
    val conf = new SparkConf().setAppName("TestApp").setMaster("local[*]")
    sc = new SparkContext(conf)
  }

  override def afterEach(): Unit = {
    // Stop SparkContext after each test
    if (sc != null) {
      sc.stop()
    }
    // Ensure Spark is fully cleared between tests
    System.clearProperty("spark.driver.port")
  }

  test("Token data should be parsed correctly") {
    val tokenDataPath = "src/test/resources/input/part-r-00000DLTT"
    val tokensRDD: RDD[TokenData] = HW2.loadTokenData(sc, tokenDataPath)

    val expectedTokenCount =7
    assert(tokensRDD.count() == expectedTokenCount)
    assert(tokensRDD.first().word == "a")
  }
}