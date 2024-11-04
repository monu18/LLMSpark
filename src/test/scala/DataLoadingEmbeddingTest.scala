import HW2.EmbeddingData
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

class DataLoadingEmbeddingTest extends AnyFunSuite with BeforeAndAfterEach{

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

  test("Embedding data should be parsed correctly") {
    val embeddingDataPath = "src/test/resources/input/embeddingsDLET.csv"
    val embeddingsRDD: RDD[EmbeddingData] = HW2.loadEmbeddingData(sc, embeddingDataPath)

    val expectedEmbeddingCount = 7
    val expectedEmbeddingDim = 100
    assert(embeddingsRDD.count() == expectedEmbeddingCount)
    assert(embeddingsRDD.first().embeddings.length == expectedEmbeddingDim)
  }
}