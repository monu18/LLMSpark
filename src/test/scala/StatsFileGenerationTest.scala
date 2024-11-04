import HW2.getRDDStorageInfo
import org.apache.spark.{SparkConf, SparkContext}
import org.nd4j.linalg.learning.config.Nesterovs
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

import java.io.File
import scala.io.Source

class StatsFileGenerationTest extends AnyFunSuite with BeforeAndAfterEach{

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

  test("Statistics file should be generated with correct format") {

    // Collect and save statistics
    val statsData = Seq(
      ("Total Training Time (ms)", "500"),
      ("Total Executors", sc.getExecutorMemoryStatus.size.toString),
      ("RDD Storage Information", getRDDStorageInfo(sc)),
      ("Gradient Stats", "Captured per iteration"),
      ("Learning Rate", "0.01"),
      ("CPU/GPU Utilization", "Available via Spark UI"),
      ("Data Shuffling and Partitioning", "Tracked in Spark UI"),
      ("Batch Size", "32")
    )

    val statsPath = "src/test/resources/output/statsSFGT.csv"
    HW2.generateStatisticsFile(sc,statsData, statsPath)

    val file = new File(statsPath)
    assert(file.exists(), "Statistics file was not created")

    val lines = Source.fromFile(file).getLines().toList
    assert(lines.head == "Metric,Value", "CSV header is incorrect")
    assert(lines.length > 1, "CSV file has no data rows")
  }
}