import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import utils.ConfigUtil

import java.io.{BufferedWriter, OutputStream, OutputStreamWriter}

object HW2 {

  case class TokenData(word: String, token: Int, frequency: Int)
  case class EmbeddingData(token: Int, word: String, embeddings: Array[Double])
  case class WindowedData(contextEmbedding: Array[Double], targetEmbedding: Array[Double])

  def main(args: Array[String]): Unit = {

    val logger = Logger.getLogger(getClass.getName)
    Logger.getLogger("org").setLevel(Level.ERROR)

    // Check if enough arguments are passed
    if (args.length < 5) {
      logger.warn("args length is less than 5")
    }

    // Initialize configuration
    ConfigUtil.initializeConfig(args.toList)

    // Access configuration values
    val config = ConfigUtil.finalConfig

    // Extract arguments or use defaults from config
    val embeddingPath = config.embeddingPath
    val tokenDataPath = config.tokenDataPath
    val modelOutputPath = config.modelOutputPath
    val statsOutputPath = config.statsOutputPath
    val master = config.master
    val learningRate = config.learningRate
    val momentum = config.momentum
    val batchSize = config.batchSize
    val windowSize = config.windowSize

    logger.info(s"Embedding path: $embeddingPath")
    logger.info(s"Token data path: $tokenDataPath")
    logger.info(s"Model output path: $modelOutputPath")
    logger.info(s"Stats output path: $statsOutputPath")
    logger.info(s"Spark master: $master")


    // Set up Spark configuration
    val conf = new SparkConf().setAppName("LLMSpark").setMaster(master)
    val sc = new SparkContext(conf)

    // Start time for tracking training duration
    val trainingStartTime = System.currentTimeMillis()

    // Load token data
    val tokensRDD: RDD[TokenData] = sc.textFile(tokenDataPath).flatMap { line =>
      val parts = line.split(",")
      if (parts.length == 3) {
        val word = parts(0)
        val tokensList = parts(1).replaceAll("[\\[\\]]", "").split(" ").flatMap(safeToInt)
        val frequency = safeToInt(parts(2))
        frequency match {
          case Some(freq) => tokensList.map(token => TokenData(word, token, freq))
          case None => None
        }
      } else None
    }

    // Load and parse the embedding data
    val embeddingsRDD: RDD[EmbeddingData] = sc.textFile(embeddingPath).flatMap { line =>
      val parts = line.split(",")
      if (parts.length > 2) {
        val tokenOpt = safeToInt(parts(0))
        val word = parts(1)
        val embeddings = parts.drop(2).flatMap(safeToDouble)
        tokenOpt match {
          case Some(token) if embeddings.nonEmpty => Some(EmbeddingData(token, word, embeddings))
          case _ => None
        }
      } else None
    }

    // Broadcast embeddings for efficiency
    val embeddingMap = embeddingsRDD.collect().map(e => e.token -> e.embeddings).toMap
    val embeddingMapBroadcast = sc.broadcast(embeddingMap)

    // Define parameters for sliding windows and embeddings
    val embeddingDim = embeddingMapBroadcast.value.head._2.length
    val positionalEmbedding = computePositionalEmbedding(windowSize, embeddingDim)
    val positionalEmbeddingBroadcast = sc.broadcast(positionalEmbedding)

    // Create sliding windows with positional embeddings
    val slidingWindowsRDD: RDD[WindowedData] = tokensRDD.mapPartitions { tokensIter =>
      createSlidingWindowsWithPositionalEmbedding(
        tokensIter.toArray,
        windowSize,
        embeddingMapBroadcast,
        positionalEmbeddingBroadcast
      ).iterator
    }

    // Prepare data for training
    val trainingDataRDD: RDD[DataSet] = slidingWindowsRDD.map { window =>
      val input = Nd4j.create(Array(window.contextEmbedding))
      val label = Nd4j.create(Array(window.targetEmbedding))
      new DataSet(input, label)
    }
    val trainingJavaRDD = trainingDataRDD.toJavaRDD()

    // Create and configure the model
    val model = createModel(numInputs = windowSize * embeddingDim, numOutputs = embeddingDim, learningRate, momentum)
    model.setListeners(new ScoreIterationListener(10))
    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
      .batchSizePerWorker(batchSize)
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2)
      .build()

    // Train the model using distributed Spark
    val sparkModel = new SparkDl4jMultiLayer(sc, model, trainingMaster)
    sparkModel.fit(trainingJavaRDD)

    // End time for tracking training duration
    val trainingEndTime = System.currentTimeMillis()
    val trainingDuration = trainingEndTime - trainingStartTime

    // Save the model directly to HDFS
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val hdfsOutputStream: OutputStream = fs.create(new Path(modelOutputPath))
    ModelSerializer.writeModel(sparkModel.getNetwork, hdfsOutputStream, true)
    hdfsOutputStream.close()

    // Collect and save statistics
    val statsData = Seq(
      ("Total Training Time (ms)", trainingDuration.toString),
      ("Total Executors", sc.getExecutorMemoryStatus.size.toString),
      ("RDD Storage Information", getRDDStorageInfo(sc)),
      ("Gradient Stats", "Captured per iteration"),
      ("Learning Rate", model.getLayerWiseConfigurations.getConf(0).getLayer.getUpdaterByParam("W").asInstanceOf[Nesterovs].getLearningRate),
      ("CPU/GPU Utilization", "Available via Spark UI"),
      ("Data Shuffling and Partitioning", "Tracked in Spark UI"),
      ("Batch Size", batchSize.toString)
    )

    // Write stats to HDFS CSV
    val statsOutputStream: OutputStream = fs.create(new Path(statsOutputPath))
    val writer = new BufferedWriter(new OutputStreamWriter(statsOutputStream))
    writer.write("Metric,Value\n")
    statsData.foreach { case (metric, value) =>
      writer.write(s"$metric,$value\n")
    }
    writer.close()

    // Stop Spark Context
    sc.stop()
  }

  // Helper function for capturing RDD Storage information
  def getRDDStorageInfo(sc: SparkContext): String = {
    sc.getRDDStorageInfo.map { rddInfo =>
      s"RDD Name: ${rddInfo.name}, ID: ${rddInfo.id}, " +
        s"StorageLevel: ${rddInfo.storageLevel}, " +
        s"CachedPartitions: ${rddInfo.numCachedPartitions}, " +
        s"TotalPartitions: ${rddInfo.numPartitions}, " +
        s"MemorySize: ${rddInfo.memSize}, " +
        s"DiskSize: ${rddInfo.diskSize}"
    }.mkString("; ")
  }

  // Helper functions for safe parsing
  def safeToInt(s: String): Option[Int] = try Some(s.toInt) catch { case _: NumberFormatException => None }
  def safeToDouble(s: String): Option[Double] = try Some(s.toDouble) catch { case _: NumberFormatException => None }

  // Compute positional embedding using sinusoidal functions
  def computePositionalEmbedding(windowSize: Int, embeddingDim: Int): Array[Array[Double]] = {
    Array.tabulate(windowSize) { pos =>
      Array.tabulate(embeddingDim) { i =>
        val angle = pos / math.pow(10000, (2.0 * i) / embeddingDim)
        if (i % 2 == 0) math.sin(angle) else math.cos(angle)
      }
    }
  }

  // Create sliding windows with positional embedding
  def createSlidingWindowsWithPositionalEmbedding(
                                                   tokens: Array[TokenData],
                                                   windowSize: Int,
                                                   embeddingMapBroadcast: org.apache.spark.broadcast.Broadcast[Map[Int, Array[Double]]],
                                                   positionalEmbeddingBroadcast: org.apache.spark.broadcast.Broadcast[Array[Array[Double]]]
                                                 ): Array[WindowedData] = {
    val embeddingDim = embeddingMapBroadcast.value.head._2.length
    tokens.sliding(windowSize + 1).collect {
      case window if window.length == windowSize + 1 =>
        val contextEmbedding = window.init.zipWithIndex.flatMap { case (tokenData, pos) =>
          val tokenEmbedding = embeddingMapBroadcast.value.getOrElse(tokenData.token, Array.fill(embeddingDim)(0.0))
          val posEmbedding = positionalEmbeddingBroadcast.value(pos)
          tokenEmbedding.zip(posEmbedding).map { case (tokenVal, posVal) => tokenVal + posVal }
        }
        val targetEmbedding = embeddingMapBroadcast.value.getOrElse(window.last.token, Array.fill(embeddingDim)(0.0))
        WindowedData(contextEmbedding, targetEmbedding)
    }.toArray
  }

  // Model creation function
  def createModel(numInputs: Int, numOutputs: Int, learningRate: Double, momentum: Double): MultiLayerNetwork = {
    val conf = new NeuralNetConfiguration.Builder()
      .updater(new Nesterovs(learningRate, momentum))
      .list()
      .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numOutputs).activation(Activation.RELU).build())
      .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(numOutputs).nOut(numOutputs).activation(Activation.IDENTITY).build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model
  }
}