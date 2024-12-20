import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.api.java.JavaRDD
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

import java.io.{BufferedWriter, ByteArrayOutputStream, OutputStream, OutputStreamWriter}
import java.lang.management.ManagementFactory
import scala.collection.mutable.ArrayBuffer

case class EpochMetrics(
                         epoch: Int,
                         timestamp: Long,
                         trainingLoss: Double,
                         trainingAccuracy: Double,
                         validationAccuracy: Double,
                         learningRate: Double,
                         usedMemoryMB: Long,
                         totalMemoryMB: Long,
                         maxMemoryMB: Long,
                         totalShuffleReadBytes: Long,
                         totalShuffleWriteBytes: Long,
                         maxTaskDuration: Long,
                         minTaskDuration: Long,
                         avgTaskDuration: Double,
                         failedTaskCount: Int,
                         processCpuLoad: Double,
                         systemCpuLoad: Double
                       )

class CustomSparkListener extends org.apache.spark.scheduler.SparkListener {
  val taskMetricsData = ArrayBuffer[TaskMetricsData]()
  var failedTaskCount: Int = 0

  override def onTaskEnd(taskEnd: org.apache.spark.scheduler.SparkListenerTaskEnd): Unit = {
    val metrics = taskEnd.taskMetrics
    val shuffleReadBytes = Option(metrics.shuffleReadMetrics).map(_.totalBytesRead).getOrElse(0L)
    val shuffleWriteBytes = Option(metrics.shuffleWriteMetrics).map(_.bytesWritten).getOrElse(0L)
    val taskDuration = taskEnd.taskInfo.duration

    taskMetricsData += TaskMetricsData(taskEnd.taskInfo.taskId, taskDuration, shuffleReadBytes, shuffleWriteBytes)
    if (taskEnd.taskInfo.failed) failedTaskCount += 1
  }

  def reset(): Unit = {
    taskMetricsData.clear()
    failedTaskCount = 0
  }
}

case class TaskMetricsData(taskId: Long, duration: Long, shuffleReadBytes: Long, shuffleWriteBytes: Long)

object HW2 {

  case class TokenData(word: String, token: Int, frequency: Int)
  case class EmbeddingData(token: Int, word: String, embeddings: Array[Double])
  case class WindowedData(contextEmbedding: Array[Double], targetEmbedding: Array[Double])

  def main(args: Array[String]): Unit = {

    val logger = Logger.getLogger(getClass.getName)
    Logger.getLogger("org").setLevel(Level.ERROR)

    // Log application start
    logger.info("Starting HW2 application")

    // Check if enough arguments are passed
    if (args.length < 5) {
      logger.warn("Not enough arguments passed. Expected at least 5 arguments. Adding Default")
    }

    // Initialize configuration
    ConfigUtil.initializeConfig(args.toList)
    val config = ConfigUtil.finalConfig

    // Access configuration values
    val embeddingPath = config.embeddingPath
    val tokenDataPath = config.tokenDataPath
    val modelOutputPath = config.modelOutputPath
    val statsOutputPath = config.statsOutputPath
    val master = config.master
    val learningRate = config.learningRate
    val momentum = config.momentum
    val batchSize = config.batchSize
    val windowSize = config.windowSize

    // Log configuration details
    logger.info(s"Configuration loaded with Master: $master, Window Size: $windowSize, Learning Rate: $learningRate")

    // Set up Spark configuration
    val conf = new SparkConf().setAppName("LLMSpark").setMaster(master)
    val sc = new SparkContext(conf)

    // Start time for tracking training duration
    val trainingStartTime = System.currentTimeMillis()
    logger.info("Training started")

    try {
      // Load token data
      val tokensRDD = loadTokenData(sc, tokenDataPath)
      logger.info("Token data loaded successfully")

      // Load and parse the embedding data
      val embeddingsRDD = loadEmbeddingData(sc, embeddingPath)
      logger.info("Embedding data loaded successfully")

      // Broadcast embeddings for efficiency
      val embeddingMap = embeddingsRDD.collect().map(e => e.token -> e.embeddings).toMap
      val embeddingMapBroadcast = sc.broadcast(embeddingMap)
      logger.trace("Embeddings broadcasted")

      // Define parameters for sliding windows and embeddings
      val embeddingDim = embeddingMapBroadcast.value.head._2.length
      val positionalEmbedding = computePositionalEmbedding(windowSize, embeddingDim)
      val positionalEmbeddingBroadcast = sc.broadcast(positionalEmbedding)
      logger.trace("Positional embeddings broadcasted")

      // Create sliding windows with positional embeddings
      val slidingWindowsRDD: RDD[WindowedData] = tokensRDD.mapPartitions { tokensIter =>
        createSlidingWindowsWithPositionalEmbedding(
          tokensIter.toArray,
          windowSize,
          embeddingMapBroadcast,
          positionalEmbeddingBroadcast
        ).iterator
      }
      logger.info("Sliding windows with positional embeddings created")

      // Prepare data for training
      val trainingDataRDD: RDD[DataSet] = slidingWindowsRDD.map { window =>
        val input = Nd4j.create(Array(window.contextEmbedding))
        val label = Nd4j.create(Array(window.targetEmbedding))
        new DataSet(input, label)
      }
      val trainingJavaRDD = trainingDataRDD.toJavaRDD()
      logger.info("Training data prepared")

      // Create and configure the model
      val model = createModel(numInputs = windowSize * embeddingDim, numOutputs = embeddingDim, learningRate, momentum)
      model.setListeners(new ScoreIterationListener(10))
      val trainingMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
        .batchSizePerWorker(batchSize)
        .averagingFrequency(5)
        .workerPrefetchNumBatches(2)
        .build()
      logger.info("Model configured and listeners added")

      // Train the model using distributed Spark
      val sparkModel = new SparkDl4jMultiLayer(sc, model, trainingMaster)

      val customListener = new CustomSparkListener()
      sc.addSparkListener(customListener)

      val metricsBuffer = ArrayBuffer[EpochMetrics]()
      val Array(trainingDataSetsRDD, validationDataSetsRDD) = trainingDataRDD.randomSplit(Array(0.8, 0.2))
      val trainingDataSetsJavaRDD = trainingDataSetsRDD.toJavaRDD()
      val validationDataSetsJavaRDD = validationDataSetsRDD.toJavaRDD()

      for (epoch <- 1 to 10) {
        val epochStartTime = System.currentTimeMillis()
        logger.info(s"Starting epoch $epoch")

        customListener.reset()
        sparkModel.fit(trainingJavaRDD)
        val epochEndTime = System.currentTimeMillis()
        val epochDuration = epochEndTime - epochStartTime

        val modelOutputStream = new ByteArrayOutputStream()
        ModelSerializer.writeModel(sparkModel.getNetwork, modelOutputStream, false)
        val modelBytes = modelOutputStream.toByteArray
        val modelBroadcast = sc.broadcast(modelBytes)

        val trainingAccuracy = computeAccuracy(trainingDataSetsJavaRDD, modelBroadcast)
        val validationAccuracy = computeAccuracy(validationDataSetsJavaRDD, modelBroadcast)

        val runtime = Runtime.getRuntime
        val usedMemoryMB = (runtime.totalMemory - runtime.freeMemory) / (1024 * 1024)
        val totalMemoryMB = runtime.totalMemory / (1024 * 1024)
        val maxMemoryMB = runtime.maxMemory / (1024 * 1024)

        val osBean = ManagementFactory.getOperatingSystemMXBean.asInstanceOf[com.sun.management.OperatingSystemMXBean]
        val processCpuLoad = osBean.getProcessCpuLoad * 100
        val systemCpuLoad = osBean.getSystemCpuLoad * 100

        val totalShuffleReadBytes = customListener.taskMetricsData.map(_.shuffleReadBytes).sum
        val totalShuffleWriteBytes = customListener.taskMetricsData.map(_.shuffleWriteBytes).sum
        val taskDurations = customListener.taskMetricsData.map(_.duration)
        val maxTaskDuration = if (taskDurations.nonEmpty) taskDurations.max else 0L
        val minTaskDuration = if (taskDurations.nonEmpty) taskDurations.min else 0L
        val avgTaskDuration = if (taskDurations.nonEmpty) taskDurations.sum.toDouble / taskDurations.size else 0.0

        logger.info(s"Epoch $epoch - Duration: $epochDuration ms, CPU Load: $processCpuLoad%, Memory Usage: Used $usedMemoryMB MB, Max: $maxMemoryMB MB")

        val epochMetrics = EpochMetrics(
          epoch,
          epochEndTime,
          sparkModel.getScore,
          trainingAccuracy,
          validationAccuracy,
          learningRate,
          usedMemoryMB,
          totalMemoryMB,
          maxMemoryMB,
          totalShuffleReadBytes,
          totalShuffleWriteBytes,
          maxTaskDuration,
          minTaskDuration,
          avgTaskDuration,
          customListener.failedTaskCount,
          processCpuLoad,
          systemCpuLoad
        )
        metricsBuffer += epochMetrics

      }
      saveMetricsToCSV(metricsBuffer, statsOutputPath)

      logger.info("Model training completed")

      // End time for tracking training duration
      val trainingEndTime = System.currentTimeMillis()
      val trainingDuration = trainingEndTime - trainingStartTime
      logger.info(s"Total training duration: $trainingDuration ms")

      // Save the model
      try {
        saveModel(sc, model, modelOutputPath)
        logger.info(s"Model saved to $modelOutputPath")
      } catch {
        case ex: Exception => logger.error(s"Error saving model to $modelOutputPath", ex)
      }

      // Collect and save statistics
      val statsData = Seq(
        ("Total Training Time (ms)", trainingDuration.toString),
        ("Total Executors", sc.getExecutorMemoryStatus.size.toString),
        ("RDD Storage Information", getRDDStorageInfo(sc)),
        ("Gradient Stats", "Captured per iteration"),
        ("Learning Rate", model.getLayerWiseConfigurations.getConf(0).getLayer.getUpdaterByParam("W").asInstanceOf[Nesterovs].getLearningRate.toString),
        ("CPU/GPU Utilization", "Available via Spark UI"),
        ("Data Shuffling and Partitioning", "Tracked in Spark UI"),
        ("Batch Size", batchSize.toString)
      )

      try {
        //generateStatisticsFile(sc, statsData, statsOutputPath)
        logger.info(s"Statistics saved to $statsOutputPath")
      } catch {
        case ex: Exception => logger.error(s"Error saving statistics to $statsOutputPath", ex)
      }

    } catch {
      case ex: Exception =>
        logger.error("An error occurred during the execution of HW2", ex)
    } finally {
      sc.stop()
      logger.info("SparkContext stopped and application ended")
    }
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


  // Helper function to load token data
  def loadTokenData(sc: SparkContext, tokenDataPath: String): RDD[TokenData] = {
    sc.textFile(tokenDataPath).flatMap { line =>
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
  }

  // Helper function to load embedding data
  def loadEmbeddingData(sc: SparkContext, embeddingPath: String): RDD[EmbeddingData] = {
    sc.textFile(embeddingPath).flatMap { line =>
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
  }

  // Function to save model to HDFS
  def saveModel(sc: SparkContext, model: MultiLayerNetwork, modelOutputPath: String): Unit = {
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val outputStream: OutputStream = fs.create(new Path(modelOutputPath))
    ModelSerializer.writeModel(model, outputStream, true)
    outputStream.close()
  }

  // Function to generate statistics CSV file
  def generateStatisticsFile(sc: SparkContext, statsData: Seq[(String, String)], statsOutputPath: String): Unit = {
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val outputStream: OutputStream = fs.create(new Path(statsOutputPath))
    val writer = new BufferedWriter(new OutputStreamWriter(outputStream))
    writer.write("Metric,Value\n")
    statsData.foreach { case (metric, value) =>
      writer.write(s"$metric,$value\n")
    }
    writer.close()
  }

  def saveMetricsToCSV(metrics: Seq[EpochMetrics], path: String): Unit = {
    val fs = FileSystem.get(new java.net.URI(path), new org.apache.hadoop.conf.Configuration())
    val output = if (fs.exists(new Path(path))) fs.append(new Path(path)) else fs.create(new Path(path))
    val writer = new BufferedWriter(new OutputStreamWriter(output))

    // Write header only if the file is being created for the first time

    writer.write("Epoch,Timestamp,TrainingLoss,LearningRate,UsedMemoryMB,TotalMemoryMB,MaxMemoryMB,TotalShuffleReadBytes,TotalShuffleWriteBytes,MaxTaskDuration,MinTaskDuration,AvgTaskDuration,FailedTaskCount,ProcessCpuLoad,SystemCpuLoad,TrainingAccuracy,ValidationAccuracy\n")

    // Write each epoch's metrics
    metrics.foreach { metric =>
      writer.write(s"${metric.epoch},${metric.timestamp},${metric.trainingLoss},${metric.learningRate},${metric.usedMemoryMB},${metric.totalMemoryMB},${metric.maxMemoryMB},${metric.totalShuffleReadBytes},${metric.totalShuffleWriteBytes},${metric.maxTaskDuration},${metric.minTaskDuration},${metric.avgTaskDuration},${metric.failedTaskCount},${metric.processCpuLoad},${metric.systemCpuLoad},${metric.trainingAccuracy},${metric.validationAccuracy}\n")
    }
    writer.close()
  }

  def computeAccuracy(data: JavaRDD[DataSet], modelBroadcast: org.apache.spark.broadcast.Broadcast[Array[Byte]]): Double = {
    val predictions = data.rdd.mapPartitions { iter =>
      // Deserialize model
      val modelInputStream = new java.io.ByteArrayInputStream(modelBroadcast.value)
      val model = ModelSerializer.restoreMultiLayerNetwork(modelInputStream, false)

      iter.map { ds =>
        val output = model.output(ds.getFeatures)
        val label = ds.getLabels
        output.equals(label)
      }
    }
    val total = predictions.count()
    val correct = predictions.filter(identity).count()
    correct.toDouble / total
  }

}