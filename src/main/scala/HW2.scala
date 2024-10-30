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
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.io.File

object HW2 {

  case class TokenData(word: String, token: Int, frequency: Int)
  case class EmbeddingData(token: Int, word: String, embeddings: Array[Double])
  case class WindowedData(contextEmbedding: Array[Double], targetEmbedding: Array[Double])

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("HW2-SlidingWindowWithPositionalEmbedding").setMaster("local[*]")
    val sc = new SparkContext(conf)

    // Helper function to safely parse a string to an integer
    def safeToInt(s: String): Option[Int] = {
      try {
        Some(s.toInt)
      } catch {
        case _: NumberFormatException => None
      }
    }

    // Load token data with improved parsing to handle token arrays
    val tokensRDD: RDD[TokenData] = sc.textFile("src/main/resources/input/part-r-00000").flatMap { line =>
      val parts = line.split(",")
      if (parts.length == 3) {
        val word = parts(0)
        val tokensList = parts(1).replaceAll("[\\[\\]]", "").split(" ").flatMap(safeToInt)
        val frequency = safeToInt(parts(2))

        frequency match {
          case Some(freq) =>
            // Map each token in tokensList to a separate TokenData instance
            tokensList.map(token => TokenData(word, token, freq))
          case None =>
            println(s"Skipping line with invalid frequency: $line")
            None
        }
      } else {
        println(s"Skipping invalid line: $line")
        None
      }
    }

    // Helper function to safely parse a string to a Double
    def safeToDouble(s: String): Option[Double] = {
      try {
        Some(s.toDouble)
      } catch {
        case _: NumberFormatException => None
      }
    }

    // Load and parse the embedding data from CSV
    val embeddingsRDD: RDD[EmbeddingData] = sc.textFile("src/main/resources/input/embeddings.csv").flatMap { line =>
      val parts = line.split(",")

      if (parts.length > 2) {
        val tokenOpt = safeToInt(parts(0))
        val word = parts(1)
        val embeddings = parts.drop(2).flatMap(safeToDouble)  // Parse each embedding as Double

        tokenOpt match {
          case Some(token) if embeddings.nonEmpty =>
            Some(EmbeddingData(token, word, embeddings))
          case _ =>
            println(s"Skipping line with invalid token ID or embedding values: $line")
            None
        }
      } else {
        println(s"Skipping invalid line: $line")
        None
      }
    }

    // Broadcast embeddings for efficiency
    val embeddingMap = embeddingsRDD.collect().map(e => e.token -> e.embedding).toMap
    val embeddingMapBroadcast = sc.broadcast(embeddingMap)

    // Define parameters for sliding windows and embeddings
    val windowSize = 4
    val embeddingDim = embeddingMapBroadcast.value.head._2.length
    val positionalEmbedding = computePositionalEmbedding(windowSize, embeddingDim)
    val positionalEmbeddingBroadcast = sc.broadcast(positionalEmbedding)

    // Create sliding windows with positional embeddings using Spark
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
      val input = Nd4j.create(window.contextEmbedding)
      val label = Nd4j.create(window.targetEmbedding)
      new DataSet(input, label)
    }
    val trainingJavaRDD = trainingDataRDD.toJavaRDD()

    // Create and configure the model
    val model = createModel(numInputs = windowSize * embeddingDim, numOutputs = embeddingDim)
    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
      .batchSizePerWorker(32)
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2)
      .build()

    // Train the model using distributed Spark
    val sparkModel = new SparkDl4jMultiLayer(sc, model, trainingMaster)
    sparkModel.fit(trainingJavaRDD)

    // Save the trained model for inference
    ModelSerializer.writeModel(sparkModel.getNetwork, new File("src/main/resources/output/decoder_model.zip"), true)

    // Stop Spark Context
    sc.stop()
  }

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

  // Create and configure DL4J model
  def createModel(numInputs: Int, numOutputs: Int): MultiLayerNetwork = {
    val conf = new NeuralNetConfiguration.Builder()
      .list()
      .layer(new DenseLayer.Builder().nIn(10).nOut(10).activation(Activation.RELU).build())
      .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(10).nOut(10).activation(Activation.SOFTMAX).build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(10))
    model
  }
}