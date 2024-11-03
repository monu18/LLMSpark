package utils

import com.typesafe.config.{Config, ConfigFactory}

object ConfigUtil {
  // Load configuration from application.conf
  private val config: Config = ConfigFactory.load()

  // Define an immutable case class for configuration fields
  case class AppConfig(
                        embeddingPath: String,
                        tokenDataPath: String,
                        modelOutputPath: String,
                        statsOutputPath: String,
                        master: String,
                        learningRate: Double,
                        momentum: Double,
                        batchSize: Int,
                        windowSize: Int
                      )

  // Load values from application.conf
  private val defaultConfig: AppConfig = AppConfig(
    embeddingPath = config.getString("app.paths.embeddingPath"),
    tokenDataPath = config.getString("app.paths.tokenDataPath"),
    modelOutputPath = config.getString("app.paths.modelOutputPath"),
    statsOutputPath = config.getString("app.paths.statsOutputPath"),
    master = config.getString("app.master"),
    learningRate = config.getDouble("model.learningRate"),
    momentum = config.getDouble("model.momentum"),
    batchSize = config.getInt("model.batchSize"),
    windowSize = config.getInt("model.windowSize")
  )

  // Option to hold the final configuration
  private var _finalConfig: Option[AppConfig] = None

  // Initialize final configuration with command-line arguments or defaults
  def initializeConfig(args: List[String]): Unit = {
    if (_finalConfig.isEmpty) {
      _finalConfig = Some(defaultConfig.copy(
        embeddingPath = args.headOption.getOrElse(defaultConfig.embeddingPath),
        tokenDataPath = args.lift(1).getOrElse(defaultConfig.tokenDataPath),
        modelOutputPath = args.lift(2).getOrElse(defaultConfig.modelOutputPath),
        statsOutputPath = args.lift(3).getOrElse(defaultConfig.statsOutputPath),
        master = args.lift(4).getOrElse(defaultConfig.master),
        learningRate = defaultConfig.learningRate,
        momentum = defaultConfig.momentum,
        batchSize = defaultConfig.batchSize,
        windowSize = defaultConfig.windowSize
      ))
    }
  }

  // Access the final configuration
  def finalConfig: AppConfig = _finalConfig.getOrElse(defaultConfig)
}