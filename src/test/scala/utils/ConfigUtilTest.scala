package utils

import org.scalatest.funsuite.AnyFunSuite

class ConfigUtilTest extends AnyFunSuite {

  test("Configuration loading should retrieve correct values") {
    ConfigUtil.initializeConfig(List())
    val config = ConfigUtil.finalConfig

    assert(config.embeddingPath == "src/main/resources/input/embeddings.csv")
    assert(config.tokenDataPath == "src/main/resources/input/part-r-00000")
    assert(config.learningRate == 0.01)
    assert(config.windowSize == 4)
    assert(config.master == "local[*]")
  }
}