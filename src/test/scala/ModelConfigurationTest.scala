import org.scalatest.funsuite.AnyFunSuite
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.learning.config.Nesterovs

class ModelConfigurationTest extends AnyFunSuite {

  test("Model should initialize with correct parameters") {
    val model: MultiLayerNetwork = HW2.createModel(numInputs = 400, numOutputs = 100, learningRate = 0.01, momentum = 0.9)

    assert(model.getLayerWiseConfigurations.getConf(0).getLayer.getUpdaterByParam("W").asInstanceOf[Nesterovs].getLearningRate == 0.01)
    assert(model.getLayerWiseConfigurations.getConf(0).getLayer.getUpdaterByParam("W").asInstanceOf[Nesterovs].getMomentum == 0.9)
  }
}