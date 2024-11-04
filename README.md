# Parallelized Model Training with Metrics Tracking and Management on Spark and DeepLearning4j
# Author: Monu Kumar
# Email: mkuma47@uic.edu

## Introduction
In this homework assignment we leverage Apache Spark and Deeplearning4j to train a neural network on large-scale text data within a distributed computing environment. By applying a sliding window approach and incorporating positional embeddings, the model captures sequential context for enhanced language modeling. Key components include data tokenization, sliding window generation, model setup, and accuracy assessment, with comprehensive metrics tracked across each epoch to monitor resource utilization and model performance. The setup is optimized for Spark clusters, facilitating efficient processing and analysis of extensive datasets.

**Video Link:** [] (The video explains the deployment of the Spark application in the AWS EMR Cluster and the project structure.)

## Environment
- **OS:** Mac
- **IDE:** IntelliJ IDEA 2024.2.1 (Ultimate Edition)
- **Scala Version:** 2.12.15
- **SBT Version:** 1.10.3
- **Spark Version:** 3.3.0

## Running the Test File
Test files can be found under the directory `src/test`:
```bash
sbt clean compile test
```

## Running the Project
1. **Clone this repository:**
   ```bash
   git clone https://github.com/monu18/LLMSpark
   ```
2. **Navigate to the Project:**
   ```bash
   cd LLMSpark
   ```
3. **Open the project in IntelliJ:**  
   [How to Open a Project in IntelliJ](https://www.jetbrains.com/help/idea/import-project-or-module-wizard.html#open-project)

## Configuration Utility for LLMForge
The `ConfigUtil` object, located in `src/main/scala/utils/ConfigUtil.scala`, is responsible for managing application configurations for the LLMSpark project. This utility loads default configurations from the `application.conf` file found in `src/main/resources/`, allowing for flexibility in path settings based on command-line arguments.

### Configuration Behavior:
- **Initialization:** The `initializeConfig` method accepts a list of command-line arguments. If provided, these arguments will override the corresponding default configuration values. If no arguments are given, the application defaults to the paths defined in `application.conf`.
- **Final Configuration Access:** The application can access the final configuration via the `finalConfig` method, ensuring that once set, the configuration remains immutable.

### Configuration Paths:
- **embeddingCsvPath:** The path for the CSV file containing embeddings (default: `src/main/resources/input/embeddings.csv`).
- **tokenDataPath:** The path of tokens (default: `src/main/resources/input/part-r-00000`).
- **modelOutputPath:** The output path of model(zipped) (default: `src/main/resources/output/decoder_model.zip`).
- **statisticsPath:** The  (default: `src/main/resources/output/training_stats.csv`).
- **masterCluster:** Master cluster (default: `local[*]`).

## Project Structure
The project comprises the following key components:

1. **Initialization and Configuration**  
   The program begins with necessary imports and case class definitions:

	1.	Imports: Essential libraries are imported to handle Spark, Deeplearning4j, and HDFS operations. Classes like EpochMetrics and TaskMetricsData are defined to structure metrics tracking data.
	2.	CustomSparkListener: Extends SparkListener to track task metrics in Spark, such as task duration, shuffle read/write bytes, and failed tasks. This data is reset at each epoch to ensure epoch-specific tracking.
	3.	Configuration Setup: Configuration parameters (paths, training parameters) are initialized through ConfigUtil, and Spark context (SparkContext) is created with app-specific settings. This setup ensures that the Spark cluster and configuration paths are accessible throughout the job.

2. **Data Preparation**  
   Once initialized, the code proceeds with loading and processing the input data:

	1.	Data Loading and Broadcasting:
	•	Token Data and Embedding Data: Loaded from specified paths and broadcasted across nodes in the cluster for efficient access. This reduces the need for repeated data retrieval.
	2.	Sliding Window Generation with Positional Embeddings:
	•	createSlidingWindowsWithPositionalEmbedding generates input windows by slicing token sequences into smaller windows. Each window includes positional embeddings to retain token order information, which is especially useful for sequence-based models.
	3.	Training Data Preparation:
	•	Each sliding window is converted into a DataSet containing features (context embeddings) and labels (target embeddings) for training, formatted as JavaRDD[DataSet].

3. **Model Setup and Distributed Training**  
   With the data prepared, the model setup and training process are configured:

	1.	Model Creation and Configuration:
	•	The neural network model is configured with two layers: a dense layer and an output layer, with Nesterovs optimizer for gradient descent. Deeplearning4j’s ParameterAveragingTrainingMaster is used to manage distributed training through parameter averaging.
	2.	Training Loop with Metrics Collection:
	•	Epoch Training: For each epoch, the model is trained on the distributed dataset, with CustomSparkListener capturing metrics such as task duration and shuffle data usage.
	•	Model Serialization: The model is serialized and broadcasted to allow accuracy computation on the training and validation sets.
	•	Accuracy Calculation: Accuracy for training and validation datasets is computed using computeAccuracy, which compares the model’s predictions with actual labels.

4. **Metrics Logging, Model Saving, and Final Statistics**  
   Following training, the program logs metrics and saves the model:

	1.	Metrics Collection and Saving:
	•	System Resource Tracking: Memory usage and CPU load are tracked using Java’s ManagementFactory, with metrics saved to EpochMetrics.
	•	Final Metrics Logging: After all epochs, metricsBuffer is saved to a CSV file using saveMetricsToCSV to allow for further analysis of training progress.
	2.	Model and Statistics Saving:
	•	The trained model is saved to the specified HDFS path, and runtime statistics are logged to a statistics file, capturing information such as total training time, RDD storage info, and shuffle data usage.


## Prerequisites
Before starting the project, ensure you have the following tools and accounts set up:
- **SPARK:** Install and configure Spark on your local machine or cluster.
- **AWS Account:** Create an AWS account and familiarize yourself with AWS Elastic MapReduce (EMR).
- **Java:** Ensure that Java is installed and properly configured.
- **Git and GitHub:** Use Git for version control and host your project repository on GitHub.
- **IDE:** Choose an Integrated Development Environment (IDE) for coding and development.

## Conclusion
In summary, this project leverages Spark and Deeplearning4j to efficiently train a distributed neural network on large text data. By utilizing sliding windows with positional embeddings, it captures sequential token relationships, essential for tasks requiring contextual understanding. The code’s design integrates robust metrics tracking for each training epoch, including accuracy, memory, and CPU usage, enabling comprehensive model evaluation. This setup demonstrates an effective, scalable approach for distributed training, suited for handling large datasets and deep learning tasks on a Spark cluster.

