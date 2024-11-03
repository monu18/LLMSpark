import sbt.Keys.libraryDependencies
import sbtassembly.AssemblyPlugin.autoImport.*

ThisBuild / version := "0.1.0-SNAPSHOT"

// Set Scala version to ensure DL4J and Spark compatibility
ThisBuild / scalaVersion := "2.12.15"

lazy val root = (project in file("."))
  .settings(
    name := "LLMSpark",
    version := "0.1.0-SNAPSHOT"
  )

// Version Definitions
val logbackVersion = "1.5.6"
val slf4jLoggerVersion = "2.0.12"
val typeSafeConfigVersion = "1.4.3"
val breezeVersion = "2.1.0"
val deepLearning4jVersion = "1.0.0-M2.1"
val sparkVersion = "3.3.0"  // Fully compatible with DL4J Parameter Server

// Library Dependencies
libraryDependencies ++= Seq(
  "ch.qos.logback" % "logback-classic" % logbackVersion,
  "org.slf4j" % "slf4j-api" % slf4jLoggerVersion,
  "com.typesafe" % "config" % typeSafeConfigVersion,

  // Breeze for numerical processing
  "org.scalanlp" %% "breeze" % breezeVersion,

  // Spark dependencies for Spark 3.3.0 (with Scala 2.12 compatibility)
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,

  // DL4J with Gradient Sharing for Distributed Training
  "org.deeplearning4j" % "deeplearning4j-core" % deepLearning4jVersion,
  "org.deeplearning4j" % "dl4j-spark-parameterserver_2.12" % deepLearning4jVersion,
  "org.nd4j" % "nd4j-native-platform" % deepLearning4jVersion,
  // Optional: Add "nd4j-native" for platform-specific native support
  // "org.nd4j" % "nd4j-native" % deepLearning4jVersion,

  // Testing libraries
  "org.scalatest" %% "scalatest" % "3.2.18" % Test,
  "org.mockito" %% "mockito-scala" % "1.16.42" % Test
)

Test / parallelExecution := false
Test / fork := true
// Main Class Configuration (Optional)
Compile / mainClass := Option("HW2")

assembly / assemblyJarName := "LLMSpark.jar" // Name of your jar file

assembly / mainClass := Some("HW2") // Ensure this points to your main class
// Merging Strategies for Assembly
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) =>
    xs match {
      case "MANIFEST.MF" :: Nil => MergeStrategy.discard
      case "services" :: _      => MergeStrategy.concat
      case _                    => MergeStrategy.discard
    }
  case "reference.conf" => MergeStrategy.concat
  case x if x.endsWith(".proto") => MergeStrategy.rename
  case x if x.contains("hadoop") => MergeStrategy.first
  case _ => MergeStrategy.first
}

assembly / assemblyOption := (assembly / assemblyOption).value.withIncludeScala(true)