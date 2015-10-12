(ns dl4j-clj-example.core
  (:import [org.apache.commons.io FileUtils]
           [org.deeplearning4j.datasets.iterator DataSetIterator SamplingDataSetIterator]
           [org.deeplearning4j.datasets.iterator.impl IrisDataSetIterator]
           [org.deeplearning4j.eval Evaluation]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf MultiLayerConfiguration Updater NeuralNetConfiguration NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.conf.layers OutputLayer OutputLayer$Builder RBM RBM$Builder RBM$HiddenUnit RBM$VisibleUnit]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.params DefaultParamInitializer]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.api IterationListener]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.dataset SplitTestAndTrain]
           [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction]
           [java.io]
           [java.nio.file Files]
           [java.nio.file Paths]
           [java.util Arrays]
           [java.util Random]))

(defn -main [& args]

  ;;; Values
  (set! Nd4j/MAX_SLICES_TO_PRINT -1)
  (set! Nd4j/MAX_ELEMENTS_PER_SLICE -1)
  (def num-rows 4)
  (def num-columns 1)
  (def output-num 3)
  (def num-samples 150)
  (def batch-size 150)
  (def iterations 5)
  (def split-train-num (int (* batch-size 0.8)))
  (def seed 111)
  (def listener-freq 1)

  ;;; Load Data
  (def iter (IrisDataSetIterator. batch-size num-samples))
  (def nxt (.next iter))
  (.normalizeZeroMeanZeroUnitVariance nxt)

  ;;; Split Data
  (def test-and-train (.splitTestAndTrain nxt split-train-num (Random. (int seed))))
  (def train (.getTrain test-and-train))
  (def tst (.getTest test-and-train))
  (set! Nd4j/ENFORCE_NUMERICAL_STABILITY true)

  ;;; Build Model
  (def conf
    (-> (NeuralNetConfiguration$Builder.)
        (.seed seed)
        (.iterations iterations)
        (.learningRate 1E+6)
        (.optimizationAlgo OptimizationAlgorithm/CONJUGATE_GRADIENT)
        (.l1 1e-1)
        (.regularization true)
        (.l2 2e-4)
        (.useDropConnect true)
        (.list 2)
        (.layer 0
                (->
                  (RBM$Builder. RBM$HiddenUnit/RECTIFIED RBM$VisibleUnit/GAUSSIAN)
                  (.nIn (* num-rows num-columns))
                  (.nOut 3)
                  (.weightInit WeightInit/XAVIER)
                  (.k 1)
                  (.activation "relu")
                  (.lossFunction LossFunctions$LossFunction/RMSE_XENT)
                  (.updater Updater/ADAGRAD)
                  (.dropOut 0.5)
                  (.build)))
        (.layer 1
                (->
                  (OutputLayer$Builder. LossFunctions$LossFunction/MCXENT)
                  (.nIn 3)
                  (.nOut output-num)
                  (.activation "softmax")
                  (.build)))
        (.build)))

  ;;; Create model
  (def model (MultiLayerNetwork. conf))
  (.init model)

  ;;; Set Listeners
  (.setListeners model (list (ScoreIterationListener. listener-freq)))

  ;;; Train
  (.fit model train)

  ;;; Evaluate and Log
  (def evaluation (Evaluation. output-num))
  (def output (.output model (.getFeatureMatrix tst)))

  (dotimes [i (.rows output)]
    (let [actual (-> tst
                     .getLabels
                     (.getRow i)
                     .toString
                     .trim)
          predicted (-> output
                        (.getRow i)
                        .toString
                        .trim)]
      (println [actual predicted])))

  (.eval evaluation (.getLabels tst) output)

  (println (.stats evaluation)))
