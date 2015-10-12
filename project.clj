(defproject dl4j-clj-example "0.1.0"
  :description "DL4J's Iris example straight port to Clojure"
  :dependencies [[org.clojure/clojure "1.6.0"]
                 [org.deeplearning4j/deeplearning4j-core "0.4-rc3.4"
                  :exclusions [ch.qos.logback/logback-classic]] ;;; Remove annoying logging
                 [org.apache.commons/commons-io "1.3.2"]
                 [org.slf4j/slf4j-nop "1.7.12"]                 ;;; Removes annoying logging
                 [org.nd4j/nd4j-jblas "0.4-rc3.5"]]             ;;; MacBook requirement
  :main ^:skip-aot dl4j-clj-example.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})                             ;;; Won't build correctly. See docs.
