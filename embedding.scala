package edu.cn.pku.net.embedding
import breeze.linalg.{ Vector, DenseVector }
import java.util.Random
import scala.math._
import org.apache.spark.SparkContext
import org.apache.spark.graphx.Graph
import org.apache.spark._
import java.io.PrintWriter
/**
 * @author Cui LI
 */
object embedd {

  val D = 2 //Num of dimensions
  //Num of iterations
  val rand = new Random(42)

  val threshold = 10
  case class DataPoint(a: Vector[Double], b: Vector[Double], y: Double)

  def omega(dist: Double): Double =
    pow(1 + exp(0 - dist), -1)

  def lbeta(cost: Double): Double =
    log(1 + exp(cost))

  def main(args: Array[String]) {
    /*
     * args0: ITER: num of iterations
     * args1: starting_alpha    
     * [args2: input model]
     * */

    val ITER = args(0).toInt
    val starting_alpha = args(1).toDouble
    val sc = new SparkContext(new SparkConf().setAppName("embedding"))
    val pdata = sc.textFile("hdfs://cp01.amazingstore.org:9000/user/licui/data/sym_adjlist.graph")
      .flatMap(line => {
        val vtxs = line.split("\t").map(v => v.toInt);
        vtxs.tail.map(v => (vtxs.head, v, 1))
      })
    val ndata = sc.textFile("hdfs://cp01.amazingstore.org:9000/user/licui/data/sym_adjlist.graph.negative")
      .flatMap(line => {
        val vtxs = line.split("\t").map(v => v.toInt);
        vtxs.tail.map(v => (vtxs.head, v, -1))
      })
    val data = pdata.union(ndata).cache()
    val N = data.flatMap { v => List(v._1, v._2) }.distinct().count()
    printf("%d %d\n", N, D) //num of vtx, dimension    

    var coordinates = new scala.collection.mutable.HashMap[Int, Vector[Double]]

    if (args.length == 4) {
      val initw = sc.textFile(args(3)).map(line => line.split(" ")).map(v => (v(0).toInt -> v.tail.map(a => a.toDouble)))
      println(initw.count())

      initw.collect().foreach { w =>
        coordinates += (w._1 -> DenseVector.apply(w._2).toVector)
      }

      if (coordinates.size != N) {
        println("dataN != loadin coordinates: " + coordinates.size)
        return
      }

    } else {

      data.flatMap { v => List(v._1, v._2) }.distinct().collect().foreach { v =>
        coordinates += (v -> DenseVector.fill(D) { 2 * rand.nextDouble - 1 })
      }

    }
    Runtime.getRuntime().exec("hadoop fs -rmr /user/licui/out/embedding/")
    var iter = 0
    var isConverge = false
    var oldcost = 0
    while (iter < ITER && !isConverge) {

      var alpha = starting_alpha * (1 - iter * 1.0 / (ITER + 1)); // 自动调整学习速率
      if (alpha < starting_alpha * 0.00001) alpha = starting_alpha * 0.00001; // 学习速率有下限

      println("iteration: " + iter + " alpha: " + alpha)

      val gradient = data.flatMap { d =>
        {
          val edge = DataPoint(coordinates(d._1), coordinates(d._2), d._3 * 1.0);
          val gradient = edge.y * omega(edge.y * ((edge.a.-(edge.b)).dot(edge.a.-(edge.b)) - threshold)) * 2 * (edge.a.-(edge.b))
          List(d._1 -> -1 * alpha * gradient, d._2 -> alpha * gradient)
        }
      }.reduceByKey(_ + _)
      gradient.collect().foreach {
        g => coordinates(g._1) += g._2
      }

      iter += 1
      if (iter < 10 || iter % 100 == 0) {
        val cost = data.map { d =>
          {
            val edge = DataPoint(coordinates(d._1), coordinates(d._2), d._3 * 1.0);
            lbeta(edge.y * ((edge.a.-(edge.b)).dot(edge.a.-(edge.b)) - threshold))
          }
        }.reduce(_ + _)
        println("cost: " + cost)
      }

      if (iter % 100 == 0) {
        val path = "hdfs://cp01.amazingstore.org:9000/user/licui/out/embedding/iter" + iter / 500

        sc.parallelize(coordinates.toSeq).map(v => v._1 + " " + v._2.toArray.mkString(" ")).saveAsTextFile(path)

      }
    }
    //val writer = new PrintWriter("/home/licui/code/embedding/embedding.out")
    sc.parallelize(coordinates.toSeq).map(v => v._1 + " " + v._2.toArray.mkString(" ")).saveAsTextFile("hdfs://cp01.amazingstore.org:9000/user/licui/out/embedding/iter" + iter)

  }

}