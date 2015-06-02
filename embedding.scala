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
     * args2: input sign graph (v1, v2, 1/-1)
     * args3: output model path
     * [args4: input model]
     * */

    val ITER = args(0).toInt
    val starting_alpha = args(1).toDouble
    val sc = new SparkContext(new SparkConf().setAppName("embedding"))
    val data = sc.textFile(args(2))
      .map(line => line.split(" ").map(v => v.toInt)).cache()

    val N = data.flatMap { v => List(v(0), v(1)) }.distinct().count()
    printf("%d %d\n", N, D) //num of vtx, dimension    

    var coordinates = new scala.collection.mutable.HashMap[Int, Vector[Double]]

    if (args.length == 5) {
      val initw = sc.textFile(args(4)).map(line => line.split(" ")).map(v => (v(0).toInt -> v.tail.map(a => a.toDouble)))
      println(initw.count())

      initw.collect().foreach { w =>
        coordinates += (w._1 -> DenseVector.apply(w._2).toVector)
      }

      if (coordinates.size != N) {
        println("dataN != loadin coordinates: " + coordinates.size)
        return
      }

    } else {
      for (i <- 0 to N.toInt - 1) {
        coordinates += (i -> DenseVector.fill(D) { 2 * rand.nextDouble - 1 }) //(-1,1)
      }
    }

    var iter = 0
    var isConverge = false
    var oldcost = 0
    while (iter < ITER && !isConverge) {

      var alpha = starting_alpha * (1 - iter * 1.0 / (ITER + 1)); // 自动调整学习速率
      if (alpha < starting_alpha * 0.00001) alpha = starting_alpha * 0.00001; // 学习速率有下限

      println("iteration: " + iter + " alpha: " + alpha)

      val cost = data.map { d =>
        {
          val edge = DataPoint(coordinates(d(0)), coordinates(d(1)), d(2) * 1.0);
          lbeta(edge.y * ((edge.a.-(edge.b)).dot(edge.a.-(edge.b)) - threshold))
        }
      }.reduce(_ + _)
      println("cost: " + cost)
      if (abs(oldcost - cost) < 1e-3) {
        println("Converage at iteration " + iter)
        isConverge = true;
      }

      val gradient = data.flatMap { d =>
        {
          val edge = DataPoint(coordinates(d(0)), coordinates(d(1)), d(2) * 1.0);
          val gradient = edge.y * omega(edge.y * ((edge.a.-(edge.b)).dot(edge.a.-(edge.b)) - threshold)) * 2 * (edge.a.-(edge.b))
          List(d(0) -> -1 * alpha * gradient, d(1) -> alpha * gradient)
        }
      }.reduceByKey(_ + _)
      gradient.collect().foreach {
        g => coordinates(g._1) += g._2
        //println(g)
      }

      //      if(gradient.toArray().forall(v => v._2.forall(v2 =>v2<1e-6)))
      //        isConverge=true
      iter += 1

      if (iter % 100 == 0) {
        coordinates.foreach {
          v => println(v._1 + " " + v._2.toArray.mkString(" "))
        }
      }
    }
    //val writer = new PrintWriter("/home/licui/code/embedding/embedding.out")
    sc.parallelize(coordinates.toSeq).map(v => v._1 + " " + v._2.toArray.mkString(" ")).saveAsTextFile(args(3))

    coordinates.foreach {
      v => println(v._1 + " " + v._2.toArray.mkString(" "))
    }

  }

}