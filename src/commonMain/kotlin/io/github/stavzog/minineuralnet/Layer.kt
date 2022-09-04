package io.github.stavzog.minineuralnet
import kotlin.math.exp
import kotlin.math.max
import io.github.stavzog.minineuralnet.Matrix
import io.github.stavzog.minineuralnet.map
import io.github.stavzog.minineuralnet.print
import io.github.stavzog.minineuralnet.randomize
import kotlin.math.exp
import kotlin.math.max

//inputs and outputs will always have the shape of Matrix( 1, outputNodes)

interface Layer {

    fun build(input: Matrix): Matrix

    /**
     * Feed the input through the layer and return output [Matrix]
     *
     * @return each neuron output as a column in a [Matrix] with one row
     */
    fun feedforward(prevOutput: Matrix): Matrix // return output matrix

}

interface ParametrizedLayer: Layer {
    var weights: Matrix
    var biases: Matrix
    var activation: Activator?

    fun update(dw: Matrix, db: Matrix)

    val hasActivation get() = activation !== null
}

val Layer.isTrainable get() = this is ParametrizedLayer
val Layer.hasActivation: Boolean get() {
    if (this is Input) return false
    if (this is Activation) return true
    if (this is ParametrizedLayer)  return this.hasActivation
    return false
}

class Dense(private val nodes: Int, override var activation: Activator? = null): ParametrizedLayer {
    override lateinit var weights: Matrix
    override lateinit var biases: Matrix

    override fun build(input: Matrix): Matrix {
        weights = Matrix(input.cols, nodes).randomize()  //inputSize is the number of nodes of the previous layer
        biases = Matrix(1, nodes)
        return feedforward(input)
    }

    //output weighted sum of neuron inputs + bias
    override fun feedforward(prevOutput: Matrix): Matrix {
        val output = (prevOutput dot weights) + biases
        if(activation != null) return output.map {activation!!.f(it)}
        return (prevOutput dot weights) + biases
    }

    override fun update(dw: Matrix, db: Matrix){
        weights -= dw
        biases -= db
    }
}

class Activation(val activation: Activator): Layer {
    override fun build(input: Matrix): Matrix = feedforward(input)

    override fun feedforward(prevOutput: Matrix): Matrix = prevOutput.map { activation.f(it) }
}

class Input( val inputSize: Int): Layer {
    fun build(): Matrix = Matrix(1, inputSize)  //return a placeholder input matrix with zeros

    override fun build(input: Matrix) = build()

    override fun feedforward(prevOutput: Matrix): Matrix = prevOutput

}

class Network(vararg layers: Layer) {
    val layers: List<Layer> = listOf(*layers)

    val nLayers get() = layers.size
    val optimizer = StochasticGradientDescent(this)

    val inputLayer: Input

    init {
        if (layers.first() !is Input) throw IllegalArgumentException("First layer must be Input layer")
        inputLayer = layers.first() as Input
        buildLayers()
    }

    fun feedforward(input: Matrix): MutableList<Matrix> {
        if(input.rows != 1 || input.cols != inputLayer.inputSize)
            throw IllegalArgumentException("Input matrix is not of the right shape for this network")

        val layerOutputs = mutableListOf(inputLayer.feedforward(input))
        for (i in 1 until layers.size) {
            if(layers[i] is Input) continue
            layerOutputs.add(layers[i].feedforward(layerOutputs[i-1]))
        }
        return layerOutputs
    }

    /**
     * The network makes a prediction for the given [input]
     *
     * @param input a [Matrix] with one row and the inputs as columns
     * @return each output as a column in a [Matrix] with one row
     */
    fun predict(input: Matrix): Matrix {
        return feedforward(input).last()
    }

    //setup layer variables by feeding forward placeholder input matrix of zeros
    private fun buildLayers() {
        val input = inputLayer.build()
        var output = input
        layers.filter {it !is Input}.forEach {layer ->
            output = layer.build(output)
        }
    }

    fun fit(input: Matrix, actual: Matrix, epochs: Int, batchSize: Int) =
        optimizer.fit(input,actual,epochs,batchSize)
}

fun List<Layer>.getActivationOf(layer: ParametrizedLayer): Activator { //fix only dense
    val index = indexOf(layer)
    if (layer.hasActivation) return layer.activation!!

    return (this[index+1] as? Activation)?.activation ?: Activations.None

}

enum class Activations: Activator {
    ReLu {
        override fun f(x: Number): Double = max(0.0, x.toDouble())

        override fun der(y: Number): Double = if(y.toDouble() > 0) 1.0 else 0.0
    },

    Sigmoid {
        override fun f(x: Number): Double = 1 / (1 + exp(-x.toDouble()))

        override fun der(y: Number): Double = f(y) * (1-f(y))

    },

    None {
        override fun f(x: Number): Double = x.toDouble()
        override fun der(y: Number): Double = y.toDouble()
    }
}

interface Activator {
    fun f(x:Number): Double
    fun der(y: Number): Double
}

interface LossFunction

class MSE: LossFunction {

    companion object {
        fun loss(yTrue: Matrix, yPred: Matrix): Double =
            ( (yPred - yTrue) * (yPred - yTrue) ).flatten().average()

        fun delta(yTrue: Matrix, yPred: Matrix, activation: Activator) =
            ((yPred - yTrue) * yPred.map { activation.der(it) })
    }
}

internal fun mean(x: Matrix, axis: Int = 1): Matrix {
    val m: Matrix = when(axis) {
        0 -> x.transposed
        1 -> x //
        else -> throw IllegalArgumentException("Axis can be 0 or 1 for 2d array (matrix)")
    }
    val result = Matrix(1, m.rows)
    for(i in 0 until m.rows) {
        var sum = 0.0
        for (j in 0 until m.cols) {
            sum += m[i,j].toDouble()
        }
        result[0,i] = sum / m.cols
    }
    return result // [mean of m.row1, mean of m.row2 ...]
}

class StochasticGradientDescent(private val net: Network) {
    val learningRate = 0.3

    fun backprop(yTrue: Matrix, layerOutputs: List<Matrix>) {
        var activation = when(val last = net.layers.last()) {
            is Activation -> last.activation
            is ParametrizedLayer -> last.activation ?: throw IllegalStateException("No activation function found for last layer")
            else -> throw IllegalStateException("Invalid last layer type")
        }

        val outputLayer = when( net.layers.last()) {
            is Activation -> net.layers[net.nLayers-2] as ParametrizedLayer
            is ParametrizedLayer -> net.layers.last() as ParametrizedLayer
            else -> throw IllegalStateException("Invalid last layer type")
        }

        //for output layer
        var delta = MSE.delta(yTrue, layerOutputs.last(), activation)
        var dw = layerOutputs[layerOutputs.size - 2].transposed dot delta
        (dw * learningRate).print()
        (mean(delta,0) * learningRate).print()
        outputLayer.update(dw * learningRate,mean(delta,0) * learningRate)

        for(i in net.nLayers-1 downTo 1) {
            val layer = if(net.layers[i].isTrainable) (net.layers[i] as ParametrizedLayer) else continue
            activation = net.layers.getActivationOf(layer)
            delta = (delta dot layer.weights.transposed) * layerOutputs[i].map { activation.der(it) }
            dw = layerOutputs[i-1].transposed dot delta
            layer.update(
                dw * learningRate,
                delta * learningRate
            )
        }

    }

    fun fit(x: Matrix, y: Matrix, epochs: Int, batchSize: Int) {
        //x list of inputs, y list of correct outputs

        if(x.rows != y.rows)
            throw IllegalArgumentException("Input matrix does not match with Actual output matrix")

        if(batchSize > x.rows)
            throw IllegalArgumentException("BatchSize cannot be greater than the ")

        for(epoch in 0..epochs) {
            val seed = (0 until x.rows).toList().shuffled().slice(0 until batchSize)
            val batchX = x.sliceRows(seed)
            val batchY = y.sliceRows(seed)

            for (i in 0 until batchX.rows) {
                val input = batchX.sliceRows(listOf(i))
                val yTrue = batchY.sliceRows(listOf(i))
                val layerOutputs = net.feedforward(input)
                backprop(yTrue,layerOutputs)
            }

        }
    }

}

fun Matrix.sliceRows(indices: Iterable<Int>): Matrix {
    val result = Matrix(indices.count(),cols)
    for(i in 0 until result.rows) {
        for(j in 0 until cols) {
            for(k in indices) {
                result[i,j] = this[k,j]
            }
        }
    }
    return result
}

private fun Matrix.sum(): Double {
    var sum = 0.0
    forEachIndexed { i, j, _ ->
        sum += this[i,j].toDouble()
    }
    return sum
}

fun mutableMatrixOf(rows:Int, cols:Int, vararg elements: Number) =
    Matrix(rows,cols, elements.asList().toMutableList())