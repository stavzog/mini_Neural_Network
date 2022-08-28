package com.stavzog.minineuralnet

import kotlin.math.exp

class NeuralNetwork(private val inputN: Int, private val hiddenN: Int, private val outputN: Int) {

    private var weightsIH = Matrix(hiddenN, inputN).randomize()
    private var weightsHO = Matrix(outputN, hiddenN).randomize()
    private var biasH = Matrix(hiddenN, 1).randomize()
    private var biasO = Matrix(outputN, 1).randomize()

    constructor(nn: NeuralNetwork): this(nn.inputN, nn.hiddenN, nn.outputN) {
        weightsIH = nn.weightsIH.copy()
        weightsHO = nn.weightsHO.copy()
        biasH = nn.biasH.copy()
        biasO = nn.biasO.copy()
    }

    fun feedforward(inputArr: DoubleArray): DoubleArray {
        val input = Matrix(inputArr)
        //hidden layer output
        var hidden = weightsIH dot input
        hidden += biasH

        //activation func
        hidden.map { sigmoid(it.toFloat()) }

        //ouput layer output
        var output = weightsHO dot hidden
        output += biasO
        output.map { sigmoid(it.toFloat()) }

        return output.flatten()
    }

    fun backprop(inp: DoubleArray, targs: DoubleArray) {

        val lr = 0.1f

        val inputs = Matrix(inp)
        //hidden layer output
        var hidden = weightsIH dot inputs
        hidden += biasH

        //activation func
        hidden.map { sigmoid(it) }

        //ouput layer output
        var outputs = weightsHO dot hidden
        outputs += biasO
        outputs.map { sigmoid(it) }

        //Calculate output layer error
        val targets = Matrix(targs)
        val errorsO = targets - outputs

        //Calculate output layer gradient
        var gradients = Matrix.change(outputs) { dsigmoid(it) }
        gradients *= errorsO
        gradients *= lr

        //Calculate deltas
        val deltaWeightsHO = gradients dot hidden.transposed

        //adjust weights & bias
        weightsHO += deltaWeightsHO
        biasO += gradients

        //hidden layer errors
        val errorsH = weightsHO.transposed dot errorsO

        //Calculate hidden layer gradient
        var hiddenGradient = Matrix.change(hidden) { dsigmoid(it) }
        hiddenGradient *= errorsH * lr

        //Calculate hidden layer deltas
        val deltaWeightsIH = hiddenGradient dot inputs.transposed

        //adjust weights & bias
        weightsIH += deltaWeightsIH
        biasH += hiddenGradient
    }

    fun copy() = NeuralNetwork(this)

    fun mutate(transform: (Number) -> Number) {
        weightsHO.map(transform)
        weightsIH.map(transform)
        biasH.map(transform)
        biasO.map(transform)
    }
}

fun sigmoid(x: Number) = 1 / (1 + exp(-x.toDouble()))
fun dsigmoid(y: Number) = y.toDouble() * (1-y.toDouble())
