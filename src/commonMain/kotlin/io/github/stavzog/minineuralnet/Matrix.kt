package io.github.stavzog.minineuralnet

import kotlin.math.roundToInt
import kotlin.random.Random

class Matrix(val rows: Int, val cols: Int) {
    var data = MutableList<Number>(rows * cols) {0}


    val isEmpty get() = data.isEmpty()
    val transposed get() = transpose()

    constructor(array: DoubleArray): this(array.size,1) {
        forEachIndexed { i, _, _ ->
            this[i,0] = array[i]
        }
    }

    constructor(rows: Int, cols: Int, list: MutableList<Number>): this(rows,cols) {
        data = list
    }

    operator fun get(r: Int, c: Int) = data[r*cols+c]
    operator fun set(r: Int, c: Int, value: Number) { data[r*cols+c] = value }

    operator fun plus(n: Number) = map {it.toDouble()+n.toDouble()}
    operator fun plus(m: Matrix): Matrix {
        if(rows != m.rows || cols != m.cols)
            throw IllegalArgumentException("Matrices do not match")
        return mapIndexed { i, j, it -> it.toDouble() + m[i,j].toDouble() }
    }
    operator fun minus(m: Matrix): Matrix {
        if(rows != m.rows || cols != m.cols)
            throw IllegalArgumentException("Matrices do not match")
        return mapIndexed { i, j, it -> it.toDouble() - m[i,j].toDouble() }
    }
    operator fun minus(n: Number) = map {it.toDouble() - n.toDouble()}
    operator fun times(n: Number) = map { it.toDouble() * n.toDouble() }
    operator fun times(m: Matrix): Matrix {
        if(rows != m.rows || cols != m.cols)
            throw IllegalArgumentException("Matrices do not match")
        return mapIndexed { i, j, it -> it.toDouble() * m[i,j].toDouble() }
    }

    override operator fun equals(other: Any?): Boolean {
        if (other !is Matrix) return false
        if (data == other.data && rows == other.rows && cols == other.cols) return true
        return false
    }

    override fun hashCode(): Int = hashCodeOf(data, rows, cols)

    private fun hashCodeOf(vararg values: Any?) =
        values.fold(0) { acc, value ->
            (acc * 31) + value.hashCode()
        }

    fun flatten(): DoubleArray {
        var arr = doubleArrayOf()
        forEachIndexed { _, _, el ->
            arr += el.toDouble()
        }
        return arr
    }

    inline fun forEachIndexed(action: (i: Int, j: Int, it: Number) -> Unit): Matrix {
        for(i in 0 until rows) {
            for (j in 0 until cols) {
                action(i,j,this[i,j])
            }
        }
        return this
    }


    //returns new matrix with rows, cols
    inline fun mapIndexed(action: (r: Int,c: Int,el: Number) -> Number): Matrix =
        createMatrix(rows, cols) { i , j -> action(i,j,this[i,j])}


    infix fun dot(m: Matrix): Matrix {
        if(cols != m.rows) throw IllegalArgumentException("Matrix A columns must match Matrix B rows")
        return createMatrix(rows, m.cols) {i,j ->
            var sum = 0.0
            for(k in 0 until cols) sum+= this[i,k].toDouble() * m[k,j].toDouble()
            sum
        }
    }

    private fun transpose() =
        createMatrix(cols,rows) { r,c -> this[c,r] }

    fun copy(): Matrix =
        createMatrix(rows,cols) { r ,c -> this[r,c] }

}

fun Matrix.print(): Matrix {
    for(i in 0 until rows) {
        print("[")
        for (j in 0 until cols) {
            print("${(this[i,j].toDouble() * 100).roundToInt() / 100.0}\t")
        }
        println("\b]")
    }
    println()
    return this
}

fun Matrix.randomize() = forEachIndexed { r, c, _ ->
    this[r,c] = Random.nextDouble() * 2 - 1
}

//Random.nextDouble() * 2 - 1

fun Matrix.map(action: (Number) -> Number) = mapIndexed { _, _, el -> action(el)}

inline fun createMatrix(rows: Int, cols: Int, init: (r: Int, c: Int) -> Number): Matrix {
    val list = mutableListOf<Number>()
    for(i in 0 until rows) {
        for(j in 0 until cols) {
            list.add(init(i,j))
        }
    }
    return Matrix(rows,cols,list)
}


