package com.stavzog.minineuralnet

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class MatrixTest {

    @Test
    fun testOperations() {
        //test equals
        assertEquals(true, uni(3) == uni(3))
        assertEquals(false, uni(3, r=4) == uni(3))
        assertEquals(false, uni(3) == uni(4))

        //test element wise addition
        testElementWise(action= { x,y -> x + y },r=4, expected= uni(1.0+1.0,4))
        //test element wise subtraction
        testElementWise(action= { x,y -> x - y }, x=3.0, expected= uni(3.0-3.0))
        //test element wise multiplication
        testElementWise(action= { x,y -> x * y },c=5, x=4.0, expected= uni(4.0*4.0,c=5))

        //test scalar addition
        testScalar(action={x -> x + 3}, x=5.0, expected= uni(5.0+3.0))
        //test scalar subtraction
        testScalar(action={x -> x - 7}, r=7, expected= uni(1.0-7.0,7))
        //test scalar multiplication
        testScalar(action={x -> x*4}, x=2.0, c=8, expected= uni(2.0*4.0,c=8))

        //if the above work then forEachIndexed and mapIndexed work

        //test transposed
        assertEquals(true, uni(3,2,3).transposed == uni(3, 3,2))

    }

    private fun uni(x: Number, r: Int = 2, c: Int = 2): Matrix {
        return Matrix(r,c).map { x }
    }

    private fun testElementWise(expected: Matrix, action: (x: Matrix, y: Matrix) -> Matrix, r:Int=2, c:Int=2, x:Double=1.0) {
        assertEquals( expected, action(uni(x,r,c),uni(x,r,c)) )
        assertFailsWith<IllegalArgumentException>{ action(uni(x,r,c),uni(x,r+3,c+2)) }
    }

    private fun testScalar(expected: Matrix, action: (x: Matrix) -> Matrix, r:Int=2, c:Int=2, x:Double=1.0) {
        assertEquals(expected , action(uni(x,r,c)) )
    }

}
