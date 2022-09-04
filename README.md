# Renewed Library
A new simple neural network library inspired by kotlin-dl.
Currently it supports neural network creation with dense layers but the stochastic gradient descent algorithm still needs work. As such the api remains unusable.

## Making a network in the new library
```kotlin
val model = Network(
  Input(2),
  Dense(2),
  Dense(2)
)
```
