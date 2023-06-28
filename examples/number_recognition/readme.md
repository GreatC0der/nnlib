## Recognition of hand-written numbers.
#### [⚠ This project lacks data in the data set, you are very welcome to help. ⚠](https://github.com/GreatC0der/nnlib/issues/1)

Look through the code and make sure that you understand most of it.

I already tweaked it for you, the only thing left to do is to train the neural network.

To do so, run:
`
  cargo run --release -- teach
`
But watch out, you shouldn't overtrain it. The weights are generated randomly, so this behaviour might occur:
- Works perfectly if you stop training at the right time.
- Neural network becomes NaN.
- Wierd things happen.

Finally, to test the program you need to run:
`
  cargo run --release -- run dataset/4/2.png
`  <--- *you can(and should) change this!*

Congratulations! Now you can try to create your own neural network using this library. 

When your project is ready, open an issue, so we can add it to successful projects that use our library!
