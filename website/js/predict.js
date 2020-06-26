import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('http://localhost:5500/html/tip-the-scales.html'); //loads model through the server that this is hosted on

if(model) {
    console.log("It works!");
}

