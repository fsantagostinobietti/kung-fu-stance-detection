/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

async function start() {
  // load 4-stances classifier 
  const tfliteModel = await tflite.loadTFLiteModel('model-stances.tflite');
  console.info(tfliteModel);

  imgTensor = tf.browser.fromPixels(document.querySelector("img"));
  console.info(imgTensor.shape);
  const inputTensor = tf.image
    // Resize.
    .resizeBilinear(imgTensor, [224, 224])  // shape [224, 224, 3]
    // Normalize.
    .expandDims() // shape [1, 224, 224, 3]
    .div(127.5)
    .sub(1);  // values in range (-1, +1)
  console.info(inputTensor.shape);
  //inputTensor.print();

  // Run the inference and get the output tensors.
  const outputTensor = tfliteModel.predict(inputTensor);
  outputTensor.print();
}

start();
console.info('End')