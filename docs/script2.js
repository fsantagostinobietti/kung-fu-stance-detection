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
  status.innerText = 'TensorFlow.js - version: ' + tf.version.tfjs;
}

//TODO load from 'labels.txt' 
const CLASS_LABEL = ['back_stance', 'bow_stance', 'cat_stance', 'horse_stance', 'unrecognised_stance'];


// classifier tflite model
let tfliteAllModel;

function classifyPic() {
  const imgTensor = tf.browser.fromPixels(document.querySelector("img"));
  console.info(imgTensor.shape);
  const inputTensor = tf.image
    // Resize.
    .resizeBilinear(imgTensor, [224, 224])  // shape [224, 224, 3]
    // Normalize.
    .expandDims() // shape [1, 224, 224, 3]
    .div(127.5)
    .sub(1);  // values in range (-1, +1)
  //console.info(inputTensor.shape);
  //inputTensor.print();

  // Run the inference and get the output tensors.
  const outputTensor = tfliteAllModel.predict(inputTensor);  // shape [1, 4]
  outputTensor.print();
  //console.info("outputTensor shape:", outputTensor.shape);

  let classIdx = outputTensor
    // find index of max probability
    .argMax(axis=1)
    // convert tensor in js array and pick value
    .arraySync()[0];
  let classProb = outputTensor
    // convert tensor in js array and pick value
    .arraySync()[0][classIdx];
  console.info("Class label:", CLASS_LABEL[classIdx], 
    " - Probability:", (100*classProb).toFixed()+"%");
  // detect unrecognised stance
  if (classProb < 0.45) {
    classIdx = CLASS_LABEL.length-1;
    classProb = (1-classProb)/(1-1/(CLASS_LABEL.length-1));
  }

  // display result
  document.querySelector('#pic-classification > img')
    .src = 'imgs/'+CLASS_LABEL[classIdx].toLowerCase()+'.png';
  document.querySelector('#pic-classification > p')
    .innerHTML = CLASS_LABEL[classIdx].toUpperCase()+" (Prob. "+(100*classProb).toFixed()+"%)";
}

async function loadModel() {
  // load 4-stances classifier tflite model
  tfliteAllModel = await tflite.loadTFLiteModel('model-all-stances.tflite');
  //console.info(tfliteAllModel);
}

function getRandomInt(max) {
  return Math.floor(Math.random() * max);
}

const NUM_PICS = 12;

function loadPic(src) {
  document.getElementById("pic").setAttribute('src',src);
}

function loadInitialPic() {
  loadPic('imgs/img'+getRandomInt(NUM_PICS)+'.jpg');
}

function populatePicPalette() {
  let picContainer = document.getElementById('pic-palette');
  for (i=0; i<NUM_PICS ;++i) {
    let img = document.createElement("img");
    img.src = 'imgs/img'+i+'.jpg';
    img.height = 50;
    img.width  = 50;
    img.setAttribute('onclick', "loadPic(this.src);");

    picContainer.appendChild(img);
  }
}

function loadPicFromFile(evt) {
  var tgt = evt.target || window.event.srcElement,
      files = tgt.files;

  // Test FileReader support
  if (FileReader && files && files.length) {
      var fr = new FileReader();
      fr.onload = function () {
        loadPic(fr.result);  
        //document.getElementById(outImage).src = fr.result;
      }
      fr.readAsDataURL(files[0]);
  }
  // Not supported
  else {
      alert('Picture load not supported!');
  }
}

//
// MAIN
//

// action: trigger classification when new image is loaded in <img>
document.getElementById('pic').onload = function() {
  classifyPic();
}
// action: enable image loading  
document.getElementById('pic-picker').onchange = function (evt) {
  loadPicFromFile(evt);
}

loadModel().then( function() {
  console.info('Model tflite loaded.');

  loadInitialPic();
  populatePicPalette();
});

