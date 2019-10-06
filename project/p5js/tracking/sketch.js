// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
PoseNet example using p5.js
=== */

let video;
let poseNet;
let stats;
let poses = [];
let w = 640;
let h = 480;

let isModelReady = false;

let minPoseConfidence;
let minPartConfidence;

const defaultQuantBytes = 2;

const defaultMobileNetMultiplier = 0.75; //isMobile() ? 0.50 : 0.75;
const defaultMobileNetStride = 16;
const defaultMobileNetInputResolution = 257;

const defaultResNetMultiplier = 1.0;
const defaultResNetStride = 32;
const defaultResNetInputResolution = 257;

const config = {
  architecture: 'MobileNetV1',
  imageScaleFactor: 0.3,
  outputStride: defaultMobileNetStride,
  flipHorizontal: false,
  minConfidence: 0.25,
  maxPoseDetections: 5,
  scoreThreshold: 0.5,
  nmsRadius: 20,
  detectionType: 'multiple',
  inputResolution: defaultMobileNetInputResolution, 
  multiplier: defaultMobileNetMultiplier,
  quantBytes: 2
}

function setup() {
  createCanvas(w, h);

  video = createCapture(VIDEO);
  video.size(width, height);
  // video.showControls(); 

  // Create a new poseNet method with a single detection
  poseNet = ml5.poseNet(video, config,  modelReady);

  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on('pose', function (results) {
    poses = results;
  });

  // Hide the video element, and just show the canvas
  video.hide();
  
  setupGui();

  stats = new Stats();
  stats.showPanel( 0 ); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild( stats.dom );
}

function modelReady() {
  select('#status').html('Model Loaded');
  isModelReady = true;
}


const tryResNetButtonName = 'tryResNetButton';
const tryResNetButtonText = '[New] Try ResNet50';

const guiState = {
  algorithm: 'multi-pose',
  input: {
    architecture: 'MobileNetV1',
    outputStride: defaultMobileNetStride,
    inputResolution: defaultMobileNetInputResolution,
    multiplier: defaultMobileNetMultiplier,
    quantBytes: defaultQuantBytes
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: config.maxPoseDetections,
    minPoseConfidence: config.minConfidence,
    minPartConfidence: config.scoreThreshold,
    nmsRadius: config.nmsRadius,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showBoundingBox: false,
  },
  estimatePoseEnable: true,
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui() {
  const gui = new dat.GUI({width: 300});

  let architectureController = null;
  guiState[tryResNetButtonName] = function() {
    architectureController.setValue('ResNet50')
  };
  gui.add(guiState, tryResNetButtonName).name(tryResNetButtonText);
  gui.add(guiState, 'estimatePoseEnable')
    .name('Estimate Pose')
    .onChange( (value) => {
      poseNet.estimatePoseEnable = value;
      if (value) {
        poseNet.load().then( () => console.log("Model Reloaded after Estimate Enable.") );
      }
    });

  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose works
  // for more than 1 person
  const algorithmController =
      gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']);

  // The input parameters have the most effect on accuracy and speed of the
  // network
  let input = gui.addFolder('Input');
  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  architectureController =
      input.add(guiState.input, 'architecture', ['MobileNetV1', 'ResNet50']);
  guiState.architecture = guiState.input.architecture;
  // Input resolution:  Internally, this parameter affects the height and width
  // of the layers in the neural network. The higher the value of the input
  // resolution the better the accuracy but slower the speed.
  let inputResolutionController = null;
  function updateGuiInputResolution(
      inputResolution,
      inputResolutionArray,
  ) {
    if (inputResolutionController) {
      inputResolutionController.remove();
    }
    guiState.inputResolution = inputResolution;
    guiState.input.inputResolution = inputResolution;
    inputResolutionController =
        input.add(guiState.input, 'inputResolution', inputResolutionArray);
    inputResolutionController.onChange(function(inputResolution) {
      guiState.changeToInputResolution = inputResolution;
    });
  }

  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  let outputStrideController = null;
  function updateGuiOutputStride(outputStride, outputStrideArray) {
    if (outputStrideController) {
      outputStrideController.remove();
    }
    guiState.outputStride = outputStride;
    guiState.input.outputStride = outputStride;
    outputStrideController =
        input.add(guiState.input, 'outputStride', outputStrideArray);
    outputStrideController.onChange(function(outputStride) {
      guiState.changeToOutputStride = outputStride;
    });
  }

  // Multiplier: this parameter affects the number of feature map channels in
  // the MobileNet. The higher the value, the higher the accuracy but slower the
  // speed, the lower the value the faster the speed but lower the accuracy.
  let multiplierController = null;
  function updateGuiMultiplier(multiplier, multiplierArray) {
    if (multiplierController) {
      multiplierController.remove();
    }
    guiState.multiplier = multiplier;
    guiState.input.multiplier = multiplier;
    multiplierController =
        input.add(guiState.input, 'multiplier', multiplierArray);
    multiplierController.onChange(function(multiplier) {
      guiState.changeToMultiplier = multiplier;
    });
  }

  // QuantBytes: this parameter affects weight quantization in the ResNet50
  // model. The available options are 1 byte, 2 bytes, and 4 bytes. The higher
  // the value, the larger the model size and thus the longer the loading time,
  // the lower the value, the shorter the loading time but lower the accuracy.
  let quantBytesController = null;
  function updateGuiQuantBytes(quantBytes, quantBytesArray) {
    if (quantBytesController) {
      quantBytesController.remove();
    }
    guiState.quantBytes = +quantBytes;
    guiState.input.quantBytes = +quantBytes;
    quantBytesController =
        input.add(guiState.input, 'quantBytes', quantBytesArray);
    quantBytesController.onChange(function(quantBytes) {
      guiState.changeToQuantBytes = +quantBytes;
    });
  }

  function updateGui() {
    if (guiState.input.architecture === 'MobileNetV1') {
      updateGuiInputResolution(
          defaultMobileNetInputResolution, [257, 353, 449, 513, 801]);
      updateGuiOutputStride(defaultMobileNetStride, [8, 16]);
      updateGuiMultiplier(defaultMobileNetMultiplier, [0.50, 0.75, 1.0])
    } else {  // guiState.input.architecture === "ResNet50"
      updateGuiInputResolution(
          defaultResNetInputResolution, [257, 353, 449, 513, 801]);
      updateGuiOutputStride(defaultResNetStride, [32, 16]);
      updateGuiMultiplier(defaultResNetMultiplier, [1.0]);
    }
    updateGuiQuantBytes(defaultQuantBytes, [1, 2, 4]);
  }

  updateGui();
  input.open();
  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  let single = gui.addFolder('Single Pose Detection');
  single.add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0)
        .onChange( (value) => poseNet.minConfidence = value );
  single.add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0)
        .onChange( (value) => poseNet.scoreThreshold = value );

  let multi = gui.addFolder('Multi Pose Detection');
  multi.add(guiState.multiPoseDetection, 'maxPoseDetections')
      .min(1)
      .max(20)
      .step(1)
      .onChange( (value) => poseNet.maxPoseDetections = value );
  multi.add(guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0)
      .onChange( (value) => poseNet.minConfidence = value );
  multi.add(guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0)
      .onChange( (value) => poseNet.scoreThreshold = value );
  // nms Radius: controls the minimum distance between poses that are returned
  // defaults to 20, which is probably fine for most use cases
  multi.add(guiState.multiPoseDetection, 'nmsRadius').min(0.0).max(40.0)
      .onChange( (value) => poseNet.nmsRadius = value );
  multi.open();

  let output = gui.addFolder('Output');
  output.add(guiState.output, 'showVideo');
  output.add(guiState.output, 'showSkeleton');
  output.add(guiState.output, 'showPoints');
  output.add(guiState.output, 'showBoundingBox');
  output.open();


  architectureController.onChange(function(architecture) {
    // if architecture is ResNet50, then show ResNet50 options
    updateGui();
    guiState.changeToArchitecture = architecture;
  });

  algorithmController.onChange(function(value) {
    switch (guiState.algorithm) {
      case 'single-pose':
        multi.close();
        single.open();
        poseNet.detectionType = 'single';
        break;
      case 'multi-pose':
        single.close();
        multi.open();
        poseNet.detectionType = 'multiple';
        break;
    }
  });
}

let captureMat, gray, blurred, thresholded;
let contours, hierarchy;
function cvSetup() {
  captureMat = new cv.Mat(h, w, cv.CV_8UC4);
  gray = new cv.Mat(h, w, cv.CV_8UC1);
  blurred = new cv.Mat(h, w, cv.CV_8UC1);
  thresholded = new cv.Mat(h, w, cv.CV_8UC1);
}

let ready = false;
function cvReady() {
  if (!cv || !cv.loaded) return false;
  if (ready) return true;
  cvSetup();
  ready = true;
  return true;
}


async function poseDetectionFrame() {
  if (guiState.changeToArchitecture) {
    poseNet.architecture = guiState.changeToArchitecture;
    poseNet.load().then( () => console.log("Model Reloaded after architecture changed.") );

    guiState.architecture = guiState.changeToArchitecture;
    guiState.changeToArchitecture = null;
  }

  if (guiState.changeToMultiplier) {
    poseNet.multiplier = +guiState.changeToMultiplier;
    poseNet.load().then( () => console.log("Model Reloaded after multiplier changed.") );
    
    guiState.multiplier = +guiState.changeToMultiplier;
    guiState.changeToMultiplier = null;
  }

  if (guiState.changeToOutputStride) {
    poseNet.outputStride = +guiState.changeToOutputStride;
    poseNet.load().then( () => console.log("Model Reloaded after outputStride changed.") );
    
    guiState.outputStride = +guiState.changeToOutputStride;
    guiState.changeToOutputStride = null;
  }

  if (guiState.changeToInputResolution) {
    poseNet.inputResolution = +guiState.changeToInputResolution;
    poseNet.load().then( () => console.log("Model Reloaded after inputResolution changed.") );
    
    guiState.inputResolution = +guiState.changeToInputResolution;
    guiState.changeToInputResolution = null;
  }

  if (guiState.changeToQuantBytes) {
    poseNet.quantBytes = guiState.changeToQuantBytes;
    poseNet.load().then( () => console.log("Model Reloaded after quantBytes changed.") );

    guiState.quantBytes = guiState.changeToQuantBytes;
    guiState.changeToQuantBytes = null;
  }

  switch (guiState.algorithm) {
    case 'single-pose':
      minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
      minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
      break;
    case 'multi-pose':
      minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
      minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
      break;
  }
}

function onResult(result) {
  poses = result;
}

function draw() {
  const showThresholded = true;
  
  stats.begin();
  /*
  if (cvReady()) {
    video.loadPixels();
    if (video.pixels.length > 0) {
      captureMat.data.set(video.pixels);

      const blurRadius = params.blurRadius;
      const threshold = params.threshold;

      cv.cvtColor(captureMat, gray, cv.COLOR_RGBA2GRAY, 0);
      cv.blur(gray, blurred, new cv.Size(blurRadius, blurRadius), new cv.Point(-1, -1), cv.BORDER_DEFAULT);
      cv.threshold(blurred, thresholded, threshold, 255, cv.THRESH_BINARY);

      if (showThresholded) {
        const src = thresholded.data;
        let  dst = video.pixels;
        const n = src.length;
        let j = 0;
        for (let i = 0; i < n; i++) {
          dst[j++] = src[i];
          dst[j++] = src[i];
          dst[j++] = src[i];
          dst[j++] = 255;
        }
        video.updatePixels();
      }

      if (contours) {
        contours.delete();
      }
      if (hierarchy) {
        hierarchy.delete();
      }
      contours = new cv.MatVector();
      hierarchy = new cv.Mat();
      cv.findContours(thresholded, contours, hierarchy, 3, 2, new cv.Point(0, 0));
    }
  }
  */


  if (guiState.output.showVideo) {
    image(video, 0, 0, width, height);
  } else {
    clear();
  }

  /*

  if (contours && !showThresholded) {
    
    noStroke();
    for (let i = 0; i < contours.size(); i++) {
      fill(0, 0, 255, 128);
      let contour = contours.get(i);

      beginShape();
      let k = 0;
      for (let j = 0; j < contour.total(); j++) {
        const x = contour.data[k++];
        const y = contour.data[k++];
        vertex(x, y);
      }
      endShape(CLOSE);

      noFill();
      stroke(255, 255, 255)
      const box = cv.boundingRect(contour);
      rect(box.x, box.y, box.width, box.height);

      // these aren't working right now:
      // https://github.com/ucisysarch/opencvjs/issues/30
      //            var minAreaRect = cv.minAreaRect(contour);
      //            var minAreaEllipse = cv.ellipse1(contour);
      //            var fitEllipse = cv.fitEllipse(contour);
    }

  }
  */
  

  if (guiState.estimatePoseEnable && isModelReady) {
    
    poseDetectionFrame().then( () => {
    
      // For each pose (i.e. person) detected in an image, loop through the poses
      // and draw the resulting skeleton and keypoints if over certain confidence
      // scores
      poses.forEach(element => {
        const pose = element.pose;

        if (pose.score >= minPoseConfidence) {
          if (guiState.output.showPoints) {
            drawKeypoints(pose.keypoints);
          }
          if (guiState.output.showSkeleton) {
            drawSkeleton(element.skeleton);
          }
          if (guiState.output.showBoundingBox) {
            drawBoundingBox(element.boundingBox)
          }
        }
        
      });
        
    });
  }
  
  stats.end();
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints(keypoints) {
  for (let j = 0; j < keypoints.length; j++) {
    // A keypoint is an object describing a body part (like rightArm or leftShoulder)
    let keypoint = keypoints[j];
    // Only draw an ellipse is the pose probability is bigger than minPartConfidence
    if (keypoint.score > minPartConfidence) {
      fill(255, 0, 0);
      noStroke();
      const r=15;
      ellipse(keypoint.position.x, keypoint.position.y, r, r);
    }
  }
}

// A function to draw the skeletons
function drawSkeleton(skeleton) {
  // For every skeleton, loop through all body connections
  for (let j = 0; j < skeleton.length; j++) {
    let partA = skeleton[j][0];
    let partB = skeleton[j][1];
    stroke(255, 0, 0);
    strokeWeight(4);
    line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
  }
}

/**
 * Draw the bounding box of a pose. For example, for a whole person standing
 * in an image, the bounding box will begin at the nose and extend to one of
 * ankles
 */
function drawBoundingBox(boundingBox) {
  noFill();
  strokeWeight(4);
  stroke(255, 0, 255);
  rect(
      boundingBox.minX, boundingBox.minY, boundingBox.maxX - boundingBox.minX,
      boundingBox.maxY - boundingBox.minY);

  // ctx.strokeStyle = boundingBoxColor;
  // ctx.stroke();
}
