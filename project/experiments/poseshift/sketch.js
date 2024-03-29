// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
PoseNet example using p5.js
=== */

let canvas;
let video;
let poseNet;
let stats;
let poses = [];

const videoSrc = { 
  'video 1': ['../../assets/u2_640x360.mp4'], 
  'video 2': ['../../assets/frevo_640x360.mp4'],
  'video 3': ['../../assets/pomplamoose_640x360.mp4'],
  'video 4': ['../../assets/dancing_640x360.mp4'],
  'video 5': ['../../assets/soccer_video_640x360.mp4'],
}


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
  minConfidence: 0.625,
  maxPoseDetections: 10,
  scoreThreshold: 0.65,
  nmsRadius: 20,
  detectionType: 'multiple',
  inputResolution: defaultMobileNetInputResolution, 
  multiplier: defaultMobileNetMultiplier,
  quantBytes: 2
}


function setup() {
  video = createVideo(videoSrc['video 1'], () => {
    video.loop();
    video.volume(0);
    video.pause();
    video.showControls();
    const w = video.width;
    const h = video.height;
    let canvas = createCanvas(w, h);
    canvas.parent('sketch-holder');
  });
  video.parent( 'video-holder' );

  setupPoseNet();
  setupGui();
  
  stats = new Stats();
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);
}


let dst = null;
let roiHist = null;
let hsv = null;
let frame = null;
let trackWindow = null;
let termCrit = null;
let trackBox = null;
const trackStartTime = 0.001;
function setupTrackingAlgorithm() {

  function getCentroidBBoxes(arr, useAllKP=false) {
    
    const centroids = [];
    const bboxes = [];
    arr.forEach(person => {
        const nose = person.pose.keypoints[0].position;
        const leftEye = person.pose.keypoints[1].position;
        const rightEye = person.pose.keypoints[2].position;
        const leftEar = person.pose.keypoints[3].position;
        const rightEar = person.pose.keypoints[4].position;
  
        const xs = [nose.x, leftEye.x, rightEye.x, leftEar.x, rightEar.x];
        const ys = [nose.y, leftEye.y, rightEye.y, leftEar.y, rightEar.y];
  
        // bounding box from nose, left eye and right eye
        let bb;
        if (useAllKP) {
            bb = {
                x0: person.boundingBox.minX,
                y0: person.boundingBox.minY,
                x1: person.boundingBox.maxX,
                y1: person.boundingBox.maxY
            };
  
        } else {
  
            bb = {
                x0: min(xs),
                y0: min(ys),
                x1: max(xs),
                y1: max(ys)
            };
        }
  
        // centroid
        const cx = (bb.x0 + bb.x1) / 2.0;
        const cy = (bb.y0 + bb.y1) / 2.0;
        const centroid = [cx, cy];
  
        centroids.push(centroid);
        bboxes.push(bb);
    });
    return {centroids: centroids, bboxes: bboxes};
  }

  ready = false;
  
  if (dst != null) dst.delete();
  if (roiHist !== null) roiHist.delete();
  if (hsv != null) hsv.delete();
  if (frame != null) frame.delete();
  if (trackWindow != null) trackWindow = null;
  if (termCrit != null) null;
  if (trackBox != null) null;

  console.log('[setupTrackingAlgorithm] video.time', video.time());
  
  video.loadPixels();
  // console.log(video.loadedmetadata, video.pixels.length, video.pixels);
  

  // take first frame of the video
  frame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  frame.data.set(video.pixels);
  
  {
    // NOTE: Using bbox from the first person detected
    const cbb = getCentroidBBoxes(poses, guiState.tracking.useAllKeypoints);
    const bb = cbb.bboxes[0];
    const _x = bb.x0;
    const _y = bb.y0;
    const _w = Math.min(bb.x1 - bb.x0, width - bb.x0);
    const _h = Math.min(bb.y1 - bb.y0, height - bb.y0);

    trackWindow = new cv.Rect(_x,_y, _w,_h);
    
  }

  // set up the ROI for tracking
  const roi = frame.roi(trackWindow);
  const hsvRoi = new cv.Mat();
  cv.cvtColor(roi, hsvRoi, cv.COLOR_RGBA2RGB);
  cv.cvtColor(hsvRoi, hsvRoi, cv.COLOR_RGB2HSV);
  const mask = new cv.Mat();
  const lowScalar = new cv.Scalar(30, 30, 0);
  const highScalar = new cv.Scalar(180, 180, 180);
  const low = new cv.Mat(hsvRoi.rows, hsvRoi.cols, hsvRoi.type(), lowScalar);
  const high = new cv.Mat(hsvRoi.rows, hsvRoi.cols, hsvRoi.type(), highScalar);
  cv.inRange(hsvRoi, low, high, mask);
  roiHist = new cv.Mat();
  const hsvRoiVec = new cv.MatVector();
  hsvRoiVec.push_back(hsvRoi);
  cv.calcHist(hsvRoiVec, [0], mask, roiHist, [180], [0, 180]);
  cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX);

  // delete useless mats.
  roi.delete(); hsvRoi.delete(); mask.delete(); low.delete(); high.delete(); hsvRoiVec.delete();

  // Setup the termination criteria, either 10 iteration or move by atleast 1 pt
  termCrit = new cv.TermCriteria(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1);

  hsv = new cv.Mat(video.height, video.width, cv.CV_8UC3);
  dst = new cv.Mat();
  hsvVec = new cv.MatVector();
  hsvVec.push_back(hsv);
  trackBox = null;

}

let ready = false;
function cvReady() {

  const inputLoaded = (guiState.source == 'webcam') 
      ? isModelReady && (poses.length > 0) 
      : (video.time() > trackStartTime) && isModelReady && (poses.length > 0);

  
  if (!cv || !cv.loaded || !inputLoaded) return false;
  if (ready) return true;
  setupTrackingAlgorithm();
  ready = true;
  return true;
}


function setupPoseNet() {
  isModelReady = false;  
  
  delete poses;
  poses = null;
  poses = [];

  if (poseNet != null) {
    poseNet.net.dispose();
    delete poseNet;
    poseNet = null;
  }
  // Create a new poseNet method with a single detection
  poseNet = ml5.poseNet(video, config,  modelReady);

  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on('pose', function (results) {
    delete poses;
    poses = [];
    results.forEach( (person, id) => {
      // console.log('score & confidence', person.pose.score, minPoseConfidence);
      person.id = id;
      
      if (person.pose.score >= minPoseConfidence) {
        poses.push(person);
      }
    });
    
  });
}


function modelReady() {
  select('#status').html('Model Loaded');
  select('#status').class('hidden');

  isModelReady = true;
  video.play();
  console.log("model loaded");

}


const tryResNetButtonName = 'tryResNetButton';
const tryResNetButtonText = '[New] Try ResNet50';

const guiState = {
  algorithm: 'multi-pose',
  source: 'video 1', // video n or webcam
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
    videoOpacity: 1.0,
    showVideo: true,
    showSkeleton: false,
    showPoints: false,
    showBoundingBox: false,
  },
  estimatePoseEnable: true,
  trackingEnable: true,
  tracking: {
    algorithm: 'Meanshift',
    useAllKeypoints: true,
  },
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
  // gui.add(guiState, tryResNetButtonName).name(tryResNetButtonText);
  gui.add(guiState, 'estimatePoseEnable')
    .name('Estimate Pose')
    .onChange( (value) => {
      poseNet.estimatePoseEnable = value;
      if (value) {
        poseNet.load().then( () => console.log("Model Reloaded after Estimate Enable.") );
      }
    });  

  gui.add(guiState, 'source', [...Object.keys(videoSrc), 'webcam']).name('Source')
    .onChange( (value) => {
      video.stop();
      video.remove();
      video = null; 
      clear();

      if (value !== 'webcam') {
        
        video = createVideo(videoSrc[value], () => {
          video.loop();
          video.volume(0);
          video.showControls();
          const w = video.width;
          const h = video.height;
          resizeCanvas(w, h);
          
        });

      } else {

        video = createCapture(VIDEO, () => {
          const w = video.width;
          const h = video.height;
          console.log('Why WxH =', w,h);
          
          // NOTE: hardcode canvas size for my camera
          // I don't know why video withxheight == 300x150;
          resizeCanvas(640, 480);

        });
        
      }
      video.parent( 'video-holder' );
      
      setupPoseNet();

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
  output.add(guiState.output, 'videoOpacity', 0.0, 1.0);
  // output.add(guiState.output, 'showVideo');
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

  let tracking = gui.addFolder('Tracking');
  tracking.add(guiState.tracking, 'algorithm', ['Meanshift', 'Camshift'])
    .onChange( (value) => {
      ready = false;
    });
    tracking.add(guiState.tracking, 'useAllKeypoints')
      .onChange( (value) => {
        ready = false;
      });
  tracking.open();

  {
    // resize canvas resets strokes and fill colours
    // https://github.com/processing/p5.js/issues/905
    colorMode(RGB, 255);
    fill(0);
    stroke(255);
  }

}

async function poseDetectionFrame() {
  if (guiState.changeToArchitecture) {
    poseNet.architecture = guiState.changeToArchitecture;
    poseNet.load().then( () => console.log("Model Reloaded after architecture changed.") );

    guiState.architecture = guiState.changeToArchitecture;
    guiState.changeToArchitecture = null;

    poses = [];
  }

  if (guiState.changeToMultiplier) {
    poseNet.multiplier = +guiState.changeToMultiplier;
    poseNet.load().then( () => console.log("Model Reloaded after multiplier changed.") );
    
    guiState.multiplier = +guiState.changeToMultiplier;
    guiState.changeToMultiplier = null;

    poses = [];
  }

  if (guiState.changeToOutputStride) {
    poseNet.outputStride = +guiState.changeToOutputStride;
    poseNet.load().then( () => console.log("Model Reloaded after outputStride changed.") );
    
    guiState.outputStride = +guiState.changeToOutputStride;
    guiState.changeToOutputStride = null;

    poses = [];
  }

  if (guiState.changeToInputResolution) {
    poseNet.inputResolution = +guiState.changeToInputResolution;
    poseNet.load().then( () => console.log("Model Reloaded after inputResolution changed.") );
    
    guiState.inputResolution = +guiState.changeToInputResolution;
    guiState.changeToInputResolution = null;

    poses = [];
  }

  if (guiState.changeToQuantBytes) {
    poseNet.quantBytes = guiState.changeToQuantBytes;
    poseNet.load().then( () => console.log("Model Reloaded after quantBytes changed.") );

    guiState.quantBytes = guiState.changeToQuantBytes;
    guiState.changeToQuantBytes = null;

    poses = [];
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


function draw() {
  stats.begin();

  clear();
  video.style('opacity', guiState.output.videoOpacity);


  video.loadPixels();
  if (video.pixels.length > 0) {
    if (cvReady()) {

      try {

        // start processing.
        frame.data.set(video.pixels);
        cv.cvtColor(frame, hsv, cv.COLOR_RGBA2RGB);
        cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
        cv.calcBackProject(hsvVec, [0], roiHist, dst, [0, 180], 1);

        if (guiState.tracking.algorithm == 'Meanshift') {
          // Apply meanshift to get the new location
          // and it also returns number of iterations meanShift took to converge,
          // which is useless in this demo.
          [, trackWindow] = cv.meanShift(dst, trackWindow, termCrit);
  
          // Draw it on image
          let [x, y, w, h] = [trackWindow.x, trackWindow.y, trackWindow.width, trackWindow.height];
          // cv.rectangle(frame, new cv.Point(x, y), new cv.Point(x + w, y + h), [255, 0, 0, 255], 2);
  
          noFill();
          stroke(0, 255, 0);
          strokeWeight(2);
          rect(x,y,w,h);
          
        } else {
          // apply camshift to get the new location
          [trackBox, trackWindow] = cv.CamShift(dst, trackWindow, termCrit);

          // Draw it on image
          let pts = cv.rotatedRectPoints(trackBox);

          noFill();
          stroke(0, 0, 255);
          strokeWeight(2);
          line(pts[0].x, pts[0].y, pts[1].x, pts[1].y);
          line(pts[1].x, pts[1].y, pts[2].x, pts[2].y);
          line(pts[2].x, pts[2].y, pts[3].x, pts[3].y);
          line(pts[3].x, pts[3].y, pts[0].x, pts[0].y);
        }
        
      } catch (err) {
        console.error(err);
        
      }

    }
  }
  video.updatePixels();



  
  if (guiState.estimatePoseEnable && isModelReady) {
    
    poseDetectionFrame().then( () => {
    
      // For each person´s pose detected in an image, loop through the poses
      // and draw the resulting skeleton and keypoints if over certain confidence
      // scores
      poses.forEach( (person, index) => {
        const pose = person.pose;

        if (pose.score >= minPoseConfidence) {
          if (guiState.output.showPoints) {
            drawKeypoints(pose.keypoints);
          }
          if (guiState.output.showSkeleton) {
            drawSkeleton(person.skeleton);
          }
          if (guiState.output.showBoundingBox) {
            drawBoundingBox(person.boundingBox, person.id, index)
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
      const r = guiState.source == 'webcam' ? 10 : 5;
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
    strokeWeight(guiState.source == 'webcam' ? 3 : 2);
    line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
  }
}

/**
 * Draw the bounding box of a pose. For example, for a whole person standing
 * in an image, the bounding box will begin at the nose and extend to one of
 * ankles
 */
function drawBoundingBox(boundingBox, pid, index) {
  fill(0,255,255);

  textSize(20);
  text(`Person ${pid}`, boundingBox.minX, boundingBox.minY);
  // text(` [ ${pid} ]`, boundingBox.minX-15, boundingBox.minY);
  
  noFill();
  strokeWeight(guiState.source == 'webcam' ? 3 : 2);
  stroke(255, 0, 255);
  rect(
      boundingBox.minX, boundingBox.minY, boundingBox.maxX - boundingBox.minX,
      boundingBox.maxY - boundingBox.minY);
}
