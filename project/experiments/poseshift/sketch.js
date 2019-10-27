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
let w = 320;
let h = 240;
const trackStartTime = 0.001;


function setup() {
  canvas = createCanvas(w, h);
  canvas.parent('sketch-holder');

  video = createCapture(VIDEO);
  // video = createVideo(['assets/cup.mp4'], () => {
  //   video.noLoop();
  //   video.volume(0);
  //   video.showControls();
  //   video.play();
  //   // video.pause();
  // });
  video.parent( 'video-holder' );
  video.size(width, height);
  video.onended( (elm) => {
    video.play();
    ready = false;
    setupTrackingAlgorithm();
  });

  // Create a new poseNet method with a single detection
  poseNet = ml5.poseNet(video, modelReady);
  
  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on('pose', function (results) {
    poses = results;
  });
  // Hide the video element, and just show the canvas
  // video.hide();

  setupGui();
  stats = new Stats();
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);
}

let modelLoaded = false;
function modelReady() {
  select('#status').html('Model Loaded');
  modelLoaded = true;
}

function getBBox() {
  console.log(poses);
  if (poses.length > 0) {
    return {x:0, y:0, w:0, h:0};
  }
}

// TODO: 
//    retornar um bbox para cada keypoint de cada pessoa
//    juntamente com seu centroid.
//    quem receber essas informações, deve usar o score 
//    para decidir se vai fazer o tracking desse keypoint
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


let dst;
let hsvVec;
let roiHist;
let hsv;
let frame;
let trackWindow;
let mask;
let lowScalar;
let highScalar;
let low;
let right;
let hsvRoiVec;
let termCrit;
let trackBox;
function setupTrackingAlgorithm() {
  // console.log('[setupTrackingAlgorithm] video.time', video.time());
  
  video.loadPixels();
  // console.log(video.loadedmetadata, video.pixels.length, video.pixels);
  

  // take first frame of the video
  frame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  frame.data.set(video.pixels);

  if (params.source == 'webcam') {

    // constant used to increase bbox's area
    const di = 30;
    const df = 30;
    const cbb = getCentroidBBoxes(poses);
    const bb = cbb.bboxes[0];
    const _x = bb.x0 - di;
    const _y = bb.y0 - di;
    const _w = bb.x1 - bb.x0 + df;
    const _h = bb.y1 - bb.y0 + df;

    trackWindow = new cv.Rect(_x,_y, _w,_h);

  } else {

    // hardcode the initial location of window
    if (params.resolution == '320x240') {

      trackWindow = new cv.Rect(150, 60, 63, 125);
      
    } else {
      
      const _x = 640*200/320;
      const _y = 480*60/240;
      const _w = 640*63/320;
      const _h = 480*150/240;
      trackWindow = new cv.Rect(_x, _y, _w, _h);

    }

  }
  

  // set up the ROI for tracking
  roi = frame.roi(trackWindow);
  hsvRoi = new cv.Mat();
  cv.cvtColor(roi, hsvRoi, cv.COLOR_RGBA2RGB);
  cv.cvtColor(hsvRoi, hsvRoi, cv.COLOR_RGB2HSV);
  mask = new cv.Mat();
  lowScalar = new cv.Scalar(30, 30, 0);
  highScalar = new cv.Scalar(180, 180, 180);
  low = new cv.Mat(hsvRoi.rows, hsvRoi.cols, hsvRoi.type(), lowScalar);
  high = new cv.Mat(hsvRoi.rows, hsvRoi.cols, hsvRoi.type(), highScalar);
  cv.inRange(hsvRoi, low, high, mask);
  roiHist = new cv.Mat();
  hsvRoiVec = new cv.MatVector();
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


let params;
let gui;
function setupGui() {
  let Parameters = function () {
    this.estimatePoseEnable = true;
    this.algorithm = 'MeanShift';
    this.source = 'webcam';
    this.resolution = '320x240';
    this.drawKeypoints = true;
    this.drawSkeleton = true;
  };

  params = new Parameters();
  gui = new dat.GUI();
  gui.add(params, 'estimatePoseEnable')
    .name('Estimate Pose')
    .onChange( (value) => {
      poseNet.estimatePoseEnable = value;
      if (value) {
        poseNet.load().then( () => console.log("Model Reloaded after Estimate Enable.") );
        ready = false;
        setupTrackingAlgorithm();
      }
    });
  gui.add(params, 'algorithm', ['MeanShift', 'CamShift']).onChange( (value) => {
    video.stop();
    video.play();
    ready = false;
    setupTrackingAlgorithm(); 
  });
  gui.add(params, 'source', ['video', 'webcam']).onChange( (value) => {
    video.stop();
    video.play();
    ready = false;
    setupTrackingAlgorithm();
    if (value == 'video') {
      //TODO
    } else {
      //TODO
    }
  });
  gui.add(params, 'resolution', ['320x240', '640x480']).onChange( (value) => {
    if (value == '320x240') {
      w = 320;
      h = 240
    } else {
      w = 640;
      h = 480;
    }
    video.stop();
    video.play();
    ready = false;
    canvas.resize(w, h, true);
    video.size(w, h);
    ready = false;
    setupTrackingAlgorithm();

    {
      // resize canvas resets strokes and fill colours
      // https://github.com/processing/p5.js/issues/905
      colorMode(RGB, 255);
      fill(0);
      stroke(255);
    }
  });
  gui.add(params, 'drawKeypoints');
  gui.add(params, 'drawSkeleton');
}

let ready = false;
function cvReady() {

  const inputLoaded = (params.source == 'webcam') 
      ? modelLoaded && (poses.length > 0) 
      : video.time() < trackStartTime;
  
  if (!cv || !cv.loaded || !inputLoaded) return false;
  if (ready) return true;
  setupTrackingAlgorithm();
  ready = true;
  return true;
}

function draw() {
  stats.begin();

  clear();
  video.loadPixels();
  // image(video, 0, 0, width, height);

  if (video.pixels.length > 0) {
    if (cvReady() && true) {

      try {

        // if (!streaming) {
        //   // clean and stop.
        //   frame.delete(); dst.delete(); hsvVec.delete(); roiHist.delete(); hsv.delete();
        //   return;
        // }

        // start processing.
        frame.data.set(video.pixels);
        cv.cvtColor(frame, hsv, cv.COLOR_RGBA2RGB);
        cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
        cv.calcBackProject(hsvVec, [0], roiHist, dst, [0, 180], 1);

        if (params.algorithm == 'MeanShift') {
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
          // cv.line(frame, pts[0], pts[1], [255, 0, 0, 255], 3);
          // cv.line(frame, pts[1], pts[2], [255, 0, 0, 255], 3);
          // cv.line(frame, pts[2], pts[3], [255, 0, 0, 255], 3);
          // cv.line(frame, pts[3], pts[0], [255, 0, 0, 255], 3);

          noFill();
          stroke(0, 0, 255);
          strokeWeight(2);
          line(pts[0].x, pts[0].y, pts[1].x, pts[1].y);
          line(pts[1].x, pts[1].y, pts[2].x, pts[2].y);
          line(pts[2].x, pts[2].y, pts[3].x, pts[3].y);
          line(pts[3].x, pts[3].y, pts[0].x, pts[0].y);
        }
        
        if (false) {
          const _src = frame.data;
          let _dst = video.pixels;
          for (let i = 0; i < _src.length; i+=4) {
            _dst[i+0] = _src[i+0];
            _dst[i+1] = _src[i+1];
            _dst[i+2] = 0; //_src[i+2];
            _dst[i+3] = 255;
          }
        }


      } catch (err) {
        utils.printError(err);
      }

    }
  }
  video.updatePixels();
  
  // We can call both functions to draw all keypoints and the skeletons
  if (params.drawKeypoints) {
    drawKeypoints();
  }

  if (params.drawSkeleton) {
    drawSkeleton();
  }

  stats.end();
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  // Loop through all the poses detected
  for (let i = 0; i < poses.length; i++) {
    // For each pose detected, loop through all the keypoints
    let pose = poses[i].pose;
    for (let j = 0; j < pose.keypoints.length; j++) {
      // A keypoint is an object describing a body part (like rightArm or leftShoulder)
      let keypoint = pose.keypoints[j];
      // Only draw an ellipse is the pose probability is bigger than 0.2
      if (keypoint.score > 0.2) {
        fill(255, 0, 0);
        noStroke();
        ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
      }
    }
  }
}

// A function to draw the skeletons
function drawSkeleton() {
  // Loop through all the skeletons detected
  for (let i = 0; i < poses.length; i++) {
    let skeleton = poses[i].skeleton;
    // For every skeleton, loop through all body connections
    for (let j = 0; j < skeleton.length; j++) {
      let partA = skeleton[j][0];
      let partB = skeleton[j][1];
      stroke(255, 0, 0);
      line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    }
  }
}
