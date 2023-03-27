#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	
	ofSetFrameRate(60);

	vidGrabber.setVerbose(true);
	int sourceWidth = ofGetWidth();
	int sourceHeight = ofGetHeight();
	vidGrabber.setup(sourceWidth, sourceHeight);
	
	blurAmount = 8;
	cvDownScale = 8;
	bContrastStretch = false;
	// store a minimum squared value to apply flow velocity
	minLengthSquared = 0.5 * 0.5;

	int scaledWidth = sourceWidth / cvDownScale;
	int scaledHeight = sourceHeight / cvDownScale;

	currentImage.clear();
	// allocate the ofxCvGrayscaleImage currentImage
	currentImage.allocate(scaledWidth, scaledHeight);
	currentImage.set(0);
	// free up the previous cv::Mat
	previousMat.release();
	// copy the contents of the currentImage to previousMat
	// this will also allocate the previousMat
	currentImage.getCvMat().copyTo(previousMat);
	// free up the flow cv::Mat
	flowMat.release();
	// notice that the argument order is height and then width
	// store as floats
	flowMat = cv::Mat(scaledHeight, scaledWidth, CV_32FC2);

	opticalFlowPixels.allocate(scaledWidth, scaledHeight, OF_IMAGE_COLOR);
	opticalFlowTexture.allocate(scaledWidth, scaledHeight, GL_RGBA8);
}

//--------------------------------------------------------------
void ofApp::update(){
	ofBackground(100,100,100);

	vidGrabber.update();
	bool bNewFrame = vidGrabber.isFrameNew();
	
	if (bNewFrame){

		colorImg.setFromPixels(vidGrabber.getPixels());
		grayImage = colorImg;
		
		// flip the image horizontally
		grayImage.mirror(false, true);
		
		// scale down the grayImage into the smaller sized currentImage
		currentImage.scaleIntoMe(grayImage);
		
		if(bContrastStretch) {
			currentImage.contrastStretch();
		}
		
		if(blurAmount > 0 ) {
			currentImage.blurGaussian(blurAmount);
		}
		
		// to perform the optical flow, we will be using cv::Mat
		// so grab the cv::Mat from the current image and store in currentMat
		cv::Mat currentMat = currentImage.getCvMat();
		// calculate the optical flow
		// we pass in the previous mat, the curent one and the flowMat where the opti flow data will be stored
		cv::calcOpticalFlowFarneback(previousMat,
									 currentMat,
									 flowMat,
									 0.5, // pyr_scale
									 3, // levels
									 15, // winsize
									 3, // iterations
									 7, // poly_n
									 1.5, // poly_sigma
									 cv::OPTFLOW_FARNEBACK_GAUSSIAN);
		
		// copy over the current mat into the previous mat
		// so that the optical flow function can calculate the difference
		currentMat.copyTo(previousMat);
	}
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofBackgroundGradient(ofColor(0), ofColor(40));
	
	int numCols = ofGetWidth() / cvDownScale;
	int numRows = ofGetHeight() / cvDownScale;
	
	ofPushMatrix();
	
	for( int x = 0; x < numCols; x++ ) {
		for( int y = 0; y < numRows; y++ ) {
			const cv::Point2f& fxy = flowMat.at<cv::Point2f>(y, x);
			glm::vec2 flowVector( fxy.x, fxy.y );
			if( glm::length2(flowVector) > minLengthSquared ) {
				ofFloatColor color( 0.5 + 0.5 * ofClamp(flowVector.x, -1, 1), 0.5 + 0.5 * ofClamp(flowVector.y, -1, 1), 0 );
				opticalFlowPixels.setColor(x, y, color);
			} else {
				opticalFlowPixels.setColor(x, y, ofFloatColor(0,0,0));
			}
		}
	}
	opticalFlowTexture.loadData(opticalFlowPixels);
	opticalFlowTexture.draw(0, 0, ofGetWidth(), ofGetHeight());
	ofPopMatrix();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	switch (key){
		case OF_KEY_UP:
			cvDownScale += 1.f;
			break;
		case OF_KEY_DOWN:
			cvDownScale -= 1.0f;
			if( cvDownScale < 2) {
				cvDownScale = 2;
			}
			break;
		case 'c':
			bContrastStretch = !bContrastStretch;
			break;
		case OF_KEY_RIGHT:
			blurAmount ++;
			if (blurAmount > 255) blurAmount = 255;
			break;
		case OF_KEY_LEFT:
			blurAmount --;
			if (blurAmount < 0) blurAmount = 0;
			break;
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}