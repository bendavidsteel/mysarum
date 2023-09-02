#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	
	ofSetFrameRate(60);

	vidGrabber.setVerbose(true);
	int sourceWidth = ofGetWidth();
	int sourceHeight = ofGetHeight();
	vidGrabber.setDesiredFrameRate(30);
	vidGrabber.setup(sourceWidth, sourceHeight);

	maxBufferSize = 100;

	// ofxGraph Init
	rawTopGraph.setup("Raw Top Graph");
	rawTopGraph.setDx(1.0); // which means delta of time
	rawTopGraph.setColor(ofColor::white);  // ofColor(255,255,255)
	rawTopGraph.setMaxLengthOfData(maxBufferSize);
	rawTopGraph.setPosition(20, ofGetHeight() / 2 + 20);
	rawTopGraph.setSize(ofGetWidth() / 2 - 40, ofGetHeight() / 4 - 40);

	rawBottomGraph.setup("Raw Bottom Graph");
	rawBottomGraph.setDx(1.0);
	rawBottomGraph.setColor(ofColor::white);
	rawBottomGraph.setMaxLengthOfData(maxBufferSize);
	rawBottomGraph.setPosition(ofGetWidth() / 2 + 20, ofGetHeight() / 2 + 20);
	rawBottomGraph.setSize(ofGetWidth() / 2 - 40, ofGetHeight() / 4 - 40);

	topHeartBeatGraph.setup("Top Heartbeat Graph");
	topHeartBeatGraph.setDx(1.0);
	topHeartBeatGraph.setColor(ofColor::white);
	topHeartBeatGraph.setMaxLengthOfData(maxBufferSize);
	topHeartBeatGraph.setPosition(20, ofGetHeight() / 4 * 3 + 20);
	topHeartBeatGraph.setSize(ofGetWidth() / 2 - 40, ofGetHeight() / 4 - 40);

	bottomHeartBeatGraph.setup("Bottom Heartbeat Graph");
	bottomHeartBeatGraph.setDx(1.0);
	bottomHeartBeatGraph.setColor(ofColor::white);
	bottomHeartBeatGraph.setMaxLengthOfData(maxBufferSize);
	bottomHeartBeatGraph.setPosition(ofGetWidth() / 2 + 20, ofGetHeight() / 4 * 3 + 20);
	bottomHeartBeatGraph.setSize(ofGetWidth() / 2 - 40, ofGetHeight() / 4 - 40);
}

//--------------------------------------------------------------
void ofApp::update(){
	ofBackground(100,100,100);

	vidGrabber.update();
	bNewFrame = vidGrabber.isFrameNew();
	
	if (bNewFrame){

		ofPixels & pixels = vidGrabber.getPixels();
		
		// get mean brightness of each half
		float topMean = 0;
		float bottomMean = 0;

		int halfHeight = pixels.getHeight() / 2;
		for (int i = 0; i < pixels.getHeight(); i++) {
			for (int j = 0; j < pixels.getWidth(); j++) {
				ofColor c = pixels.getColor(j, i);
				if (i < halfHeight) {
					bottomMean += c.getLightness();
				} else {
					topMean += c.getLightness();
				}
			}
		}
		topMean /= pixels.getWidth() * halfHeight;
		bottomMean /= pixels.getWidth() * halfHeight;

		// add into buffers
		topMeanBuffer.push_back(topMean);
		bottomMeanBuffer.push_back(bottomMean);

		timeBuffer.push_back(ofGetElapsedTimef());

		if (topMeanBuffer.size() > maxBufferSize) {
			topMeanBuffer.erase(topMeanBuffer.begin());
			bottomMeanBuffer.erase(bottomMeanBuffer.begin());
			timeBuffer.erase(timeBuffer.begin());
		}

		float topBPM = 0;
		float bottomBPM = 0;
		float topHeartbeat = 0;
		float bottomHeartbeat = 0;

		getHeartBeat(topMeanBuffer, timeBuffer, topBPM, topHeartbeat, topPrunedFFT, topPrunedFreqs);
		getHeartBeat(bottomMeanBuffer, timeBuffer, bottomBPM, bottomHeartbeat, bottomPrunedFFT, bottomPrunedFreqs);

		topHeartbeatBuffer.push_back(topHeartbeat);
		bottomHeartbeatBuffer.push_back(bottomHeartbeat);

		if (topHeartbeatBuffer.size() > maxBufferSize) {
			topHeartbeatBuffer.erase(topHeartbeatBuffer.begin());
			bottomHeartbeatBuffer.erase(bottomHeartbeatBuffer.begin());
		}

		rawTopGraph.add(topMean);
		rawBottomGraph.add(bottomMean);

		topHeartBeatGraph.add(topHeartbeat);
		bottomHeartBeatGraph.add(bottomHeartbeat);
	}
}

//--------------------------------------------------------------
void ofApp::draw(){
	if (bNewFrame){

		ofPixels & pixels = vidGrabber.getPixels();
		
		// get mean brightness of each half
		ofPixels topPixels;
		topPixels.allocate(vidGrabber.getWidth(), vidGrabber.getHeight() / 2, OF_IMAGE_COLOR);
		ofPixels bottomPixels;
		bottomPixels.allocate(vidGrabber.getWidth(), vidGrabber.getHeight() / 2, OF_IMAGE_COLOR);

		int halfHeight = pixels.getHeight() / 2;
		for (int i = 0; i < pixels.getHeight(); i++) {
			for (int j = 0; j < pixels.getWidth(); j++) {
				ofColor c = pixels.getColor(j, i);
				if (i < halfHeight) {
					bottomPixels.setColor(j, i, c);
				} else {
					topPixels.setColor(j, i - halfHeight, c);
				}
			}
		}

		topTexture.loadData(topPixels);
		bottomTexture.loadData(bottomPixels);
	}
	topTexture.draw(20, 20, vidGrabber.getWidth() / 2 - 40, vidGrabber.getHeight() / 4 - 40);
	bottomTexture.draw(ofGetWidth() / 2 + 20, 20, vidGrabber.getWidth() / 2 - 40, vidGrabber.getHeight() / 4 - 40);

	ofDrawBitmapString("Top BPM: " + ofToString(topBPM), 20, 20);
	ofDrawBitmapString("Bottom BPM: " + ofToString(bottomBPM), ofGetWidth() / 2 + 20, 20);

	if (topPrunedFFT.size() > 0) {
		ofSetColor(ofColor::cyan);
		float top_bin_w = (ofGetWidth() / 2 - 40) / topPrunedFFT.size();
		float top_sum_fft = 0;
		for (int i = 0; i < topPrunedFFT.size(); i++){
			top_sum_fft += topPrunedFFT[i];
		}
		for (int i = 0; i < topPrunedFFT.size(); i++){
			float scaledValue = ofMap(topPrunedFFT[i], 0., top_sum_fft, 0., 1., true);//clamped value
			float bin_h = -1 * (scaledValue * ofGetHeight() / 2 - 20);
			ofDrawRectangle(i*top_bin_w + 20, ofGetHeight() / 2 - 20, top_bin_w, bin_h);
		}
	}

	if (bottomPrunedFFT.size() > 0) {
		ofSetColor(ofColor::cyan);
		float bottom_bin_w = (ofGetWidth() / 2 - 40) / bottomPrunedFFT.size();
		float bottom_sum_fft = 0;
		for (int i = 0; i < bottomPrunedFFT.size(); i++){
			bottom_sum_fft += bottomPrunedFFT[i];
		}
		for (int i = 0; i < bottomPrunedFFT.size(); i++){
			float scaledValue = ofMap(topPrunedFFT[i], 0., bottom_sum_fft, 0., 1., true);//clamped value
			float bin_h = -1 * (scaledValue * ofGetHeight() / 2 - 20);
			ofDrawRectangle(i*bottom_bin_w + 20 + ofGetWidth() / 2, ofGetHeight() / 2 - 20, bottom_bin_w, bin_h);
		}
	}

	rawTopGraph.draw();
	rawBottomGraph.draw();

	topHeartBeatGraph.draw();
	bottomHeartBeatGraph.draw();
}


void getHeartBeat(vector<float> buffer, vector<float> timeBuffer, float & bpm, float & heartbeat, vector<float> & pruned_fft, vector<float> & pruned_freqs) {
	if (buffer.size() > 10) {
		// get fps
		float fps = (float) (buffer.size() - 1) / (timeBuffer[timeBuffer.size() - 1] - timeBuffer[0]);

		// linspace
		vector<float> even_times;
		even_times.resize(buffer.size());
		for (int i = 0; i < buffer.size(); i++) {
			even_times[i] = timeBuffer[0] + (i / (buffer.size() - 1)) * (timeBuffer[timeBuffer.size() - 1] - timeBuffer[0]);
		}

		// do interpolation with dlib matrix
		vector<float> interpolated;
		interpolated.resize(buffer.size());
		for (int i = 0; i < buffer.size(); i++) {
			interpolated[i] = 0;
		}
		for (int i = 0; i < buffer.size(); i++) {
			float t = timeBuffer[0];
			float v = buffer[0];
			for (int j = 0; j < timeBuffer.size() - 1; j++) {
				if ((timeBuffer[j] <= even_times[i]) && (timeBuffer[j + 1] > even_times[i])) {
					t = (even_times[i] - timeBuffer[j]) / (timeBuffer[j + 1] - timeBuffer[j]);
					v = buffer[j] + t * (buffer[j + 1] - buffer[j]);
				}
			}
			interpolated[i] = v;
		}

		// hamming window
		vector<float> hamming;
		hamming.resize(buffer.size());
		for (int i = 0; i < buffer.size(); i++) {
			hamming[i] = 0.54 - 0.46 * cos(2 * M_PI * i / buffer.size());
		}
		for (int i = 0; i < buffer.size(); i++) {
			interpolated[i] = hamming[i] * interpolated[i];
		}

		// normalize
		float mean = 0;
		for (int i = 0; i < buffer.size(); i++) {
			mean += interpolated[i];
		}
		mean /= buffer.size();
		for (int i = 0; i < buffer.size(); i++) {
			interpolated[i] -= mean;
		}

		vector<std::complex<float>> interpolated_complex;
		interpolated_complex.resize(buffer.size());
		for (int i = 0; i < buffer.size(); i++) {
			interpolated_complex[i] = std::complex<float>(interpolated[i], 0);
		}

		// do FFT
		dlib::matrix<std::complex<float>> raw = dlib::fft(interpolated_complex);

		// get phase
		vector<float> phase;
		phase.resize(buffer.size());
		for (int i = 0; i < buffer.size(); i++) {
			phase[i] = std::arg(raw(i));
		}

		// get fft
		vector<float> fft;
		fft.resize(buffer.size());
		for (int i = 0; i < buffer.size(); i++) {
			fft[i] = std::abs(raw(i));
		}

		// get freqs
		vector<float> freqs;
		freqs.resize(buffer.size() / 2 + 1);
		for (int i = 0; i < buffer.size() / 2 + 1; i++) {
			freqs[i] = 60 * (float) fps / buffer.size() * i;
		}

		//get pruned
		pruned_fft.clear();
		pruned_freqs.clear();
		vector<float> pruned_phase;

		for (int i = 0; i < freqs.size(); i++) {
			if ((freqs[i] > 50) && (freqs[i] < 180)) {
				pruned_fft.push_back(fft[i]);
				pruned_phase.push_back(phase[i]);
				pruned_freqs.push_back(freqs[i]);
			}
		}

		if (pruned_fft.size() > 0) {
			// get max
			int max_idx = 0;
			float max_val = 0;
			for (int i = 0; i < pruned_fft.size(); i++) {
				if (pruned_fft[i] > max_val) {
					max_idx = i;
					max_val = pruned_fft[i];
				}
			}

			// get bpm
			bpm = pruned_freqs[max_idx];

			// get heartbeat val
			heartbeat = (std::sin(pruned_phase[max_idx]) + 1) / 2;
		}

		// self.fps = float(L) / (self.times[-1] - self.times[0])
		// even_times = np.linspace(self.times[0], self.times[-1], L)
		// interpolated = np.interp(even_times, self.times, processed)
		// interpolated = np.hamming(L) * interpolated
		// interpolated = interpolated - np.mean(interpolated)
		// raw = np.fft.rfft(interpolated)
		// phase = np.angle(raw)
		// self.fft = np.abs(raw)
		// self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)

		// freqs = 60. * self.freqs
		// idx = np.where((freqs > 50) & (freqs < 180))

		// pruned = self.fft[idx]
		// if len(pruned) == 0:
		// 	return

		// phase = phase[idx]

		// pfreq = freqs[idx]
		// self.freqs = pfreq
		// self.fft = pruned
		// idx2 = np.argmax(pruned)

		// t = (np.sin(phase[idx2]) + 1.) / 2.
		// t = 0.9 * t + 0.1
		// alpha = t
		// beta = 1 - t

		// self.bpm = self.freqs[idx2]
	}
}