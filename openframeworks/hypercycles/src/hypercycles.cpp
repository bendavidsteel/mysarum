#include "ColorWheelSchemes.h"

#include "hypercycles.h"

void Hypercycles::patch() {
	addModuleOutput("signal", gain);
}

pdsp::Patchable & Hypercycles::out_signal() {
	return out("signal");
}

//--------------------------------------------------------------
void Hypercycles::setup(int _width, int _height){
	ofDisableArbTex();
	shader.load("shader");

	float mapFactor = 1.0;

	width = _width;
	height = _height;

	int w = width * mapFactor;
	int h = height * mapFactor;

	numSpecies = 9;
	cells.resize(w * h);

	speciesFriends.resize(numSpecies+1);
	for (int i = 1; i < numSpecies+1; i++) {
		if (i == 1) {
			speciesFriends[i] = numSpecies;
		} else {
			speciesFriends[i] = i - 1;
		}
	}

	for (int i = 0; i < 5; i++) {
		float x = ofRandom(w);
		float y = ofRandom(h);
		Arpeggio arpeggio(x, y, numSpecies);
		arpeggios.push_back(arpeggio);
		arpeggios.back().out_signal() >> gain;
	}

	int colorScheme = 0;
	ofColor primaryColor = ofColor(255, 0, 0);
	scheme = ColorWheelSchemes::SCHEMES.at(colorScheme);
    scheme->setPrimaryColor(primaryColor);
    vector<ofColor> colors = scheme->interpolate(numSpecies);

	speciesColours.resize(numSpecies+1);
	speciesColours[0].set(0, 0, 0);
	for (int i = 1; i < numSpecies+1; i++) {
		speciesColours[i] = colors[i-1];
	}

	for (int i = 0; i < cells.size(); i++) {
		cells[i] = 0;
	}

	gui.setup();
	gui.add(birthRate.set("birthRate", 0.01, 0.0, 1.0));
	gui.add(deathRate.set("deathRate", 0.1, 0.0, 1.0));
	gui.add(moveRate.set("moveRate", 0.05, 0.0, 1.0));
	gui.add(replicateRate.set("replicateRate", 0.2, 0.0, 1.0));
	gui.add(catalyticSupport.set("catalyticSupport", 0.1, 0.0, 1.0));
	gui.add(synthAddRate.set("synthAddRate", 0.01, 0.0, 1.0));

	img.allocate(w, h, OF_IMAGE_COLOR);

	plane.set(width, height, w, h);
	plane.mapTexCoordsFromTexture(img.getTexture());
}

//--------------------------------------------------------------
void Hypercycles::update(){
	
	int w = img.getWidth();
	int h = img.getHeight();

	for (int j = 0; j < h; j++) {
		for (int i = 0; i < w; i++) {
			int index = j * w + i;
			if (cells[index] > 0) {
				// possibilty of death
				if (ofRandom(1) < deathRate) {
					cells[index] = 0;
					continue;
				}

				vector<int> vacantX;
				vector<int> vacantY;
				int neighboursFriends = 0;

				// check for vacant cells
				for (int l = -1; l <= 1; l++) {
					for (int k = -1; k <= 1; k++) {
						int x = i + l;
						int y = j + k;
						if (x >= 0 && x < w && y >= 0 && y < h) {
							int neighbourIndex = y * w + x;
							if (cells[neighbourIndex] == 0) {
								vacantX.push_back(x);
								vacantY.push_back(y);
							} else if (cells[neighbourIndex] == speciesFriends[cells[index]]) {
								neighboursFriends++;
							}
						}
					}
				}

				// chance to move to vacant cell
				if (vacantX.size() > 0 && ofRandom(1) < moveRate) {
					int r = floor(ofRandom(vacantX.size()));
					int newX = vacantX[r];
					int newY = vacantY[r];
					int newIndex = newY * w + newX;
					cells[newIndex] = cells[index];
					cells[index] = 0;
					continue;
				}

				// chance to replicate
				if (vacantX.size() > 0 && ofRandom(1) < (replicateRate + (catalyticSupport * neighboursFriends))) {
					int r = floor(ofRandom(vacantX.size()));
					int newX = vacantX[r];
					int newY = vacantY[r];
					int newIndex = newY * w + newX;
					cells[newIndex] = cells[index];
					continue;
				}
			} else {
				// spawn cells in certain positions
				float r = sqrt(pow(i - w/2, 2) + pow(j - h/2, 2));
				if (r < w / 10 && ofRandom(1) < birthRate) {
					cells[index] = floor(ofRandom(numSpecies+0.5));
				}
			}
		}
	}

	ofPixels & pixels = img.getPixels();
	for(int y=0; y<h; y++) {
		for(int x=0; x<w; x++) {
			int i = y * w + x;
			int species = cells[i];
			ofColor colour = speciesColours[species];
			pixels.setColor(x, y, colour);
		}
	}
	img.update();

	// if (ofRandom(1) < synthAddRate) {
	// 	float x = ofRandom(w);
	// 	float y = ofRandom(h);
	// 	Arpeggio arpeggio(x, y, numSpecies);
	// 	arpeggios.push_back(arpeggio);
	// 	arpeggios.back().out_signal() >> gain;
	// }
	for (int i = 0; i < arpeggios.size(); i++) {
		ofVec2f pos = arpeggios[i].pos;
		int species = cells[pos.y * w + pos.x];
		if (species > 0) {
			arpeggios[i].setSpecies(species, speciesColours[species]);
		}
	}
}

//--------------------------------------------------------------
void Hypercycles::draw(){

	// img.draw(0, 0, ofGetWidth(), ofGetHeight());
	// bind our texture. in our shader this will now be tex0 by default
	// so we can just go ahead and access it there.
	img.getTexture().bind();

	shader.begin();

	ofPushMatrix();

	// translate plane into center screen.
	float tx = width / 2;
	float ty = height / 2;
	ofTranslate(tx, 0, ty);

	float rotation = -90.;
	ofRotateDeg(rotation, 1, 0, 0);

	plane.draw();

	ofPopMatrix();

	shader.end();

	img.getTexture().unbind();

	for (int i = 0; i < arpeggios.size(); i++) {
		arpeggios[i].draw();
	}
}

void Hypercycles::drawGui(){
	gui.draw();
}
