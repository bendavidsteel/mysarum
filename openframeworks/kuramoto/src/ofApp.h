#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
#include "ofxGui.h"

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();
		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		
		struct Particle{
			glm::vec4 pos;
			glm::vec4 vel;
			glm::vec4 attr;
			ofFloatColor color;
		};

		ofShader compute;
		vector<Particle> particles;
		ofBufferObject particlesBuffer, particlesBuffer2;
		GLuint vaoID;
		ofEasyCam camera;
		ofVbo vbo;
		ofxPanel gui;
		ofParameter<float> fov;
		ofParameter<float> attractionCoeff, alignmentCoeff, repulsionCoeff;
		ofParameter<float> attractionMaxDist, alignmentMaxDist, repulsionMaxDist;
		ofParameter<float> maxSpeed;
		ofParameter<float> randomForce;
		ofParameter<float> kuramotoStrength, kuramotoMaxDist;
		ofParameterGroup shaderUniforms;
		ofParameter<float> fps;
		ofParameter<bool> dirAsColor;

		int numParticles;
};
