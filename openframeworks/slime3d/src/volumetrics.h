#pragma once

#include "ofFbo.h"
#include "ofShader.h"
#include "ofxVolumetrics.h"

class Volumetrics 
{
	public:

		Volumetrics();
		virtual ~Volumetrics();
		
		void setup(int w, int h, int d, glm::vec3 voxelSize, bool usePowerOfTwoTexSize = false);
		
		void destroy();
		
		void drawVolume(float x, float y, float z, float size, int zTexOffset);
		void drawVolume(float x, float y, float z, float w, float h, float d, int zTexOffset);
		
		bool isInitialized();	
		int getVolumeWidth();
		int getVolumeHeight();
		int getVolumeDepth();
		const ofFbo& getFbo() const;
		int getRenderWidth();
		int getRenderHeight();
		float getXyQuality();
		float getZQuality();
		float getThreshold();
		float getDensity();
		
		void setXyQuality(float q);
		void setZQuality(float q);
		void setThreshold(float t);
		void setDensity(float d);
		void setRenderSettings(float xyQuality, float zQuality, float dens, float thresh);
		void setVolumeTextureFilterMode(GLint filterMode);
		void setDrawDebugVolume(bool b);

	protected:

	private:

		void updateRenderDimentions();

		ofFbo fboRender;
		ofShader volumeShader;
		ofVboMesh volumeMesh; 	
		
		glm::vec3 voxelRatio;
		bool bIsInitialized;
		int volWidth, volHeight, volDepth;
		int volWidthPOT, volHeightPOT, volDepthPOT;
		bool bIsPowerOfTwo;
		glm::vec3 quality;
		float threshold;
		float density;
		int renderWidth, renderHeight;
		
		bool bDrawDebugVolume;
};
