(params) => {
          if (!params) {
            // Hide tooltip
            if (tooltipSimulation) {
              tooltipSimulation.pause();
            }
            return null;
          }
          
          const tooltipId = `tooltip-${Date.now()}`;
          
          // Update simulation parameters after canvas is in DOM
          const waitForCanvas = () => {
            const canvas = document.getElementById(`tooltip-canvas-${tooltipId}`);
            if (canvas && canvas.offsetParent !== null) {
              updateTooltipSimulation(params, tooltipId);
            } else {
              requestAnimationFrame(waitForCanvas);
            }
          };
          requestAnimationFrame(waitForCanvas);
          
          return `
            <div class="lenia-tooltip" id="${tooltipId}">
            <div class="simulation-container">
                <canvas 
                id="tooltip-canvas-${tooltipId}" 
                width="240" 
                height="240"
                style="border: 1px solid #444; border-radius: 4px; background: #000;"
                ></canvas>
                <div class="loading-overlay" id="loading-${tooltipId}">
                <div>Loading simulation...</div>
                </div>
            </div>
            <div class="tooltip-params">
                <div class="param-row">
                <span><strong>Species:</strong> ${params.speciesCount || 'N/A'}</span>
                <span><strong>Particles:</strong> ${params.particleCount || 'N/A'}</span>
                </div>
                <div class="param-row">
                <span><strong>Kernels:</strong> ${params.kernelsCount || 'N/A'}</span>
                <span><strong>Growth:</strong> ${params.growthFuncsCount || 'N/A'}</span>
                </div>
                <div class="param-row">
                <span><strong>Map Size:</strong> ${params.mapSize || 'N/A'}</span>
                <span><strong>Max Force:</strong> ${params.maxForce?.toFixed(1) || 'N/A'}</span>
                </div>
            </div>
            </div>`;
          }