(params) => {
          if (!params) {
            // Hide tooltip
            if (tooltipSimulation) {
              tooltipSimulation.pause();
            }
            return null;
          }
          
          const tooltipId = `tooltip-${Date.now()}`;

          let speciesCount = params.w_k.length;
          let kernelsCount = params.w_k[0][0].length;
          let growthFuncsCount = params.mu_g[0].length;
          
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
                <span><strong>Species:</strong> ${speciesCount || 'N/A'}</span>
                <span><strong>Kernels:</strong> ${kernelsCount || 'N/A'}</span>
                <span></span>
                </div>
                <div class="param-row">
                <span><strong>Growth:</strong> ${growthFuncsCount || 'N/A'}</span>
                <span></span>
                </div>
            </div>
            </div>`;
          }