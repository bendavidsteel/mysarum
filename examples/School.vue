<template>
    <canvas class="swimbox">
    </canvas>
</template>
<script>
import $ from 'jquery';

export default {
    data() {
        return {
            amount: 750,
			speed: 2.5,
			image: null,
			zIndex: null,
			resize: true,
            title: 'Schooling Fish',
            slug: 'School',
            image_path: '@/assets/visuals/school.png'
        };
    },
    mounted() {
        var self = this;
        const CANVAS = $(".swimbox").get(0);
        const CONTEXT = CANVAS.getContext('2d');
        var canvasHeight = CANVAS.offsetHeight;
        var canvasWidth = CANVAS.offsetWidth;
        var flakes = [];
        const turnSpeed = 1 / 32;
        const detectDist = 100;
        const numTrack = 3;

        CANVAS.height = canvasHeight;
        CANVAS.width = canvasWidth;
        CANVAS.style.zIndex = self.zIndex ? self.zIndex : 'auto';
        function init() {
            for (var i = 0; i < self.amount; i++) {
                let x = random(0, canvasWidth);
                let y = random(0, canvasHeight);
                let theta = random(0, Math.PI * 2);
                flakes.push({
                    x: x,
                    y: y,
                    theta: theta
                });
            }
            draw();
        }
        function draw() {
            CONTEXT.clearRect(0, 0, canvasWidth, canvasHeight);
            for (var i = 0; i < self.amount; i++) {
                var flake = flakes[i];

                let velX = self.speed * Math.sin(flake.theta);
                let velY = self.speed * Math.cos(flake.theta);
                let normBackX = flake.x - ((velX / self.speed) * 5);
                let normBackY = flake.y - ((velY / self.speed) * 5);

                CONTEXT.save();
                CONTEXT.beginPath();
                CONTEXT.strokeStyle = '#000';
                CONTEXT.lineWidth = 2;
                CONTEXT.moveTo(flake.x, flake.y);
                CONTEXT.lineTo(normBackX, normBackY);
                CONTEXT.stroke();
                //CONTEXT.closePath();
                CONTEXT.restore();

                flake.theta = normTheta(random(flake.theta - (Math.PI * turnSpeed), flake.theta + (Math.PI * turnSpeed)));
                
                let neighbourFlakes = rankNeighbours(flake, flakes, detectDist);
                // only checking closest
                for (let m = 0; m < Math.min(numTrack, neighbourFlakes.length); m++) {
                    let neighbourFlake = neighbourFlakes[m];
                    let avoidTheta = getAvoidTheta(flake, neighbourFlake);
                    flake.theta += avoidTheta / 1.2;

                    // turn in same direction as neighbours
                    let flakeThetaDiff = getThetaDiff(flake.theta, neighbourFlake.flakeTheta);
                    flake.theta += flakeThetaDiff / 3;
                }

                // turn in direction of centroid
                let centroidX = 0;
                let centroidY = 0;
                for (let m = 0; m < neighbourFlakes.length; m++) {
                    centroidX += neighbourFlakes[m].x / neighbourFlakes.length;
                    centroidY += neighbourFlakes[m].y / neighbourFlakes.length;
                }
                let centroidTheta = thetaBetweenPoints(centroidX - flake.x, centroidY - flake.y);
                let centroidThetaDiff = getThetaDiff(flake.theta, centroidTheta);
                flake.theta += centroidThetaDiff / 30;

                flake.theta = normTheta(flake.theta);
                velX = self.speed * Math.sin(flake.theta);
                velY = self.speed * Math.cos(flake.theta);
                flake.y += velY;
                flake.x += velX;
                checkReset(flake);
            }
            
            requestAnimationFrame(draw);
        }
        function checkReset(flake) {
            if (flake.x > canvasWidth) {
                flake.x = 0;
            }
            else if (flake.x < 0) {
                flake.x = canvasWidth;
            }
            if (flake.y > canvasHeight) {
                flake.y = 0;
            }
            else if (flake.y < 0) {
                flake.y = canvasHeight;
            }
        }
        init();
        if (self.resize) {
            window.addEventListener('resize', function() {
                var H0 = CANVAS.height,
                        W0 = CANVAS.width,
                        H1 = CANVAS.offsetHeight,
                        W1 = CANVAS.offsetWidth;
                CANVAS.height = canvasHeight = H1;
                CANVAS.width = canvasWidth = W1;
                for (var i = 0; i < self.amount; i++) {
                    var flake = flakes[i];
                    flake.x = flake.x / W0 * W1;
                    flake.y = flake.y / H0 * H1;
                }
            });
        }
    }
}
function random(min, max) {
    var delta = max - min;
    return max === min ? min : Math.random() * delta + min;
}
function normTheta(theta) {
    while (theta < 0) {
        theta += 2 * Math.PI;
    }
    while (theta >= (2 * Math.PI)) {
        theta -= 2 * Math.PI;
    }
    return theta;
}
function thetaBetweenPoints(x, y) {
    let theta = -Math.atan2(y, x) + (Math.PI / 2);
    let normedTheta = normTheta(theta);
    return normedTheta;
}
function objTheta(objA, objB) {
    return thetaBetweenPoints(objB.x - objA.x, objB.y - objA.y);
}
function objDist(objA, objB) {
    return Math.sqrt((objB.x - objA.x) ** 2 + (objB.y - objA.y) ** 2);
}
function rankNeighbours(flake, flakes, detectDist) {
    let neighbours = [];
    for (let i = 0; i < flakes.length; i++) {
        let flakeNeighbour = flakes[i];
        let dist = objDist(flake, flakeNeighbour);
        if (dist < detectDist) {
            neighbours.push({
                x: flakeNeighbour.x,
                y: flakeNeighbour.y,
                dist: dist,
                theta: objTheta(flake, flakeNeighbour),
                flakeTheta: flakeNeighbour.theta,
                idx: i
            });
        }
    }
    neighbours.sort((a, b) => a.dist - b.dist);
    return neighbours;
}
function getThetaDiff(thetaA, thetaB) {
    if (thetaB > thetaA) {
        if (thetaB < thetaA + Math.PI) {
            return thetaB - thetaA;
        }
        else if (thetaB > thetaA + Math.PI) {
            return -(thetaA + ((2 * Math.PI) - thetaB));
        }
        else {
            return Math.PI;
        }
    }
    else if (thetaB < thetaA) {
        if (thetaB > thetaA - Math.PI) {
            return thetaB - thetaA;
        }
        else if (thetaB < thetaA - Math.PI) {
            return thetaB + ((2 * Math.PI) - thetaA);
        }
        else {
            return Math.PI;
        }
    }
    else {
        return 0;
    }
}
function getAvoidTheta(flake, obj) {
    // if already collided just carry on
    if (obj.dist == 0) {
        return 0;
    }

    let thetaDiff = getThetaDiff(flake.theta, obj.theta);

    // check if in sight
    let absThetaDiff = Math.abs(thetaDiff);
    let avoidDir = 0;
    if (thetaDiff == 0) {
        avoidDir = Math.random() < 0.5 ? 1 : -1;
    }
    else if (absThetaDiff < (Math.PI / 2)) {
        // weight by angle
        // is 1 or 
        avoidDir = (-1 * thetaDiff) / absThetaDiff;
    }
    return avoidDir * (Math.PI / Math.min(obj.dist, 8)) * Math.cos(thetaDiff / 2);
}
</script>

<style scoped>
.swimbox {
    height: 100%;
    width: 100%;
}
</style>