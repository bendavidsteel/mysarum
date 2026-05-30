use nannou::prelude::*;

/// World box the whole scene lives in (matches the OF `width/height/depth = 640`).
pub const WORLD: f32 = 640.0;

/// Orbiting camera, ported from `ofApp::draw`:
///   timeOfDay = t * 0.1
///   camDist   = width * 1.1
///   pos = centre + (camDist*sin, depth*0.2*sin(t/3) - depth*0.2, camDist*cos)
///   lookAt(centre)
pub struct Camera {
    pub eye:    Vec3,
    pub centre: Vec3,
}

impl Camera {
    pub fn orbit(time: f32) -> Self {
        let centre = Vec3::splat(WORLD * 0.5);
        let time_of_day = time * 0.1;
        let cam_dist = WORLD * 1.1;
        let eye = Vec3::new(
            centre.x + cam_dist * time_of_day.sin(),
            centre.y - WORLD * 0.2 + WORLD * 0.2 * (time_of_day / 3.0).sin(),
            centre.z + cam_dist * time_of_day.cos(),
        );
        Self { eye, centre }
    }

    /// view * proj for the given aspect ratio.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.centre, Vec3::Y);
        let proj = Mat4::perspective_rh(
            60.0_f32.to_radians(),
            aspect.max(0.0001),
            0.1,
            WORLD * 10.0,
        );
        proj * view
    }
}
