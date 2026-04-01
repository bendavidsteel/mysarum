/// Freeverb-style stereo reverb (8 parallel comb filters + 4 series allpass per channel).
///
/// Based on Jezar's Freeverb with standard tunings.

const NUM_COMBS: usize = 8;
const NUM_ALLPASSES: usize = 4;

// Comb filter delay lengths (in samples at 44100 Hz), Jezar tunings
const COMB_LENGTHS_L: [usize; NUM_COMBS] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
const ALLPASS_LENGTHS_L: [usize; NUM_ALLPASSES] = [556, 441, 341, 225];

// Stereo spread: offset right channel delays by this many samples
const STEREO_SPREAD: usize = 23;

const FIXED_GAIN: f32 = 0.015;
const ALLPASS_FEEDBACK: f32 = 0.5;

struct CombFilter {
    buffer: Vec<f32>,
    index: usize,
    filterstore: f32,
}

impl CombFilter {
    fn new(size: usize) -> Self {
        Self { buffer: vec![0.0; size], index: 0, filterstore: 0.0 }
    }

    #[inline]
    fn process(&mut self, input: f32, feedback: f32, damp1: f32, damp2: f32) -> f32 {
        let output = self.buffer[self.index];
        self.filterstore = output * damp2 + self.filterstore * damp1;
        self.buffer[self.index] = input + self.filterstore * feedback;
        self.index += 1;
        if self.index >= self.buffer.len() { self.index = 0; }
        output
    }
}

struct AllpassFilter {
    buffer: Vec<f32>,
    index: usize,
}

impl AllpassFilter {
    fn new(size: usize) -> Self {
        Self { buffer: vec![0.0; size], index: 0 }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let bufout = self.buffer[self.index];
        let output = bufout - input;
        self.buffer[self.index] = input + bufout * ALLPASS_FEEDBACK;
        self.index += 1;
        if self.index >= self.buffer.len() { self.index = 0; }
        output
    }
}

pub struct Freeverb {
    combs_l: Vec<CombFilter>,
    combs_r: Vec<CombFilter>,
    allpasses_l: Vec<AllpassFilter>,
    allpasses_r: Vec<AllpassFilter>,
    room_size: f32,
    damp: f32,
    wet: f32,
    dry: f32,
    // Derived
    feedback: f32,
    damp1: f32,
    damp2: f32,
}

impl Freeverb {
    pub fn new(sample_rate: u32) -> Self {
        let scale = sample_rate as f32 / 44100.0;
        let scale_len = |n: usize| ((n as f32 * scale) as usize).max(1);

        let combs_l = COMB_LENGTHS_L.iter().map(|&n| CombFilter::new(scale_len(n))).collect();
        let combs_r = COMB_LENGTHS_L.iter()
            .map(|&n| CombFilter::new(scale_len(n + STEREO_SPREAD)))
            .collect();
        let allpasses_l = ALLPASS_LENGTHS_L.iter().map(|&n| AllpassFilter::new(scale_len(n))).collect();
        let allpasses_r = ALLPASS_LENGTHS_L.iter()
            .map(|&n| AllpassFilter::new(scale_len(n + STEREO_SPREAD)))
            .collect();

        let mut rv = Self {
            combs_l, combs_r, allpasses_l, allpasses_r,
            room_size: 0.85,
            damp: 0.5,
            wet: 0.3,
            dry: 1.0,
            feedback: 0.0,
            damp1: 0.0,
            damp2: 0.0,
        };
        rv.update_derived();
        rv
    }

    fn update_derived(&mut self) {
        self.feedback = self.room_size;
        self.damp1 = self.damp;
        self.damp2 = 1.0 - self.damp;
    }

    pub fn set_room_size(&mut self, v: f32) { self.room_size = v; self.update_derived(); }
    pub fn set_damp(&mut self, v: f32) { self.damp = v; self.update_derived(); }
    pub fn set_wet(&mut self, v: f32) { self.wet = v; }
    pub fn set_dry(&mut self, v: f32) { self.dry = v; }

    pub fn room_size(&self) -> f32 { self.room_size }
    pub fn damp(&self) -> f32 { self.damp }
    pub fn wet(&self) -> f32 { self.wet }

    /// Process one stereo sample pair in-place.
    #[inline]
    pub fn process(&mut self, left: f32, right: f32) -> (f32, f32) {
        let input = (left + right) * FIXED_GAIN;

        let mut out_l = 0.0f32;
        let mut out_r = 0.0f32;

        for comb in &mut self.combs_l {
            out_l += comb.process(input, self.feedback, self.damp1, self.damp2);
        }
        for comb in &mut self.combs_r {
            out_r += comb.process(input, self.feedback, self.damp1, self.damp2);
        }

        for ap in &mut self.allpasses_l {
            out_l = ap.process(out_l);
        }
        for ap in &mut self.allpasses_r {
            out_r = ap.process(out_r);
        }

        let final_l = left * self.dry + out_l * self.wet;
        let final_r = right * self.dry + out_r * self.wet;
        (final_l, final_r)
    }
}
