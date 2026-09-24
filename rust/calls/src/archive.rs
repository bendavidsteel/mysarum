//! Reads a MAP-Elites `archive.npz` written by `callsong.archive.Archive.save`.

use std::io::Read;
use std::path::{Path, PathBuf};

pub struct Archive {
    pub path: PathBuf,
    pub res: usize,
    /// Row-major (res × res); non-finite means the cell is empty.
    pub fitness: Vec<f64>,
    /// Row-major (res × res × n_params), normalised [0, 1] genomes.
    pub genomes: Vec<f32>,
    pub n_params: usize,
}

impl Archive {
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut zip = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
        let (fshape, fitness) = read_npy(&mut zip, "fitness.npy")?;
        let (gshape, genomes) = read_npy(&mut zip, "genomes.npy")?;
        if fshape.len() != 2 || gshape.len() != 3 || fshape[0] != fshape[1] || gshape[..2] != fshape[..] {
            return Err(format!("unexpected shapes {fshape:?} / {gshape:?}"));
        }
        Ok(Archive {
            path: path.to_path_buf(),
            res: fshape[0],
            fitness,
            genomes: genomes.into_iter().map(|v| v as f32).collect(),
            n_params: gshape[2],
        })
    }

    pub fn genome(&self, i: usize, j: usize) -> &[f32] {
        let k = (i * self.res + j) * self.n_params;
        &self.genomes[k..k + self.n_params]
    }

    pub fn fitness(&self, i: usize, j: usize) -> f64 {
        self.fitness[i * self.res + j]
    }

    pub fn fitness_range(&self) -> (f64, f64) {
        self.fitness
            .iter()
            .filter(|v| v.is_finite())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
    }
}

fn read_npy(zip: &mut zip::ZipArchive<std::fs::File>, name: &str) -> Result<(Vec<usize>, Vec<f64>), String> {
    let mut bytes = Vec::new();
    zip.by_name(name)
        .map_err(|e| format!("{name}: {e}"))?
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{name}: not a .npy"));
    }
    let (hlen, start) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12)
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen]).map_err(|e| e.to_string())?;
    if header.contains("'fortran_order': True") {
        return Err(format!("{name}: fortran order unsupported"));
    }
    let descr = header
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .ok_or(format!("{name}: no descr"))?;
    let shape: Vec<usize> = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or(format!("{name}: no shape"))?
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let data = &bytes[start + hlen..];
    let values: Vec<f64> = match descr {
        "<f8" => data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
        "<f4" => data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        other => return Err(format!("{name}: dtype {other} unsupported")),
    };
    Ok((shape, values))
}

/// `archive.npz` files under the Python outputs directory, newest first.
pub fn discover(outputs: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(outputs)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path().join("archive.npz"))
        .filter(|p| p.is_file())
        .collect();
    found.sort();
    found.reverse();
    found
}
