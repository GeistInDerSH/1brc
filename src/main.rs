use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{stdout, Read, Write};
use std::mem::size_of;
use std::ops::BitXor;
use std::os::fd::{AsRawFd, FromRawFd};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::available_parallelism;
use std::{env, io, slice, thread};

const INPUT_FILE_NAME: &str = "measurements.txt";
/// The size of the Mmap segment to read
const SEGMENT_SIZE: usize = 1 << 21;
/// 64-bit hash constant from FxHash
const FX_HASH_CONST: usize = 0x517cc1b727220a95;
/// There are up to 10 000 unique stations.
/// Don't bother with the closest power of 2 (16 384), because we'd way over shoot
const MAP_CAPACITY: usize = 10_000;
/// Cover the size of the format string to allow for us to have an approximately
/// correct write buffer size
const ESTIMATED_PRINT_SIZE: usize = 20 // station name
        + 1 // equals
        + (3 * 5) // -?\d?\d.\d for the min/max/ave
        + 2 // slashes between the numbers
        + 2 // comma and space
;

type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// Pulled out of the rustc_hash crate to avoid having it as a dependency
#[derive(Default, Clone)]
struct FxHasher {
    hash: usize,
}

impl FxHasher {
    #[inline]
    fn add_to_hash(&mut self, i: usize) {
        self.hash = self
            .hash
            .rotate_left(5)
            .bitxor(i)
            .wrapping_mul(FX_HASH_CONST);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash as u64
    }

    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        const _: () = assert!(size_of::<usize>() <= size_of::<u64>());
        const _: () = assert!(size_of::<u32>() <= size_of::<usize>());
        let mut state = self.clone();
        while let Some(&usize_bytes) = take_first_chunk(&mut bytes) {
            state.add_to_hash(usize::from_ne_bytes(usize_bytes));
        }
        if let Some(&u32_bytes) = take_first_chunk(&mut bytes) {
            state.add_to_hash(u32::from_ne_bytes(u32_bytes) as usize);
        }
        if let Some(&u16_bytes) = take_first_chunk(&mut bytes) {
            state.add_to_hash(u16::from_ne_bytes(u16_bytes) as usize);
        }
        if let Some(&[u8_byte]) = take_first_chunk(&mut bytes) {
            state.add_to_hash(u8_byte as usize);
        }
        *self = state;
    }
}

#[inline]
fn take_first_chunk<'a, const N: usize>(slice: &mut &'a [u8]) -> Option<&'a [u8; N]> {
    let (first, rest) = slice.split_first_chunk()?;
    *slice = rest;
    Some(first)
}

struct Mmap {
    addr: *mut c_void,
    size: usize,
}

impl Mmap {
    #[inline]
    fn from_file_name(name: &str) -> Result<Self, io::Error> {
        let file = File::open(name)?;
        Mmap::from_file(file)
    }

    #[inline]
    fn from_file(file: File) -> Result<Self, io::Error> {
        let size = file.metadata()?.len() as usize;

        let addr = unsafe {
            let fd = file.as_raw_fd();
            libc::mmap(
                std::ptr::null_mut(),
                size as libc::size_t,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                fd,
                0,
            )
        };

        if addr == libc::MAP_FAILED {
            Err(io::Error::last_os_error())
        } else {
            Ok(Self { addr, size })
        }
    }

    #[inline]
    fn as_slice(&self) -> &'static [u8] {
        unsafe { slice::from_raw_parts(self.addr.cast(), self.size) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.addr, self.size) };
    }
}

// i64 is definitely overkill for the min and max, but i16/i32
// doesn't seem to provide much of a benefit in terms of speed. Likely because the
// CPU can load it all into memory anyway ¯\_(ツ)_/¯
#[derive(Clone)]
struct Data<'a> {
    name: &'a [u8],
    min: i64,
    max: i64,
    sum: i64,
    count: i64,
}

impl<'a> Data<'a> {
    #[inline]
    fn new(name: &'a [u8], val: i64) -> Self {
        Data {
            name,
            min: val,
            max: val,
            sum: val,
            count: 1,
        }
    }

    #[inline]
    fn mean(&self) -> f64 {
        (self.sum as f64) / 10.0 / (self.count as f64)
    }

    #[inline]
    fn min(&self) -> f64 {
        self.min as f64 / 10.0
    }

    #[inline]
    fn max(&self) -> f64 {
        self.max as f64 / 10.0
    }

    #[inline]
    fn add_value(&mut self, value: i64) {
        self.max = self.max.max(value);
        self.min = self.min.min(value);
        self.sum += value;
        self.count += 1;
    }

    #[inline]
    fn add_data(&mut self, data: &Data) {
        self.max = self.max.max(data.max);
        self.min = self.min.min(data.min);
        self.sum += data.sum;
        self.count += data.count;
    }
}

// Copy of the winning submission, with some minor changes to adapt the Java code to Rust
// https://github.com/gunnarmorling/1brc/blob/3372b6b29072af7359c5137bd1893d98828029a2/src/main/java/dev/morling/onebrc/CalculateAverage_thomaswue.java#L267
fn next_newline(memory: &[u8], prev: usize) -> usize {
    let mut prev = prev;
    loop {
        // If we were to try and slice past the end, just return the end
        if prev + 8 >= memory.len() {
            return memory.len();
        }

        let slice: [u8; 8] = memory[prev..prev + 8].try_into().unwrap();
        let word = usize::from_ne_bytes(slice);
        let input = word ^ 0x0A0A0A0A0A0A0A0A; // xor with a \n
        let newline_position = (input - 0x0101010101010101) & !input & 0x8080808080808080;
        if newline_position != 0 {
            prev += (usize::trailing_zeros(newline_position) >> 3) as usize;
            break;
        } else {
            prev += 8;
        }
    }

    prev
}

// Copy of arthurlm's version
// https://github.com/arthurlm/one-brc-rs/blob/139807ce242fb33b33f07778f38a103e4057ed23/src/main.rs#L124
#[inline]
fn parse_line(line: &[u8]) -> (&[u8], i64) {
    unsafe {
        let len = line.len();

        let float_digit = (*line.get_unchecked(len - 1) & 0x0F) as i64;
        let int_2 = (*line.get_unchecked(len - 3) & 0x0F) as i64 * 10;

        let (sep, is_neg, int_1) = match *line.get_unchecked(len - 4) {
            b';' => (len - 4, false, 0),
            b'-' => (len - 5, true, 0),
            val => {
                let int_1 = (val & 0x0F) as i64 * 100;
                match *line.get_unchecked(len - 5) {
                    b';' => (len - 5, false, int_1),
                    _ => (len - 6, true, int_1),
                }
            }
        };

        let tmp = int_1 + int_2 + float_digit;
        let temp = if is_neg { -tmp } else { tmp };
        let station = line.get_unchecked(..sep);

        (station, temp)
    }
}

fn worker<'a>(
    memory: &'a [u8],
    file_size: usize,
    seg: Arc<Mutex<usize>>,
    entries: Arc<Mutex<FxHashMap<u64, Data<'a>>>>,
) {
    let mut local_values = FxHashMap::with_capacity_and_hasher(MAP_CAPACITY, Default::default());

    loop {
        // Update the segment, so the next thread to read doesn't read the same one as us
        let segment = {
            let mut cs = seg.lock().unwrap();
            let was = *cs;
            *cs += SEGMENT_SIZE;
            was
        };
        if segment >= file_size {
            break;
        }

        let end_of_segment = file_size.min(segment + SEGMENT_SIZE);
        let end = if end_of_segment == file_size {
            file_size
        } else {
            next_newline(memory, end_of_segment)
        };
        let mut start = if segment == 0 {
            segment
        } else {
            next_newline(memory, segment) + 1
        };

        // Create a local map we will commit back to the "global" one once we finish processing
        // this segment
        while start < end {
            let newline = next_newline(memory, start);

            let line = &memory[start..newline];
            let (station, value) = parse_line(line);

            let hash = {
                let mut hasher = FxHasher::default();
                hasher.write(&station);
                hasher.finish()
            };

            local_values
                .entry(hash)
                .and_modify(|data: &mut Data<'a>| data.add_value(value))
                .or_insert_with(|| Data::new(station, value));

            start = newline + 1;
        }
    }

    // We have to send any data we have that has not already been sent
    if let Ok(mut shared_map) = entries.lock() {
        for (station, data) in local_values.into_iter() {
            shared_map
                .entry(station)
                .and_modify(|map_data| map_data.add_data(&data))
                .or_insert(data);
        }
    }
}

/// spawn_child spawns a child process that will actually do the data processing.
/// The parent process, the one calling this function, waits for the output then exits.
/// This saves the required time to call [Drop] for [Mmap] which is about 25ms
fn spawn_child() {
    let mut child = Command::new(env::current_exe().unwrap())
        .arg("--worker")
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to create subprocess");

    let mut output = child.stdout.take().unwrap();

    let mut buf = Vec::with_capacity(MAP_CAPACITY * ESTIMATED_PRINT_SIZE);
    output.read_to_end(&mut buf).unwrap();
    stdout().lock().write_all(&buf).unwrap();
}

fn main() -> io::Result<()> {
    // Only "main" program is being run, not via a subprocess
    if env::args().len() == 1 {
        spawn_child();
        return Ok(());
    }

    let mapped_file = Mmap::from_file_name(INPUT_FILE_NAME)?;
    let file_size = mapped_file.size;
    let file_data = mapped_file.as_slice();

    let current_segment = Arc::new(Mutex::new(0));
    let entries = Arc::new(Mutex::new(FxHashMap::with_capacity_and_hasher(
        MAP_CAPACITY,
        Default::default(),
    )));

    let cores = available_parallelism()?.get();
    let workers: Vec<_> = (0..cores)
        .map(|_| {
            let mmap_data = file_data;
            let segment = current_segment.clone();
            let map = entries.clone();
            thread::Builder::new()
                .stack_size(1024)
                .spawn(move || worker(mmap_data, file_size, segment, map))
        })
        .collect();

    for worker in workers {
        worker?.join().unwrap();
    }

    if let Ok(entry_list) = entries.lock() {
        let mut records: Vec<_> = entry_list.values().collect();
        records.sort_unstable_by_key(|d| d.name);

        let mut writer: Vec<u8> = Vec::with_capacity(MAP_CAPACITY * ESTIMATED_PRINT_SIZE);
        writer.push(b'{');
        for (i, val) in records.into_iter().enumerate() {
            if i > 0 {
                writer.extend_from_slice(b", ");
            }

            writer.extend_from_slice(val.name);
            writer.push(b'=');

            write!(
                writer,
                "{:.1}/{:.1}/{:.1}",
                val.min(),
                val.mean(),
                val.max(),
            )?;
        }
        writer.extend_from_slice(b"}\n");
        stdout().lock().write_all(&writer)?;
    };

    let f = unsafe {
        let out = stdout().as_raw_fd();
        File::from_raw_fd(out)
    };
    drop(f);
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse_line_test() {
        assert_eq!(
            parse_line("phoenix;50.0".as_bytes()),
            ("phoenix".as_bytes(), 500)
        );
        assert_eq!(
            parse_line("phoenix;-50.0".as_bytes()),
            ("phoenix".as_bytes(), -500)
        );
    }
}
