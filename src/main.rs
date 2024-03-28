use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{stdout, Read, Write};
use std::mem::size_of;
use std::ops::BitXor;
use std::os::fd::AsRawFd;
use std::sync::{Arc, Mutex};
use std::thread::available_parallelism;
use std::{io, slice, thread};

const INPUT_FILE_NAME: &str = "measurements.txt";
const SEGMENT_SIZE: usize = 1 << 21;
const HASH_CONST: usize = 0x517cc1b727220a95;
const MAP_CAPACITY: usize = 512;

type FastEnoughHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FastEnoughHasher>>;

#[derive(Default, Clone)]
struct FastEnoughHasher {
    hash: usize,
}

impl FastEnoughHasher {
    #[inline]
    fn add_to_hash(&mut self, i: usize) {
        self.hash = self.hash.rotate_left(5).bitxor(i).wrapping_mul(HASH_CONST);
    }
}

impl Hasher for FastEnoughHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash as u64
    }

    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
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
    if slice.len() < N {
        None
    } else {
        let (first, rest) = slice.split_at(N);
        *slice = rest;
        Some(first.try_into().unwrap())
    }
}

struct Mmap {
    addr: *mut c_void,
    size: usize,
}

impl Mmap {
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

    fn as_slice(&self) -> &'static [u8] {
        unsafe { slice::from_raw_parts(self.addr.cast(), self.size) }
    }
}

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

#[inline]
fn parse_to_int(bytes: &[u8]) -> i64 {
    let is_negative = unsafe { *bytes.get_unchecked(0) == b'-' };
    let mut index = if is_negative { 1 } else { 0 };

    let num = unsafe {
        let mut num = if *bytes.get_unchecked(index + 1) == b'.' {
            // -?\d.\d
            let value = 10 * (*bytes.get_unchecked(index) - b'0') as i64;
            index += 1;
            value
        } else {
            // -?\d\d.\d
            let value = 100 * (*bytes.get_unchecked(index) - b'0') as i64
                + 10 * (*bytes.get_unchecked(index + 1) - b'0') as i64;
            index += 2;
            value
        };
        index += 1; // skip .

        // read the decimal
        num += (*bytes.get_unchecked(index) - b'0') as i64;
        num
    };

    if is_negative {
        -num
    } else {
        num
    }
}

#[inline]
#[allow(dead_code)]
fn parse_to_int_bit_shift(bytes: &[u8]) -> i64 {
    let mut buff: [u8; 8] = [0u8; size_of::<i64>()];
    buff.as_mut().write(bytes).unwrap();

    let word = i64::from_ne_bytes(buff);
    let decimal_pos = i64::trailing_zeros(!word & 0x10101000) as i64;

    let shift = 28 - decimal_pos;
    let signed = (!word << 59) >> 63;
    let mask = !(signed & 0xFF);
    let digits = ((word & mask) << shift) & 0x0F000F0F00;
    let abs = ((digits * 0x640A0001) >> 32) & 0x3FF;

    (abs ^ signed) - signed
}

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
    entries: Arc<Mutex<FastEnoughHashMap<u64, Data<'a>>>>,
) {
    let mut local_values: FastEnoughHashMap<u64, Data> =
        FastEnoughHashMap::with_capacity_and_hasher(MAP_CAPACITY, Default::default());

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
                let mut hasher = FastEnoughHasher::default();
                hasher.write(&station);
                hasher.finish()
            };

            local_values
                .entry(hash)
                .and_modify(|data| data.add_value(value))
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

fn main() {
    let fp = File::open(INPUT_FILE_NAME).unwrap();

    let mapped_file = Mmap::from_file(fp).unwrap();
    let file_size = mapped_file.size;
    let file_data = mapped_file.as_slice();

    let current_segment = Arc::new(Mutex::new(0));
    let entries = Arc::new(Mutex::new(FastEnoughHashMap::with_capacity_and_hasher(
        MAP_CAPACITY,
        Default::default(),
    )));

    let cores = available_parallelism().unwrap().get();
    let workers: Vec<_> = (0..cores)
        .map(|_| {
            let mmap_data = file_data;
            let segment = current_segment.clone();
            let map = entries.clone();
            thread::spawn(move || worker(mmap_data, file_size, segment, map))
        })
        .collect();

    for worker in workers {
        worker.join().unwrap();
    }

    if let Ok(entry_list) = entries.lock() {
        let estimated_size = 20 + 1 + 15 + 2 + 2;

        let mut writer: Vec<u8> = Vec::with_capacity(entry_list.len() * estimated_size);
        writer.push(b'{');
        for (i, val) in entry_list.values().enumerate() {
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
            )
            .unwrap();
        }
        writer.extend_from_slice(b"}\n");
        stdout().lock().write_all(&writer).unwrap();
    };
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse_to_int_test() {
        assert_eq!(parse_to_int("99.9".as_bytes()), 999);
        assert_eq!(parse_to_int("-99.9".as_bytes()), -999);
    }

    #[test]
    fn parse_to_int_2_test() {
        assert_eq!(parse_to_int_bit_shift("99.9".as_bytes()), 999);
        assert_eq!(parse_to_int_bit_shift("-99.9".as_bytes()), -999);
    }
}
