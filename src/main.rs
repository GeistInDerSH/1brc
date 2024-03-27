use std::collections::HashMap;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufWriter, stdout, Write};
use std::ops::BitXor;
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::available_parallelism;

use memmap2::Mmap;

const INPUT_FILE_NAME: &str = "measurements.txt";
const BUFFER_CAPACITY: usize = 512 * 512;
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

#[derive(Clone)]
struct Data {
    name: String,
    min: i64,
    max: i64,
    sum: i64,
    count: i64,
}

impl Data {
    #[inline]
    fn new(name: String, val: i64) -> Self {
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
    let is_negative = bytes[0] == b'-';
    let mut index = if is_negative { 1 } else { 0 };

    // read the first int
    let mut num = (bytes[index] - b'0') as i64;
    index += 1;

    // optionally read the second int
    if bytes[index] != b'.' {
        num *= 10;
        num += (bytes[index] - b'0') as i64;
        index += 1;
    }
    index += 1; // skip .

    // read the decimal
    num *= 10;
    num += (bytes[index] - b'0') as i64;

    if is_negative {
        -num
    } else {
        num
    }
}

fn next_newline(memory: &Mmap, prev: usize) -> usize {
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

fn worker(
    memory: Arc<Mmap>,
    file_size: usize,
    seg: Arc<Mutex<usize>>,
    entries: Arc<Mutex<FastEnoughHashMap<u64, Data>>>,
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
            next_newline(&memory, end_of_segment)
        };
        let mut start = if segment == 0 {
            segment
        } else {
            next_newline(&memory, segment) + 1
        };

        // Create a local map we will commit back to the "global" one once we finish processing
        // this segment
        while start < end {
            let newline = next_newline(&memory, start);

            let mut position = newline - 1;
            for c in memory[start..newline].iter().rev() {
                if *c == b';' {
                    break;
                }
                position -= 1;
            }

            let value = parse_to_int(&memory[position + 1..newline]);
            let hash = {
                let mut hasher = FastEnoughHasher::default();
                hasher.write(&memory[start..position]);
                hasher.finish()
            };

            local_values
                .entry(hash)
                .and_modify(|data| data.add_value(value))
                .or_insert_with(|| {
                    let station =
                        unsafe { std::str::from_utf8_unchecked(&memory[start..position]) };
                    Data::new(station.to_string(), value)
                });

            start = newline + 1;
        }

        // Try to update the shared map, or just move on if we can't get the lock
        if let Ok(mut shared_entries) = entries.try_lock() {
            for (station, data) in &local_values {
                shared_entries
                    .entry(*station)
                    .and_modify(|map_data| map_data.add_data(&data))
                    .or_insert(data.clone());
            }
            local_values.clear();
        }
    }

    if local_values.is_empty() {
        return;
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
    let mapped_file = unsafe { Mmap::map(&fp).unwrap() };
    let file_size = mapped_file.len();

    let current_segment = Arc::new(Mutex::new(0));
    let memory_region = Arc::new(mapped_file);
    let entries = Arc::new(Mutex::new(FastEnoughHashMap::with_capacity_and_hasher(
        MAP_CAPACITY,
        Default::default(),
    )));

    let cores = available_parallelism().unwrap().get();
    let workers: Vec<_> = (0..cores)
        .map(|_| {
            let region = memory_region.clone();
            let segment = current_segment.clone();
            let map = entries.clone();
            thread::spawn(move || worker(region, file_size, segment, map))
        })
        .collect();

    for worker in workers {
        worker.join().unwrap();
    }

    if let Ok(entry_list) = entries.lock() {
        let mut writer = BufWriter::with_capacity(BUFFER_CAPACITY, stdout());
        writer.write_all("{".as_bytes()).unwrap();

        let size = entry_list.len() - 1;
        for (i, val) in entry_list.values().enumerate() {
            writer
                .write_fmt(format_args!(
                    "{}={:.1}/{:.1}/{:.1}{}",
                    val.name,
                    val.min(),
                    val.mean(),
                    val.max(),
                    if i == size { "" } else { ", " },
                ))
                .unwrap();
        }
        writer.write_all("}".as_bytes()).unwrap();
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
}
