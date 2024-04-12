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
const VEC_CAPACITY: usize = 1 << 17;
const VEC_END: usize = VEC_CAPACITY - 1;
/// Cover the size of the format string to allow for us to have an approximately
/// correct write buffer size
const ESTIMATED_PRINT_SIZE: usize = 20 // station name
        + 1 // equals
        + (3 * 5) // -?\d?\d.\d for the min/max/ave
        + 2 // slashes between the numbers
        + 2 // comma and space
;
const MASK_1: [i64; 9] = [
    0xFF,
    0xFFFF,
    0xFFFFFF,
    0xFFFFFFFF,
    0xFFFFFFFFFF,
    0xFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFF,
    -1,
    -1,
];
const MASK_2: [i64; 9] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, -1];

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
    #[inline]
    fn drop(&mut self) {
        unsafe { libc::munmap(self.addr, self.size) };
    }
}

// i64 is definitely overkill for the min and max, but i16/i32
// doesn't seem to provide much of a benefit in terms of speed. Likely because the
// CPU can load it all into memory anyway ¯\_(ツ)_/¯
#[derive(Clone)]
struct Data {
    name_address: usize,
    word_1: i64,
    word_2: i64,
    sum: i64,
    count: i64,
    min: i32,
    max: i32,
}

impl Default for Data {
    #[inline]
    fn default() -> Self {
        Data {
            name_address: usize::MAX,
            word_1: 0,
            word_2: 0,
            min: 0,
            max: 0,
            sum: 0,
            count: 0,
        }
    }
}

impl Data {
    #[inline]
    fn name<'a>(&self, memory: &'a [u8]) -> &'a [u8] {
        unsafe {
            let mut sep = self.name_address;
            while *memory.get_unchecked(sep) != b';' {
                sep += 1;
            }
            memory.get_unchecked(self.name_address..sep)
        }
    }

    #[inline]
    fn mean(&self) -> f64 {
        (self.sum as f64) / 10.0 / (self.count as f64)
    }

    #[inline]
    fn min(&self) -> f32 {
        self.min as f32 / 10.0
    }

    #[inline]
    fn max(&self) -> f32 {
        self.max as f32 / 10.0
    }

    #[inline]
    fn add_value(&mut self, value: i64) {
        self.max = self.max.max(value as i32);
        self.min = self.min.min(value as i32);
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

    #[inline]
    fn is_unmodified(&self) -> bool {
        self.name_address == usize::MAX
    }
}

// Copy of the winning submission, with some minor changes to adapt the Java code to Rust
// https://github.com/gunnarmorling/1brc/blob/3372b6b29072af7359c5137bd1893d98828029a2/src/main/java/dev/morling/onebrc/CalculateAverage_thomaswue.java#L267
fn next_newline(memory: &[u8], prev: usize) -> usize {
    let mut prev = prev;
    loop {
        // If we were to try and slice past the end, just return the end
        if prev + 8 >= memory.len() {
            return memory.len() - 1;
        }

        let word = get_word(memory, prev) as usize;
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
fn get_separator(word: i64) -> i64 {
    let input = word.bitxor(0x3B3B3B3B3B3B3B3B);
    (input - 0x0101010101010101) & !input & 0x8080808080808080u64 as i64
}

#[inline]
fn hash_to_index(hash: i64) -> i64 {
    let hash = hash as usize;
    let as_int = hash ^ (hash >> 33) ^ (hash >> 15);
    (as_int & VEC_END) as i64
}

#[inline]
fn get_word(memory: &[u8], addr: usize) -> i64 {
    if addr + 8 >= memory.len() {
        let mut buff = [0; 8];
        let range = addr..memory.len();
        buff[..range.len()].copy_from_slice(&memory[range]);
        i64::from_ne_bytes(buff)
    } else {
        let buff: [u8; 8] = memory[addr..addr + 8].try_into().unwrap();
        i64::from_ne_bytes(buff)
    }
}

#[inline]
fn parse_number(word: i64, decimal_pos: i64) -> i64 {
    let shift = 28 - decimal_pos;
    let signed = (!word << 59) >> 63;
    let mask = !(signed & 0xFF);
    let digits = ((word & mask) << shift) & 0x0F000F0F00;
    let abs = ((digits * 0x640A0001) >> 32) & 0x3FF;

    (abs ^ signed) - signed
}

#[inline]
fn scan_number(memory: &[u8], addr: &mut usize) -> i64 {
    let word = get_word(memory, *addr + 1);
    let decimal = i64::trailing_zeros(!word & 0x10101000) as i64;
    *addr += ((decimal >> 3) + 4) as usize;
    parse_number(word, decimal)
}

fn locate_and_update(
    word_1: i64,
    word_2: i64,
    sep_1: i64,
    sep_2: i64,
    memory: &[u8],
    start: &mut usize,
    values: &mut [Data],
) {
    let mut word_1 = word_1;
    let mut word_2 = word_2;
    let mut hash;
    let name_address = *start;

    if (sep_1 | sep_2) != 0 {
        // The ';' exists in the first 16 bytes. Faster path
        let word_1_len = (i64::trailing_zeros(sep_1) >> 3) as usize;
        let word_2_len = (i64::trailing_zeros(sep_2) >> 3) as usize;
        let mask = unsafe { MASK_2.get_unchecked(word_1_len) };
        word_1 &= unsafe { MASK_1.get_unchecked(word_1_len) };
        word_2 = unsafe { mask & word_2 & MASK_1.get_unchecked(word_2_len) };
        hash = word_1 ^ word_2;
        *start += word_1_len + (word_2_len as i64 & mask) as usize;

        let index = hash_to_index(hash) as usize;
        let entry = unsafe { values.get_unchecked_mut(index) };
        if entry.word_1 == word_1 && entry.word_2 == word_2 {
            let value = scan_number(memory, &mut *start);
            entry.add_value(value);
            return;
        }
    } else {
        // Not in the first 16 bytes, need to locate it
        hash = word_1 ^ word_2;
        *start += 16;
        loop {
            word_1 = get_word(memory, *start);
            let mask = get_separator(word_1);
            if mask != 0 {
                let zeros = i64::trailing_zeros(mask) as usize;
                word_1 <<= 63 - zeros;
                *start += zeros >> 3;
                hash ^= word_1;
                break;
            } else {
                *start += 8;
                hash ^= word_1;
            }
        }
    }

    let name_length = (*start - name_address) as i64 + 1;
    let mut index = hash_to_index(hash) as usize;
    let mut entry;
    'outer: loop {
        entry = unsafe { values.get_unchecked_mut(index) };

        // Add the value if it DNE
        if entry.is_unmodified() {
            let len = name_length as usize;
            entry.name_address = name_address;
            entry.word_1 = get_word(memory, name_address);
            if len <= 8 {
                entry.word_1 &= unsafe { MASK_1.get_unchecked(len - 1) };
            } else if len < 16 {
                entry.word_2 = get_word(memory, name_address + 8);
                entry.word_2 &= unsafe { MASK_1.get_unchecked(len - 9) };
            }
            break;
        }

        // If the value already exists, we want to check for collisions to ensure we have the correct
        // entry for the station
        let mut i = 0;
        while i < name_length - 8 {
            let entry_word = get_word(memory, entry.name_address + i as usize);
            let other_word = get_word(memory, name_address + i as usize);
            if entry_word != other_word {
                index = (index + 31) & VEC_END;
                continue 'outer;
            }
            i += 8;
        }

        let remainder = 64 - ((name_length - i) << 3);
        let entry_word = get_word(memory, entry.name_address + i as usize);
        let other_word = get_word(memory, name_address + i as usize);
        if ((entry_word ^ other_word) << remainder) == 0 {
            break;
        } else {
            index = (index + 31) & VEC_END;
        }
    }

    let value = scan_number(memory, &mut *start);
    entry.add_value(value);
}

fn worker<'a>(
    memory: &'a [u8],
    file_size: usize,
    seg: Arc<Mutex<usize>>,
    entries: Arc<Mutex<FxHashMap<&'a [u8], Data>>>,
) {
    let mut local_values = vec![Data::default(); VEC_CAPACITY];

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
            let w1 = get_word(memory, start);
            let w2 = get_word(memory, start + 8);
            let sep1 = get_separator(w1);
            let sep2 = get_separator(w2);

            locate_and_update(w1, w2, sep1, sep2, memory, &mut start, &mut local_values);
        }
    }

    // We have to send any data we have that has not already been sent
    if let Ok(mut shared_map) = entries.lock() {
        for data in local_values {
            if data.is_unmodified() {
                continue;
            }
            shared_map
                .entry(data.name(memory))
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
        let mut records: Vec<_> = entry_list.iter().collect();
        records.sort_unstable_by_key(|(name, _)| *name);

        let mut writer: Vec<u8> = Vec::with_capacity(MAP_CAPACITY * ESTIMATED_PRINT_SIZE);
        writer.push(b'{');
        for (i, (name, val)) in records.into_iter().enumerate() {
            if i > 0 {
                writer.extend_from_slice(b", ");
            }

            writer.extend_from_slice(name);
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
    fn get_delim() {
        let word = get_word("asdf;50.0".as_bytes(), 0);
        assert_ne!(get_separator(word), 0);
        let word2 = get_word("asdfasdfqwe".as_bytes(), 0);
        assert_eq!(get_separator(word2), 0);
    }
}
