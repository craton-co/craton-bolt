// SPDX-License-Identifier: Apache-2.0
//! Bounds checks on untrusted Parquet structure, run before `parquet` reads
//! it.
//!
//! `parquet` 53 trusts several sizes it reads from a file and allocates them
//! before it notices the bytes are not there:
//!
//! * a page header's `compressed_page_size` — `Vec::with_capacity` before the
//!   page read (`SerializedPageReader::get_next_page`);
//! * its `uncompressed_page_size` — the decompression buffer (`decode_page`);
//! * binary lengths inside a page header — thrift's stream reader zero-fills
//!   them (`vec![0; len]`);
//! * a bloom filter's length and its header's `numBytes`
//!   (`File::get_bytes` reserves the length up front);
//! * thrift list lengths in the footer (`Vec::with_capacity(count)`).
//!
//! Each is up to `i32::MAX` (binaries and lists: `u32::MAX`), so a file of a
//! few hundred bytes can make it reserve gigabytes, and on a host without
//! memory overcommit a failed reservation aborts the process — no error, no
//! unwind. The checks here bound every such size by the bytes really present
//! (a decompressed page by [`MAX_PAGE_UNCOMPRESSED`]) and report a violation
//! as an error message instead.

use std::io::{Read, Seek, SeekFrom};

/// Largest decompressed page accepted (512 MiB). Writers cut pages at about
/// 1 MiB; a page this large is either hostile or unreadable on a GPU anyway.
pub(crate) const MAX_PAGE_UNCOMPRESSED: i64 = 512 << 20;

/// Largest page header accepted (16 MiB; real ones are tens of bytes plus
/// statistics).
const MAX_PAGE_HEADER: u64 = 16 << 20;

/// Deepest thrift nesting walked (thrift's own skip limit).
const MAX_DEPTH: u32 = 64;

/// Why a thrift walk stopped.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ThriftError {
    /// The struct needs more bytes than the buffer holds.
    Truncated,
    /// The bytes are not a compact-protocol struct.
    Invalid(&'static str),
}

/// A field value the checks look at: integers, and (one level down) the
/// integer fields of a nested struct. Everything else is skipped.
#[derive(Debug)]
enum Value {
    Int(i64),
    Struct(Vec<(i16, Value)>),
    Other,
}

/// A minimal thrift compact-protocol reader that only measures and skips:
/// `parquet`'s own slice reader is crate-private in 53. Every length it reads
/// is checked against the bytes left before it is used.
struct Walker<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Walker<'_> {
    fn left(&self) -> u64 {
        (self.buf.len() - self.pos) as u64
    }

    fn byte(&mut self) -> Result<u8, ThriftError> {
        let b = *self.buf.get(self.pos).ok_or(ThriftError::Truncated)?;
        self.pos += 1;
        Ok(b)
    }

    fn skip(&mut self, n: u64) -> Result<(), ThriftError> {
        if n > self.left() {
            return Err(ThriftError::Truncated);
        }
        self.pos += n as usize;
        Ok(())
    }

    fn varint(&mut self) -> Result<u64, ThriftError> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.byte()?;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(ThriftError::Invalid("varint longer than 10 bytes"))
    }

    fn zigzag(&mut self) -> Result<i64, ThriftError> {
        let v = self.varint()?;
        Ok((v >> 1) as i64 ^ -((v & 1) as i64))
    }

    /// One collection element of compact type `ty`.
    fn element(&mut self, ty: u8, depth: u32) -> Result<(), ThriftError> {
        match ty {
            // A boolean element is a whole byte (in a field it lives in the
            // field header).
            1 | 2 => self.skip(1),
            _ => self.value(ty, depth, false).map(|_| ()),
        }
    }

    fn value(&mut self, ty: u8, depth: u32, keep: bool) -> Result<Value, ThriftError> {
        match ty {
            1 | 2 => Ok(Value::Other),
            3 => self.skip(1).map(|_| Value::Other),
            4..=6 => self.zigzag().map(Value::Int),
            7 => self.skip(8).map(|_| Value::Other),
            8 => {
                let n = self.varint()?;
                self.skip(n).map(|_| Value::Other)
            }
            9 | 10 => {
                if depth >= MAX_DEPTH {
                    return Err(ThriftError::Invalid("nesting too deep"));
                }
                let h = self.byte()?;
                let mut n = u64::from(h >> 4);
                if n == 15 {
                    n = self.varint()?;
                }
                // Every element takes at least one byte.
                if n > self.left() {
                    return Err(ThriftError::Truncated);
                }
                for _ in 0..n {
                    self.element(h & 0x0f, depth + 1)?;
                }
                Ok(Value::Other)
            }
            11 => {
                if depth >= MAX_DEPTH {
                    return Err(ThriftError::Invalid("nesting too deep"));
                }
                let n = self.varint()?;
                if n == 0 {
                    return Ok(Value::Other);
                }
                let kv = self.byte()?;
                if n.saturating_mul(2) > self.left() {
                    return Err(ThriftError::Truncated);
                }
                for _ in 0..n {
                    self.element(kv >> 4, depth + 1)?;
                    self.element(kv & 0x0f, depth + 1)?;
                }
                Ok(Value::Other)
            }
            12 => {
                let fields = self.fields(depth + 1, false)?;
                Ok(if keep {
                    Value::Struct(fields)
                } else {
                    Value::Other
                })
            }
            _ => Err(ThriftError::Invalid("unknown thrift type")),
        }
    }

    /// The fields of a struct up to its stop byte: integers always, nested
    /// structs (as their integer fields) when `keep`.
    fn fields(&mut self, depth: u32, keep: bool) -> Result<Vec<(i16, Value)>, ThriftError> {
        if depth > MAX_DEPTH {
            return Err(ThriftError::Invalid("nesting too deep"));
        }
        let mut out = Vec::new();
        let mut last: i16 = 0;
        loop {
            let h = self.byte()?;
            if h == 0 {
                return Ok(out);
            }
            let delta = h >> 4;
            let id = if delta == 0 {
                i16::try_from(self.zigzag()?).map_err(|_| ThriftError::Invalid("field id"))?
            } else {
                last.checked_add(i16::from(delta))
                    .ok_or(ThriftError::Invalid("field id"))?
            };
            last = id;
            match self.value(h & 0x0f, depth, keep)? {
                Value::Other => {}
                v => out.push((id, v)),
            }
        }
    }
}

fn int_field(fields: &[(i16, Value)], id: i16) -> Option<i64> {
    fields.iter().find_map(|(f, v)| match v {
        Value::Int(i) if *f == id => Some(*i),
        _ => None,
    })
}

fn struct_field(fields: &[(i16, Value)], id: i16) -> Option<&[(i16, Value)]> {
    fields.iter().find_map(|(f, v)| match v {
        Value::Struct(s) if *f == id => Some(s.as_slice()),
        _ => None,
    })
}

/// Walk one thrift struct at the start of `buf`: its length in bytes, and
/// its integer fields with those of its directly nested structs.
fn read_struct(buf: &[u8]) -> Result<(usize, Vec<(i16, Value)>), ThriftError> {
    let mut w = Walker { buf, pos: 0 };
    let fields = w.fields(0, true)?;
    Ok((w.pos, fields))
}

/// Check that `buf` starts with a well-formed thrift struct whose every
/// length fits the buffer (the footer's `FileMetaData`).
pub(crate) fn check_struct(buf: &[u8]) -> Result<(), ThriftError> {
    let mut w = Walker { buf, pos: 0 };
    w.fields(0, false).map(|_| ())
}

fn read_at<R: Read + Seek>(r: &mut R, pos: u64, buf: &mut [u8]) -> Result<(), String> {
    r.seek(SeekFrom::Start(pos))
        .and_then(|_| r.read_exact(buf))
        .map_err(|e| format!("reading {} bytes at byte {pos}: {e}", buf.len()))
}

/// Read and check the footer metadata of a Parquet file of `file_len`
/// bytes: the trailer's metadata length must fit the file, and every thrift
/// length inside the metadata must fit the metadata. Returns the metadata
/// bytes.
pub(crate) fn read_footer<R: Read + Seek>(r: &mut R, file_len: u64) -> Result<Vec<u8>, String> {
    // "PAR1" + metadata + 4-byte length + "PAR1".
    if file_len < 12 {
        return Err(format!("file is {file_len} bytes, too small to be Parquet"));
    }
    let mut trailer = [0u8; 8];
    read_at(r, file_len - 8, &mut trailer)?;
    match &trailer[4..] {
        b"PAR1" => {}
        b"PARE" => return Err("encrypted Parquet files are not supported".into()),
        _ => return Err("no Parquet magic at the end of the file".into()),
    }
    let len = u64::from(u32::from_le_bytes([
        trailer[0], trailer[1], trailer[2], trailer[3],
    ]));
    if len > file_len - 12 {
        return Err(format!(
            "footer claims {len} bytes of metadata, but the file holds only {file_len} bytes"
        ));
    }
    let mut meta = vec![0u8; len as usize];
    read_at(r, file_len - 8 - len, &mut meta)?;
    match check_struct(&meta) {
        Ok(()) => Ok(meta),
        Err(ThriftError::Truncated) => {
            Err("footer metadata is truncated or claims more bytes than it holds".into())
        }
        Err(ThriftError::Invalid(m)) => Err(format!("malformed footer metadata: {m}")),
    }
}

/// Check the column chunk at `start` of `len` bytes (as the footer gives
/// them) of a file of `file_len` bytes: the chunk must lie inside the file,
/// and every page header in it must be well formed, with a compressed size
/// that fits the rest of the chunk, an uncompressed size of at most
/// [`MAX_PAGE_UNCOMPRESSED`], and (data page v2) level lengths that fit the
/// page. Reads the page headers only, skipping the page bodies.
pub(crate) fn check_column_chunk<R: Read + Seek>(
    r: &mut R,
    file_len: u64,
    start: i64,
    len: i64,
) -> Result<(), String> {
    let (Ok(start), Ok(len)) = (u64::try_from(start), u64::try_from(len)) else {
        return Err(format!(
            "column chunk has a negative offset or length ({start}, {len})"
        ));
    };
    if start.checked_add(len).is_none_or(|end| end > file_len) {
        return Err(format!(
            "column chunk at byte {start} of {len} bytes lies outside the file ({file_len} bytes)"
        ));
    }
    let (mut pos, mut remaining) = (start, len);
    let mut buf = Vec::new();
    while remaining > 0 {
        let mut want = remaining.min(1024);
        let (header_len, fields) = loop {
            buf.resize(want as usize, 0);
            read_at(r, pos, &mut buf)?;
            match read_struct(&buf) {
                Ok(h) => break h,
                Err(ThriftError::Truncated) if want < remaining && want < MAX_PAGE_HEADER => {
                    want = remaining.min(MAX_PAGE_HEADER).min(want * 8);
                }
                Err(ThriftError::Truncated) => {
                    return Err(format!(
                        "page header at byte {pos} runs past the end of the column chunk \
                         or exceeds {MAX_PAGE_HEADER} bytes"
                    ))
                }
                Err(ThriftError::Invalid(m)) => {
                    return Err(format!("malformed page header at byte {pos}: {m}"))
                }
            }
        };
        let body_left = remaining - header_len as u64;
        let (Some(compressed), Some(uncompressed)) = (int_field(&fields, 3), int_field(&fields, 2))
        else {
            return Err(format!("page header at byte {pos} lacks its page sizes"));
        };
        if compressed < 0 || compressed as u64 > body_left {
            return Err(format!(
                "page at byte {pos} claims {compressed} compressed bytes, \
                 but {body_left} remain in the column chunk"
            ));
        }
        if !(0..=MAX_PAGE_UNCOMPRESSED).contains(&uncompressed) {
            return Err(format!(
                "page at byte {pos} claims {uncompressed} uncompressed bytes \
                 (limit {MAX_PAGE_UNCOMPRESSED})"
            ));
        }
        if let Some(v2) = struct_field(&fields, 8) {
            let def = int_field(v2, 5).unwrap_or(0);
            let rep = int_field(v2, 6).unwrap_or(0);
            let levels = def.saturating_add(rep);
            if def < 0 || rep < 0 || levels > compressed || levels > uncompressed {
                return Err(format!(
                    "data page v2 at byte {pos} claims {levels} level bytes in a page of \
                     {compressed} compressed / {uncompressed} uncompressed bytes"
                ));
            }
        }
        let step = header_len as u64 + compressed as u64;
        pos += step;
        remaining -= step;
    }
    Ok(())
}

/// Check the bloom filter a column chunk's metadata points at (`offset`,
/// optional `length`) in a file of `file_len` bytes, the way `parquet` will
/// read it: it must lie inside the file, its header must parse, and its
/// bitset must be a non-empty run of whole blocks inside the file (an empty
/// bitset makes `Sbbf::check` index out of bounds).
pub(crate) fn check_bloom_filter<R: Read + Seek>(
    r: &mut R,
    file_len: u64,
    offset: Option<i64>,
    length: Option<i32>,
) -> Result<(), String> {
    let Some(offset) = offset else {
        return Ok(());
    };
    let off = u64::try_from(offset)
        .ok()
        .filter(|&o| o < file_len)
        .ok_or_else(|| format!("bloom filter offset {offset} lies outside the file"))?;
    let avail = file_len - off;
    let length = match length {
        Some(l) => Some(
            u64::try_from(l)
                .ok()
                .filter(|&l| l <= avail)
                .ok_or_else(|| {
                    format!("bloom filter at byte {off} claims {l} bytes, {avail} remain")
                })?,
        ),
        None => None,
    };
    // `parquet` parses the header from `length` bytes, or from 20 without one.
    let mut head = vec![0u8; length.unwrap_or(20).min(avail).min(64) as usize];
    read_at(r, off, &mut head)?;
    let (header_len, fields) = read_struct(&head)
        .map_err(|e| format!("bloom filter header at byte {off} is malformed ({e:?})"))?;
    let header_len = header_len as u64;
    let bitset = match length {
        Some(l) => l.checked_sub(header_len),
        None => int_field(&fields, 1)
            .and_then(|n| u64::try_from(n).ok())
            .filter(|&n| n <= avail - header_len),
    }
    .ok_or_else(|| format!("bloom filter at byte {off} claims a bitset beyond the file"))?;
    if bitset < 32 {
        return Err(format!(
            "bloom filter at byte {off} has a {bitset}-byte bitset (at least 32 needed)"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::format::{
        BloomFilterAlgorithm, BloomFilterCompression, BloomFilterHash, BloomFilterHeader,
        DataPageHeader, DataPageHeaderV2, Encoding, PageHeader, PageType, SplitBlockAlgorithm,
        Statistics, Uncompressed, XxHash,
    };
    use parquet::thrift::{TCompactOutputProtocol, TSerializable};
    use std::io::Cursor;

    fn encode<T: TSerializable>(v: &T) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut p = TCompactOutputProtocol::new(&mut buf);
            v.write_to_out_protocol(&mut p).unwrap();
        }
        buf
    }

    fn data_page(uncompressed: i32, compressed: i32) -> PageHeader {
        PageHeader {
            type_: PageType::DATA_PAGE,
            uncompressed_page_size: uncompressed,
            compressed_page_size: compressed,
            crc: None,
            data_page_header: Some(DataPageHeader {
                num_values: 4,
                encoding: Encoding::PLAIN,
                definition_level_encoding: Encoding::RLE,
                repetition_level_encoding: Encoding::RLE,
                statistics: Some(Statistics {
                    min_value: Some(vec![1, 2, 3]),
                    max_value: Some(vec![4, 5, 6]),
                    ..Default::default()
                }),
            }),
            index_page_header: None,
            dictionary_page_header: None,
            data_page_header_v2: None,
        }
    }

    /// A chunk of `pages` (header + `compressed_page_size` body bytes each),
    /// preceded by `pad` bytes; returns the file and the chunk length.
    fn chunk(pad: usize, pages: &[PageHeader]) -> (Vec<u8>, i64) {
        let mut f = vec![0xAB; pad];
        for p in pages {
            f.extend(encode(p));
            f.extend(std::iter::repeat_n(
                7u8,
                p.compressed_page_size.max(0) as usize,
            ));
        }
        let len = (f.len() - pad) as i64;
        (f, len)
    }

    fn check(file: &[u8], start: i64, len: i64) -> Result<(), String> {
        check_column_chunk(&mut Cursor::new(file), file.len() as u64, start, len)
    }

    #[test]
    fn walker_matches_parquet_encoding() {
        let h = data_page(100, 50);
        let bytes = encode(&h);
        let (n, fields) = read_struct(&bytes).unwrap();
        assert_eq!(n, bytes.len());
        assert_eq!(int_field(&fields, 1), Some(0));
        assert_eq!(int_field(&fields, 2), Some(100));
        assert_eq!(int_field(&fields, 3), Some(50));
        let dp = struct_field(&fields, 5).unwrap();
        assert_eq!(int_field(dp, 1), Some(4));
        // Every proper prefix is truncated, never accepted.
        for cut in 0..bytes.len() {
            assert_eq!(
                read_struct(&bytes[..cut]).unwrap_err(),
                ThriftError::Truncated
            );
        }
    }

    #[test]
    fn valid_chunks_pass() {
        let (f, len) = chunk(
            4,
            &[data_page(30, 30), data_page(1000, 10), data_page(0, 0)],
        );
        check(&f, 4, len).unwrap();
        // A header larger than the first read (statistics of 5000 bytes).
        let mut big = data_page(8, 8);
        big.data_page_header.as_mut().unwrap().statistics = Some(Statistics {
            min_value: Some(vec![9; 5000]),
            ..Default::default()
        });
        let (f, len) = chunk(0, &[big, data_page(4, 4)]);
        check(&f, 0, len).unwrap();
    }

    #[test]
    fn oversized_and_negative_page_sizes_are_rejected() {
        // compressed_page_size = i32::MAX in a chunk of a few bytes.
        let mut bytes = encode(&data_page(10, i32::MAX));
        bytes.extend([0u8; 10]);
        let e = check(&bytes, 0, bytes.len() as i64).unwrap_err();
        assert!(e.contains("2147483647 compressed bytes"), "{e}");
        let bytes = encode(&data_page(10, -1));
        let e = check(&bytes, 0, bytes.len() as i64).unwrap_err();
        assert!(e.contains("-1 compressed bytes"), "{e}");
        // uncompressed_page_size beyond the cap, or negative.
        for u in [i32::MAX, -5] {
            let (f, len) = chunk(0, &[data_page(u, 4)]);
            let e = check(&f, 0, len).unwrap_err();
            assert!(e.contains("uncompressed bytes"), "{e}");
        }
        // A second page that overruns the chunk.
        let (mut f, _) = chunk(0, &[data_page(4, 4)]);
        let first = f.len() as i64;
        f.extend(encode(&data_page(4, 1 << 20)));
        assert!(check(&f, 0, f.len() as i64).is_err());
        check(&f, 0, first).unwrap();
    }

    #[test]
    fn data_page_v2_level_lengths_must_fit() {
        let mut h = data_page(16, 16);
        h.type_ = PageType::DATA_PAGE_V2;
        h.data_page_header = None;
        h.data_page_header_v2 = Some(DataPageHeaderV2 {
            num_values: 1,
            num_nulls: 0,
            num_rows: 1,
            encoding: Encoding::PLAIN,
            definition_levels_byte_length: 8,
            repetition_levels_byte_length: 8,
            is_compressed: Some(true),
            statistics: None,
        });
        let (f, len) = chunk(0, std::slice::from_ref(&h));
        check(&f, 0, len).unwrap();
        h.data_page_header_v2
            .as_mut()
            .unwrap()
            .repetition_levels_byte_length = 9;
        let (f, len) = chunk(0, &[h]);
        assert!(check(&f, 0, len).unwrap_err().contains("level bytes"));
    }

    #[test]
    fn chunk_ranges_must_lie_inside_the_file() {
        let (f, len) = chunk(8, &[data_page(4, 4)]);
        assert!(check(&f, 8, len + 1)
            .unwrap_err()
            .contains("outside the file"));
        assert!(check(&f, -1, len).unwrap_err().contains("negative"));
        assert!(check(&f, 8, -1).unwrap_err().contains("negative"));
        assert!(check(&f, i64::MAX, i64::MAX).is_err());
        // Bytes that are not a page header.
        assert!(check(&[0xFF; 32], 0, 32).is_err());
        check(&f, 0, 0).unwrap();
    }

    #[test]
    fn hostile_thrift_lengths_are_truncation_not_allocation() {
        // Field 1, binary, length u32::MAX: no bytes follow.
        let bin = [0x18, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F];
        assert_eq!(check_struct(&bin).unwrap_err(), ThriftError::Truncated);
        // Field 1, list of i64 with 2^31 elements.
        let list = [0x19, 0xF6, 0xFF, 0xFF, 0xFF, 0xFF, 0x07, 0, 0];
        assert_eq!(check_struct(&list).unwrap_err(), ThriftError::Truncated);
        // Field 1, map of 2^31 entries.
        let map = [0x1B, 0xFF, 0xFF, 0xFF, 0xFF, 0x07, 0x55, 0];
        assert_eq!(check_struct(&map).unwrap_err(), ThriftError::Truncated);
        // A varint that never ends.
        assert!(matches!(
            check_struct(&[0x15, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0]),
            Err(ThriftError::Invalid(_))
        ));
        // Nesting past the depth limit: struct in struct in ...
        let mut deep = vec![0x1C; 200];
        deep.extend([0u8; 201]);
        assert_eq!(
            check_struct(&deep).unwrap_err(),
            ThriftError::Invalid("nesting too deep")
        );
        // Unknown type nibble.
        assert!(matches!(
            check_struct(&[0x1E, 0]),
            Err(ThriftError::Invalid(_))
        ));
    }

    #[test]
    fn footer_lengths_are_checked() {
        let mut f = b"PAR1".to_vec();
        let meta = [0x15, 0x02, 0x00]; // { 1: i32 = 1 }
        f.extend(meta);
        f.extend((meta.len() as u32).to_le_bytes());
        f.extend(b"PAR1");
        let n = f.len() as u64;
        assert_eq!(read_footer(&mut Cursor::new(&f), n).unwrap(), meta);
        // Metadata length beyond the file.
        let mut bad = f.clone();
        let at = bad.len() - 8;
        bad[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(read_footer(&mut Cursor::new(&bad), n)
            .unwrap_err()
            .contains("footer claims"));
        // A list in the metadata claiming more elements than bytes.
        let mut bad = b"PAR1".to_vec();
        let meta = [0x19, 0xF6, 0xFF, 0xFF, 0xFF, 0x07, 0];
        bad.extend(meta);
        bad.extend((meta.len() as u32).to_le_bytes());
        bad.extend(b"PAR1");
        let n = bad.len() as u64;
        assert!(read_footer(&mut Cursor::new(&bad), n)
            .unwrap_err()
            .contains("truncated"));
        assert!(read_footer(&mut Cursor::new(b"PAR1PAR1"), 8).is_err());
    }

    fn bloom_header(num_bytes: i32) -> Vec<u8> {
        encode(&BloomFilterHeader {
            num_bytes,
            algorithm: BloomFilterAlgorithm::BLOCK(SplitBlockAlgorithm {}),
            hash: BloomFilterHash::XXHASH(XxHash {}),
            compression: BloomFilterCompression::UNCOMPRESSED(Uncompressed {}),
        })
    }

    #[test]
    fn bloom_filter_sizes_are_checked() {
        let mut f = vec![0u8; 10];
        let h = bloom_header(64);
        let hl = h.len() as i32;
        f.extend(&h);
        f.extend([0u8; 64]);
        let n = f.len() as u64;
        let ck = |f: &[u8], off, len| check_bloom_filter(&mut Cursor::new(f), n, off, len);
        ck(&f, None, None).unwrap();
        ck(&f, Some(10), None).unwrap();
        ck(&f, Some(10), Some(hl + 64)).unwrap();
        assert!(ck(&f, Some(10), Some(i32::MAX)).is_err());
        assert!(ck(&f, Some(10), Some(-1)).is_err());
        assert!(ck(&f, Some(-3), None).is_err());
        assert!(ck(&f, Some(n as i64), None).is_err());
        // An empty bitset.
        assert!(ck(&f, Some(10), Some(hl)).is_err());
        // numBytes = i32::MAX with no length in the metadata.
        let mut g = vec![0u8; 10];
        g.extend(bloom_header(i32::MAX));
        g.extend([0u8; 64]);
        let m = g.len() as u64;
        assert!(check_bloom_filter(&mut Cursor::new(&g), m, Some(10), None)
            .unwrap_err()
            .contains("beyond the file"));
    }
}
