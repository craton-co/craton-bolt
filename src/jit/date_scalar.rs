// SPDX-License-Identifier: Apache-2.0

//! PTX codegen + host reference math for the date/time scalar functions
//! `EXTRACT(field FROM ts)` and `DATE_TRUNC(unit, ts)`.
//!
//! Both functions lower to **pure integer arithmetic** on the fixed-width
//! storage of the temporal types:
//!
//!   - `Date32`    — `i32` days since the Unix epoch (1970-01-01).
//!   - `Timestamp` — `i64` ticks since the Unix epoch, where one tick is a
//!     [`TimeUnit`] (second / milli / micro / nano).
//!
//! ## Civil-date decomposition
//!
//! `YEAR` / `MONTH` / `DAY` need the proleptic-Gregorian civil date for a given
//! day count. We use Howard Hinnant's branch-free `civil_from_days` algorithm
//! (<http://howardhinnant.github.io/date_algorithms.html#civil_from_days>),
//! which is exact for the entire `i32` day range and uses only `+ - * / %` on
//! signed integers — a perfect fit for both the host reference path and the
//! GPU kernel. The intra-day fields (`HOUR` / `MINUTE` / `SECOND`) are plain
//! modular arithmetic on the tick count and are only defined on `Timestamp`.
//!
//! ## Two surfaces
//!
//! 1. **Host reference** ([`civil_from_days`], [`extract_date32`],
//!    [`extract_timestamp`], [`date_trunc_date32_days`],
//!    [`date_trunc_timestamp_ticks`]) — the single source of truth for the
//!    arithmetic, used by the host fallback path and pinned by unit tests.
//! 2. **PTX emitters** ([`compile_extract_kernel`], [`compile_date_trunc_kernel`])
//!    — emit a per-row kernel that mirrors the host arithmetic 1:1, validated
//!    by PTX-assertion tests (we can't run CUDA here).
//!
//! The PTX kernels share the standard one-input-one-output element-wise ABI:
//!
//! ```text
//! .visible .entry <entry>(
//!     .param .u64 input_ptr,    // *const i32 (Date32) | *const i64 (Timestamp)
//!     .param .u64 output_ptr,   // *mut i64 (EXTRACT) | same dtype as input (DATE_TRUNC)
//!     .param .u32 n_rows
//! )
//! ```

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{DataType, DateField, DateTruncUnit, TimeUnit};

/// Adapt an `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("date_scalar: write failed: {}", e))
}

/// Ticks per day for a [`TimeUnit`]. Used to split a `Timestamp` tick count
/// into a day count (for the calendar fields) and an intra-day remainder (for
/// the clock fields).
pub fn ticks_per_day(unit: TimeUnit) -> i64 {
    let per_second: i64 = match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    };
    per_second * 86_400
}

/// Ticks per second for a [`TimeUnit`].
pub fn ticks_per_second(unit: TimeUnit) -> i64 {
    match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    }
}

// ===========================================================================
// Host reference arithmetic.
// ===========================================================================

/// Floor division for signed integers (round toward negative infinity), as
/// required by the civil-date math for pre-epoch (negative) day counts. Rust's
/// `/` truncates toward zero, so we adjust when the operands' signs differ and
/// the division isn't exact.
fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if (r != 0) && ((r < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// Floor modulo matching [`floor_div`] (result has the sign of `b`).
fn floor_mod(a: i64, b: i64) -> i64 {
    let r = a % b;
    if (r != 0) && ((r < 0) != (b < 0)) {
        r + b
    } else {
        r
    }
}

/// Hinnant's `civil_from_days`: map a count of days since 1970-01-01 to the
/// proleptic-Gregorian `(year, month, day)` triple. `month` is 1..=12 and
/// `day` is 1..=31. Exact across the whole `i32` day range.
pub fn civil_from_days(days: i64) -> (i64, i64, i64) {
    // Shift the epoch to 0000-03-01 so leap days fall at the end of the
    // 400-year cycle. `z` is the shifted day number.
    let z = days + 719_468;
    let era = floor_div(z, 146_097);
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}

/// Inverse of [`civil_from_days`]: map a `(year, month, day)` to days since
/// 1970-01-01. Used by `DATE_TRUNC` to rebuild the truncated day count.
pub fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = floor_div(y, 400);
    let yoe = y - era * 400; // [0, 399]
    let mp = if month > 2 { month - 3 } else { month + 9 }; // [0, 11]
    let doy = (153 * mp + 2) / 5 + day - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Host reference for `EXTRACT(field FROM <Date32>)`. `days` is the raw
/// `Date32` value (days since epoch). Intra-day fields are rejected by the
/// type-checker before reaching here, but we map them to `0` defensively.
pub fn extract_date32(field: DateField, days: i32) -> i64 {
    let (y, m, d) = civil_from_days(days as i64);
    match field {
        DateField::Year => y,
        DateField::Month => m,
        DateField::Day => d,
        // Undefined for a bare date; the planner rejects this combination.
        DateField::Hour | DateField::Minute | DateField::Second => 0,
    }
}

/// Host reference for `EXTRACT(field FROM <Timestamp>)`. `ticks` is the raw
/// timestamp value in `unit` resolution.
pub fn extract_timestamp(field: DateField, ticks: i64, unit: TimeUnit) -> i64 {
    let tpd = ticks_per_day(unit);
    let days = floor_div(ticks, tpd);
    match field {
        DateField::Year | DateField::Month | DateField::Day => {
            let (y, m, d) = civil_from_days(days);
            match field {
                DateField::Year => y,
                DateField::Month => m,
                DateField::Day => d,
                _ => unreachable!(),
            }
        }
        DateField::Hour => {
            let tod = floor_mod(ticks, tpd); // ticks-of-day, [0, tpd)
            tod / ticks_per_second(unit) / 3600
        }
        DateField::Minute => {
            let tod = floor_mod(ticks, tpd);
            (tod / ticks_per_second(unit) / 60) % 60
        }
        DateField::Second => {
            let tod = floor_mod(ticks, tpd);
            (tod / ticks_per_second(unit)) % 60
        }
    }
}

/// Host reference for `DATE_TRUNC(unit, <Date32>)`. Returns the truncated day
/// count (still a `Date32` value).
pub fn date_trunc_date32_days(unit: DateTruncUnit, days: i32) -> i32 {
    let (y, m, _d) = civil_from_days(days as i64);
    let truncated = match unit {
        DateTruncUnit::Year => days_from_civil(y, 1, 1),
        DateTruncUnit::Month => days_from_civil(y, m, 1),
        // Day truncation on a bare date is a no-op.
        DateTruncUnit::Day => days as i64,
        // Sub-day units are rejected by the type-checker for Date32.
        DateTruncUnit::Hour | DateTruncUnit::Minute | DateTruncUnit::Second => days as i64,
    };
    truncated as i32
}

/// Host reference for `DATE_TRUNC(unit, <Timestamp>)`. Returns the truncated
/// tick count in the same `unit` resolution.
pub fn date_trunc_timestamp_ticks(unit: DateTruncUnit, ticks: i64, time_unit: TimeUnit) -> i64 {
    let tpd = ticks_per_day(time_unit);
    let tps = ticks_per_second(time_unit);
    match unit {
        DateTruncUnit::Year | DateTruncUnit::Month | DateTruncUnit::Day => {
            let days = floor_div(ticks, tpd);
            let trunc_days = match unit {
                DateTruncUnit::Day => days,
                _ => {
                    let (y, m, _d) = civil_from_days(days);
                    match unit {
                        DateTruncUnit::Year => days_from_civil(y, 1, 1),
                        DateTruncUnit::Month => days_from_civil(y, m, 1),
                        _ => unreachable!(),
                    }
                }
            };
            trunc_days * tpd
        }
        DateTruncUnit::Hour => {
            let bucket = tps * 3600;
            floor_div(ticks, bucket) * bucket
        }
        DateTruncUnit::Minute => {
            let bucket = tps * 60;
            floor_div(ticks, bucket) * bucket
        }
        DateTruncUnit::Second => floor_div(ticks, tps) * tps,
    }
}

// ===========================================================================
// PTX emitters.
//
// The kernels are element-wise: one thread per row, bail when `tid >= n_rows`.
// All arithmetic is in s64 registers (the input is sign-extended from s32 for
// Date32). The civil decomposition mirrors `civil_from_days` instruction for
// instruction; the intra-day fields use modular arithmetic.
//
// PTX `div.s64` / `rem.s64` truncate toward zero, matching Rust's `/` and `%`.
// For the calendar math the day count fed to the decomposition is first
// floored (via the emitted `floor_div` sequence) so the truncating ops below
// operate on the already-non-negative shifted day number `z`, where truncation
// and flooring agree.
// ===========================================================================

/// PTX entry-point name for an `EXTRACT` kernel over `(field, src)`.
pub fn extract_entry(field: DateField, src: DataType) -> String {
    format!(
        "bolt_extract_{}_{}",
        field.sql_name().to_ascii_lowercase(),
        dtype_tag(src)
    )
}

/// PTX entry-point name for a `DATE_TRUNC` kernel over `(unit, src)`.
pub fn date_trunc_entry(unit: DateTruncUnit, src: DataType) -> String {
    format!("bolt_date_trunc_{}_{}", unit.sql_name(), dtype_tag(src))
}

/// Short content-addressed tag for a temporal source dtype.
fn dtype_tag(src: DataType) -> &'static str {
    match src {
        DataType::Date32 => "date32",
        DataType::Timestamp(_, _) => "ts",
        _ => "other",
    }
}

/// Emit the common PTX header + entry signature + `tid` setup, returning the
/// builder with the input base pointer in `%rd0` (globalized) and the loaded
/// per-row source value in `%rd1` (sign-extended to s64). On out-of-range
/// `tid` the kernel has already branched to `DONE`.
///
/// `input_elem_bytes` is 4 for Date32 (loaded `s32` then `cvt.s64.s32`) or 8
/// for Timestamp (loaded `s64`).
fn emit_prologue(entry: &str, input_elem_bytes: usize) -> BoltResult<String> {
    let mut ptx = String::new();
    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {}(", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_2", entry).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<8>;").map_err(write_err)?;
    // Generous s64 scratch pool — the civil decomposition uses ~20 temporaries.
    writeln!(ptx, "\t.reg .b64   %rd<48>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // tid = ctaid.x * ntid.x + tid.x; bail if tid >= n_rows.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{}_param_2];", entry).map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Load the source value into %rd1 (s64).
    writeln!(ptx, "\tld.param.u64 %rd0, [{}_param_0];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd2, %r3, {bytes};",
        bytes = input_elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd3, %rd0, %rd2;").map_err(write_err)?;
    if input_elem_bytes == 4 {
        writeln!(ptx, "\tld.global.nc.s32 %r5, [%rd3];").map_err(write_err)?;
        writeln!(ptx, "\tcvt.s64.s32 %rd1, %r5;").map_err(write_err)?;
    } else {
        writeln!(ptx, "\tld.global.nc.s64 %rd1, [%rd3];").map_err(write_err)?;
    }
    Ok(ptx)
}

/// Emit the store of an s64 result (`src_reg`) to `output[tid]` and the kernel
/// epilogue (`DONE:` / `ret;` / closing brace). `out_elem_bytes` is 8 for the
/// `EXTRACT` Int64 output and the Timestamp `DATE_TRUNC` output, 4 for the
/// Date32 `DATE_TRUNC` output (where the s64 result is narrowed to s32).
fn emit_store_epilogue(
    ptx: &mut String,
    entry: &str,
    src_reg: &str,
    out_elem_bytes: usize,
) -> BoltResult<()> {
    writeln!(ptx, "\tld.param.u64 %rd40, [{}_param_1];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd40, %rd40;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd41, %r3, {bytes};",
        bytes = out_elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd42, %rd40, %rd41;").map_err(write_err)?;
    if out_elem_bytes == 4 {
        // Narrow the s64 result to s32 for a Date32 output column.
        writeln!(ptx, "\tcvt.s32.s64 %r6, {};", src_reg).map_err(write_err)?;
        writeln!(ptx, "\tst.global.s32 [%rd42], %r6;").map_err(write_err)?;
    } else {
        writeln!(ptx, "\tst.global.s64 [%rd42], {};", src_reg).map_err(write_err)?;
    }
    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;
    Ok(())
}

/// Emit a branch-free `floor_div` of `%rd_a` by the immediate `b` (b > 0),
/// leaving the quotient in `dst`. Uses scratch `%rd44`/`%rd45`/`%rd46` and
/// predicate `%p3`. For `b > 0` the correction is `q - 1` when the remainder
/// is non-zero and negative.
fn emit_floor_div_imm(ptx: &mut String, dst: &str, a: &str, b: i64) -> BoltResult<()> {
    writeln!(ptx, "\tdiv.s64 %rd44, {a}, {b};", a = a, b = b).map_err(write_err)?;
    writeln!(ptx, "\trem.s64 %rd45, {a}, {b};", a = a, b = b).map_err(write_err)?;
    // need_fix = (rem != 0) && (rem < 0)   (b is a positive immediate)
    writeln!(ptx, "\tsetp.lt.s64 %p3, %rd45, 0;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s64 %rd46, %rd44, 1;").map_err(write_err)?;
    writeln!(ptx, "\tselp.b64 {dst}, %rd46, %rd44, %p3;", dst = dst).map_err(write_err)?;
    Ok(())
}

/// Emit the Hinnant `civil_from_days` decomposition. On entry `%rd1` holds the
/// (already sign-extended) day count. On exit `%rd10` = year, `%rd11` = month,
/// `%rd12` = day. Clobbers the `%rd2x`..`%rd3x` scratch range and `%p3`.
fn emit_civil_from_days(ptx: &mut String) -> BoltResult<()> {
    // z = days + 719468
    writeln!(ptx, "\tadd.s64 %rd20, %rd1, 719468;").map_err(write_err)?;
    // era = floor_div(z, 146097)   (z may be negative for very old dates)
    emit_floor_div_imm(ptx, "%rd21", "%rd20", 146_097)?;
    // doe = z - era * 146097   (always in [0, 146096])
    writeln!(ptx, "\tmul.lo.s64 %rd22, %rd21, 146097;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s64 %rd23, %rd20, %rd22;").map_err(write_err)?; // %rd23 = doe
                                                                         // yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365
    writeln!(ptx, "\tdiv.s64 %rd24, %rd23, 1460;").map_err(write_err)?;
    writeln!(ptx, "\tdiv.s64 %rd25, %rd23, 36524;").map_err(write_err)?;
    writeln!(ptx, "\tdiv.s64 %rd26, %rd23, 146096;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s64 %rd27, %rd23, %rd24;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd27, %rd27, %rd25;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s64 %rd27, %rd27, %rd26;").map_err(write_err)?;
    writeln!(ptx, "\tdiv.s64 %rd28, %rd27, 365;").map_err(write_err)?; // %rd28 = yoe
                                                                       // y = yoe + era*400
    writeln!(ptx, "\tmul.lo.s64 %rd29, %rd21, 400;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd30, %rd28, %rd29;").map_err(write_err)?; // %rd30 = y
                                                                         // doy = doe - (365*yoe + yoe/4 - yoe/100)
    writeln!(ptx, "\tmul.lo.s64 %rd31, %rd28, 365;").map_err(write_err)?;
    writeln!(ptx, "\tdiv.s64 %rd32, %rd28, 4;").map_err(write_err)?;
    writeln!(ptx, "\tdiv.s64 %rd33, %rd28, 100;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd31, %rd31, %rd32;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s64 %rd31, %rd31, %rd33;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s64 %rd34, %rd23, %rd31;").map_err(write_err)?; // %rd34 = doy
                                                                        // mp = (5*doy + 2) / 153
    writeln!(ptx, "\tmul.lo.s64 %rd35, %rd34, 5;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd35, %rd35, 2;").map_err(write_err)?;
    writeln!(ptx, "\tdiv.s64 %rd36, %rd35, 153;").map_err(write_err)?; // %rd36 = mp
                                                                       // d = doy - (153*mp + 2)/5 + 1
    writeln!(ptx, "\tmul.lo.s64 %rd37, %rd36, 153;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd37, %rd37, 2;").map_err(write_err)?;
    writeln!(ptx, "\tdiv.s64 %rd37, %rd37, 5;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s64 %rd12, %rd34, %rd37;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd12, 1;").map_err(write_err)?; // %rd12 = day
                                                                     // m = mp < 10 ? mp + 3 : mp - 9
    writeln!(ptx, "\tsetp.lt.s64 %p3, %rd36, 10;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd38, %rd36, 3;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s64 %rd39, %rd36, 9;").map_err(write_err)?;
    writeln!(ptx, "\tselp.b64 %rd11, %rd38, %rd39, %p3;").map_err(write_err)?; // %rd11 = month
                                                                              // year = m <= 2 ? y + 1 : y
    writeln!(ptx, "\tsetp.le.s64 %p3, %rd11, 2;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd10, %rd30, 1;").map_err(write_err)?;
    writeln!(ptx, "\tselp.b64 %rd10, %rd10, %rd30, %p3;").map_err(write_err)?; // %rd10 = year
    Ok(())
}

/// Emit the Hinnant `days_from_civil` recomposition. On entry `%rd10`=year,
/// `%rd11`=month, `%rd12`=day. On exit `%rd13` = days since epoch. Clobbers the
/// `%rd2x` scratch range and `%p3`.
fn emit_days_from_civil(ptx: &mut String) -> BoltResult<()> {
    // y = month <= 2 ? year - 1 : year
    writeln!(ptx, "\tsetp.le.s64 %p3, %rd11, 2;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s64 %rd20, %rd10, 1;").map_err(write_err)?;
    writeln!(ptx, "\tselp.b64 %rd20, %rd20, %rd10, %p3;").map_err(write_err)?; // %rd20 = y
                                                                              // era = floor_div(y, 400)
    emit_floor_div_imm(ptx, "%rd21", "%rd20", 400)?;
    // yoe = y - era*400
    writeln!(ptx, "\tmul.lo.s64 %rd22, %rd21, 400;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s64 %rd23, %rd20, %rd22;").map_err(write_err)?; // %rd23 = yoe
                                                                        // mp = month > 2 ? month - 3 : month + 9
    writeln!(ptx, "\tsetp.gt.s64 %p3, %rd11, 2;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s64 %rd24, %rd11, 3;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd25, %rd11, 9;").map_err(write_err)?;
    writeln!(ptx, "\tselp.b64 %rd26, %rd24, %rd25, %p3;").map_err(write_err)?; // %rd26 = mp
                                                                              // doy = (153*mp + 2)/5 + day - 1
    writeln!(ptx, "\tmul.lo.s64 %rd27, %rd26, 153;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd27, %rd27, 2;").map_err(write_err)?;
    writeln!(ptx, "\tdiv.s64 %rd27, %rd27, 5;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd27, %rd27, %rd12;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s64 %rd27, %rd27, 1;").map_err(write_err)?; // %rd27 = doy
                                                                     // doe = yoe*365 + yoe/4 - yoe/100 + doy
    writeln!(ptx, "\tmul.lo.s64 %rd28, %rd23, 365;").map_err(write_err)?;
    writeln!(ptx, "\tdiv.s64 %rd29, %rd23, 4;").map_err(write_err)?;
    writeln!(ptx, "\tdiv.s64 %rd30, %rd23, 100;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd28, %rd28, %rd29;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s64 %rd28, %rd28, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd28, %rd28, %rd27;").map_err(write_err)?; // %rd28 = doe
                                                                        // days = era*146097 + doe - 719468
    writeln!(ptx, "\tmul.lo.s64 %rd31, %rd21, 146097;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd13, %rd31, %rd28;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s64 %rd13, %rd13, 719468;").map_err(write_err)?; // %rd13 = days
    Ok(())
}

/// Compile the PTX for an `EXTRACT(field FROM src)` element-wise kernel.
///
/// `src` must be `Date32` or `Timestamp`. The output column is always `Int64`.
/// Intra-day fields require a `Timestamp` source (the type-checker enforces
/// this; we error here defensively for a hand-built call).
pub fn compile_extract_kernel(field: DateField, src: DataType) -> BoltResult<String> {
    let entry = extract_entry(field, src);
    let (input_bytes, time_unit) = match src {
        DataType::Date32 => {
            if field.is_intraday() {
                return Err(BoltError::Type(format!(
                    "date_scalar: EXTRACT({} FROM Date32) is undefined",
                    field.sql_name()
                )));
            }
            (4usize, None)
        }
        DataType::Timestamp(tu, _) => (8usize, Some(tu)),
        other => {
            return Err(BoltError::Type(format!(
                "date_scalar: EXTRACT requires Date32/Timestamp, got {:?}",
                other
            )))
        }
    };

    let mut ptx = emit_prologue(&entry, input_bytes)?;

    // For a Timestamp source, the calendar fields need the day count
    // (floor_div(ticks, ticks_per_day)); the clock fields need ticks-of-day.
    // For a Date32 source %rd1 is already the day count.
    if field.is_intraday() {
        let tu = time_unit.expect("intra-day field implies Timestamp");
        let tpd = ticks_per_day(tu);
        let tps = ticks_per_second(tu);
        // tod = floor_mod(ticks, tpd) = ticks - floor_div(ticks, tpd)*tpd
        emit_floor_div_imm(&mut ptx, "%rd5", "%rd1", tpd)?;
        writeln!(ptx, "\tmul.lo.s64 %rd6, %rd5, {tpd};", tpd = tpd).map_err(write_err)?;
        writeln!(ptx, "\tsub.s64 %rd7, %rd1, %rd6;").map_err(write_err)?; // %rd7 = tod
                                                                         // seconds-of-day = tod / tps
        writeln!(ptx, "\tdiv.s64 %rd8, %rd7, {tps};", tps = tps).map_err(write_err)?;
        let result = match field {
            DateField::Hour => {
                // hour = sod / 3600
                writeln!(ptx, "\tdiv.s64 %rd9, %rd8, 3600;").map_err(write_err)?;
                "%rd9"
            }
            DateField::Minute => {
                // minute = (sod / 60) % 60
                writeln!(ptx, "\tdiv.s64 %rd9, %rd8, 60;").map_err(write_err)?;
                writeln!(ptx, "\trem.s64 %rd9, %rd9, 60;").map_err(write_err)?;
                "%rd9"
            }
            DateField::Second => {
                // second = sod % 60
                writeln!(ptx, "\trem.s64 %rd9, %rd8, 60;").map_err(write_err)?;
                "%rd9"
            }
            _ => unreachable!("is_intraday gated this branch"),
        };
        emit_store_epilogue(&mut ptx, &entry, result, 8)?;
        return Ok(ptx);
    }

    // Calendar field. For a Timestamp, reduce ticks → days first.
    if let Some(tu) = time_unit {
        let tpd = ticks_per_day(tu);
        emit_floor_div_imm(&mut ptx, "%rd1", "%rd1", tpd)?; // %rd1 = day count
    }
    emit_civil_from_days(&mut ptx)?;
    let result = match field {
        DateField::Year => "%rd10",
        DateField::Month => "%rd11",
        DateField::Day => "%rd12",
        _ => unreachable!("intra-day handled above"),
    };
    emit_store_epilogue(&mut ptx, &entry, result, 8)?;
    Ok(ptx)
}

/// Compile the PTX for a `DATE_TRUNC(unit, src)` element-wise kernel.
///
/// The output column has the **same dtype** as the input (`Date32` → `Date32`,
/// `Timestamp` → the same Timestamp type). Sub-day units require a `Timestamp`
/// source.
pub fn compile_date_trunc_kernel(unit: DateTruncUnit, src: DataType) -> BoltResult<String> {
    let entry = date_trunc_entry(unit, src);
    match src {
        DataType::Date32 => {
            if unit.is_intraday() {
                return Err(BoltError::Type(format!(
                    "date_scalar: DATE_TRUNC('{}', Date32) is undefined",
                    unit.sql_name()
                )));
            }
            let mut ptx = emit_prologue(&entry, 4)?;
            // %rd1 holds the day count. Day-trunc is a no-op; year/month go
            // through civil decomposition + recomposition.
            let result = match unit {
                DateTruncUnit::Day => "%rd1",
                DateTruncUnit::Year | DateTruncUnit::Month => {
                    emit_civil_from_days(&mut ptx)?;
                    // %rd10=year, %rd11=month, %rd12=day. Set the truncated
                    // fields to 1 and rebuild the day count.
                    writeln!(ptx, "\tmov.u64 %rd12, 1;").map_err(write_err)?; // day = 1
                    if matches!(unit, DateTruncUnit::Year) {
                        writeln!(ptx, "\tmov.u64 %rd11, 1;").map_err(write_err)?; // month = 1
                    }
                    emit_days_from_civil(&mut ptx)?;
                    "%rd13"
                }
                _ => unreachable!("intra-day rejected for Date32"),
            };
            emit_store_epilogue(&mut ptx, &entry, result, 4)?;
            Ok(ptx)
        }
        DataType::Timestamp(tu, _) => {
            let tpd = ticks_per_day(tu);
            let tps = ticks_per_second(tu);
            let mut ptx = emit_prologue(&entry, 8)?;
            let result = match unit {
                DateTruncUnit::Year | DateTruncUnit::Month | DateTruncUnit::Day => {
                    // days = floor_div(ticks, tpd)
                    emit_floor_div_imm(&mut ptx, "%rd1", "%rd1", tpd)?;
                    let trunc_days_reg = if matches!(unit, DateTruncUnit::Day) {
                        "%rd1"
                    } else {
                        emit_civil_from_days(&mut ptx)?;
                        writeln!(ptx, "\tmov.u64 %rd12, 1;").map_err(write_err)?; // day = 1
                        if matches!(unit, DateTruncUnit::Year) {
                            writeln!(ptx, "\tmov.u64 %rd11, 1;").map_err(write_err)?; // month = 1
                        }
                        emit_days_from_civil(&mut ptx)?;
                        "%rd13"
                    };
                    // ticks = trunc_days * tpd
                    writeln!(
                        ptx,
                        "\tmul.lo.s64 %rd15, {reg}, {tpd};",
                        reg = trunc_days_reg,
                        tpd = tpd
                    )
                    .map_err(write_err)?;
                    "%rd15"
                }
                DateTruncUnit::Hour | DateTruncUnit::Minute | DateTruncUnit::Second => {
                    let bucket = match unit {
                        DateTruncUnit::Hour => tps * 3600,
                        DateTruncUnit::Minute => tps * 60,
                        DateTruncUnit::Second => tps,
                        _ => unreachable!(),
                    };
                    // floor_div(ticks, bucket) * bucket
                    emit_floor_div_imm(&mut ptx, "%rd14", "%rd1", bucket)?;
                    writeln!(
                        ptx,
                        "\tmul.lo.s64 %rd15, %rd14, {bucket};",
                        bucket = bucket
                    )
                    .map_err(write_err)?;
                    "%rd15"
                }
            };
            emit_store_epilogue(&mut ptx, &entry, result, 8)?;
            Ok(ptx)
        }
        other => Err(BoltError::Type(format!(
            "date_scalar: DATE_TRUNC requires Date32/Timestamp, got {:?}",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- Host reference math -----

    /// `civil_from_days(0)` is the Unix epoch, 1970-01-01.
    #[test]
    fn civil_epoch() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
    }

    /// A handful of known civil dates (cross-checked against the Gregorian
    /// calendar), including a leap day and pre-epoch dates.
    #[test]
    fn civil_known_dates() {
        // 2000-01-01 is 10957 days after the epoch.
        assert_eq!(civil_from_days(10957), (2000, 1, 1));
        // 2000-02-29 (leap day) is 10957 + 31 + 28 = 11016.
        assert_eq!(civil_from_days(11016), (2000, 2, 29));
        // 2021-03-01.
        assert_eq!(civil_from_days(18687), (2021, 3, 1));
        // Pre-epoch: 1969-12-31 is -1.
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        // 1900-01-01 is -25567.
        assert_eq!(civil_from_days(-25567), (1900, 1, 1));
    }

    /// `days_from_civil` round-trips `civil_from_days` across a wide range,
    /// including pre-epoch negatives and leap days.
    #[test]
    fn civil_roundtrip() {
        for days in [-100_000i64, -25_567, -1, 0, 1, 10_957, 11_016, 18_687, 100_000] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days, "roundtrip failed at {days}");
        }
    }

    /// EXTRACT over Date32 pulls the right calendar field.
    #[test]
    fn extract_date32_fields() {
        // 2021-03-15 → day 18701.
        let days = days_from_civil(2021, 3, 15) as i32;
        assert_eq!(extract_date32(DateField::Year, days), 2021);
        assert_eq!(extract_date32(DateField::Month, days), 3);
        assert_eq!(extract_date32(DateField::Day, days), 15);
    }

    /// EXTRACT over Timestamp pulls both calendar and clock fields.
    #[test]
    fn extract_timestamp_fields() {
        // 2021-03-15 13:45:30 UTC in seconds since epoch.
        let days = days_from_civil(2021, 3, 15);
        let secs = days * 86_400 + 13 * 3600 + 45 * 60 + 30;
        let u = TimeUnit::Second;
        assert_eq!(extract_timestamp(DateField::Year, secs, u), 2021);
        assert_eq!(extract_timestamp(DateField::Month, secs, u), 3);
        assert_eq!(extract_timestamp(DateField::Day, secs, u), 15);
        assert_eq!(extract_timestamp(DateField::Hour, secs, u), 13);
        assert_eq!(extract_timestamp(DateField::Minute, secs, u), 45);
        assert_eq!(extract_timestamp(DateField::Second, secs, u), 30);

        // Same instant in milliseconds must agree on every field.
        let millis = secs * 1_000 + 123;
        let m = TimeUnit::Millisecond;
        assert_eq!(extract_timestamp(DateField::Hour, millis, m), 13);
        assert_eq!(extract_timestamp(DateField::Minute, millis, m), 45);
        assert_eq!(extract_timestamp(DateField::Second, millis, m), 30);
    }

    /// DATE_TRUNC over Date32 rounds down to the unit boundary.
    #[test]
    fn date_trunc_date32_boundaries() {
        let days = days_from_civil(2021, 3, 15) as i32;
        assert_eq!(
            date_trunc_date32_days(DateTruncUnit::Year, days),
            days_from_civil(2021, 1, 1) as i32
        );
        assert_eq!(
            date_trunc_date32_days(DateTruncUnit::Month, days),
            days_from_civil(2021, 3, 1) as i32
        );
        assert_eq!(date_trunc_date32_days(DateTruncUnit::Day, days), days);
    }

    /// DATE_TRUNC over Timestamp rounds down to the unit boundary on the tick
    /// count, preserving the resolution.
    #[test]
    fn date_trunc_timestamp_boundaries() {
        let days = days_from_civil(2021, 3, 15);
        let secs = days * 86_400 + 13 * 3600 + 45 * 60 + 30;
        let u = TimeUnit::Second;
        assert_eq!(
            date_trunc_timestamp_ticks(DateTruncUnit::Year, secs, u),
            days_from_civil(2021, 1, 1) * 86_400
        );
        assert_eq!(
            date_trunc_timestamp_ticks(DateTruncUnit::Month, secs, u),
            days_from_civil(2021, 3, 1) * 86_400
        );
        assert_eq!(
            date_trunc_timestamp_ticks(DateTruncUnit::Day, secs, u),
            days * 86_400
        );
        assert_eq!(
            date_trunc_timestamp_ticks(DateTruncUnit::Hour, secs, u),
            days * 86_400 + 13 * 3600
        );
        assert_eq!(
            date_trunc_timestamp_ticks(DateTruncUnit::Minute, secs, u),
            days * 86_400 + 13 * 3600 + 45 * 60
        );
        assert_eq!(
            date_trunc_timestamp_ticks(DateTruncUnit::Second, secs, u),
            secs
        );
    }

    /// Pre-epoch DATE_TRUNC must floor toward the past, not toward zero.
    #[test]
    fn date_trunc_pre_epoch_floors() {
        // 1969-12-31 12:00:00 → truncate to day should be 1969-12-31 00:00:00,
        // i.e. day -1 * 86400, NOT 0.
        let days = days_from_civil(1969, 12, 31); // -1
        let secs = days * 86_400 + 12 * 3600;
        assert_eq!(
            date_trunc_timestamp_ticks(DateTruncUnit::Day, secs, TimeUnit::Second),
            days * 86_400
        );
    }

    // ----- PTX-assertion tests -----

    /// EXTRACT(YEAR FROM Date32) emits a single entry with the 3-param ABI and
    /// the civil-decomposition divides.
    #[test]
    fn extract_year_date32_ptx_shape() {
        let ptx = compile_extract_kernel(DateField::Year, DataType::Date32).unwrap();
        let entry = extract_entry(DateField::Year, DataType::Date32);
        assert!(ptx.contains(&format!(".visible .entry {}(", entry)));
        assert!(ptx.contains("_param_0"));
        assert!(ptx.contains("_param_2"));
        // The +719468 epoch shift is the civil-decomposition fingerprint.
        assert!(ptx.contains("add.s64 %rd20, %rd1, 719468;"), "missing epoch shift");
        // Date32 input is sign-extended from s32.
        assert!(ptx.contains("cvt.s64.s32 %rd1, %r5;"), "missing s32->s64 widen");
        // EXTRACT output is Int64.
        assert!(ptx.contains("st.global.s64"), "EXTRACT must store s64");
        assert!(!ptx.contains("st.global.s32"), "EXTRACT must not store s32");
    }

    /// EXTRACT(HOUR FROM Timestamp) uses modular intra-day arithmetic, NOT the
    /// civil decomposition.
    #[test]
    fn extract_hour_timestamp_ptx_shape() {
        let src = DataType::Timestamp(TimeUnit::Second, None);
        let ptx = compile_extract_kernel(DateField::Hour, src).unwrap();
        // No epoch shift — intra-day path doesn't decompose the calendar date.
        assert!(
            !ptx.contains("719468"),
            "HOUR extract must not run the civil decomposition"
        );
        // hour = sod / 3600.
        assert!(ptx.contains("div.s64 %rd9, %rd8, 3600;"), "missing hour divide");
        // 8-byte Timestamp load (no s32 widen).
        assert!(ptx.contains("ld.global.nc.s64 %rd1, [%rd3];"));
    }

    /// EXTRACT(HOUR FROM Date32) is a type error (no time-of-day component).
    #[test]
    fn extract_hour_date32_rejected() {
        let err = compile_extract_kernel(DateField::Hour, DataType::Date32).unwrap_err();
        assert!(matches!(err, BoltError::Type(_)));
    }

    /// DATE_TRUNC(year, Date32) preserves the Date32 (s32) output dtype and
    /// runs both decomposition and recomposition (year/month set to 1).
    #[test]
    fn date_trunc_year_date32_ptx_shape() {
        let ptx = compile_date_trunc_kernel(DateTruncUnit::Year, DataType::Date32).unwrap();
        // Output narrows back to s32 for the Date32 column.
        assert!(ptx.contains("st.global.s32"), "Date32 DATE_TRUNC stores s32");
        assert!(!ptx.contains("st.global.s64"), "Date32 DATE_TRUNC must not store s64");
        // Both directions of the civil math appear (decompose + recompose).
        assert!(ptx.contains("add.s64 %rd20, %rd1, 719468;"), "missing decompose");
        assert!(ptx.contains("sub.s64 %rd13, %rd13, 719468;"), "missing recompose");
        // Year truncation sets month and day to 1.
        assert!(ptx.contains("mov.u64 %rd11, 1;"), "year-trunc sets month=1");
        assert!(ptx.contains("mov.u64 %rd12, 1;"), "year-trunc sets day=1");
    }

    /// DATE_TRUNC(hour, Timestamp) is a pure bucket floor — no civil math.
    #[test]
    fn date_trunc_hour_timestamp_ptx_shape() {
        let src = DataType::Timestamp(TimeUnit::Millisecond, None);
        let ptx = compile_date_trunc_kernel(DateTruncUnit::Hour, src).unwrap();
        assert!(
            !ptx.contains("719468"),
            "hour DATE_TRUNC must not run the civil decomposition"
        );
        // Output is the same Timestamp (s64) dtype.
        assert!(ptx.contains("st.global.s64"));
        // bucket = ticks_per_second(ms) * 3600 = 1000 * 3600 = 3_600_000.
        assert!(
            ptx.contains("3600000"),
            "hour bucket for ms resolution should be 3_600_000"
        );
    }

    /// DATE_TRUNC(hour, Date32) is rejected — a date has no sub-day component.
    #[test]
    fn date_trunc_hour_date32_rejected() {
        let err =
            compile_date_trunc_kernel(DateTruncUnit::Hour, DataType::Date32).unwrap_err();
        assert!(matches!(err, BoltError::Type(_)));
    }

    /// Entry names are content-addressed by (field/unit, dtype) so the module
    /// cache keys never alias across shapes.
    #[test]
    fn entry_names_are_distinct() {
        let a = extract_entry(DateField::Year, DataType::Date32);
        let b = extract_entry(DateField::Year, DataType::Timestamp(TimeUnit::Second, None));
        let c = extract_entry(DateField::Month, DataType::Date32);
        assert_ne!(a, b);
        assert_ne!(a, c);
        let d = date_trunc_entry(DateTruncUnit::Year, DataType::Date32);
        let e = date_trunc_entry(DateTruncUnit::Month, DataType::Date32);
        assert_ne!(d, e);
        assert_ne!(a, d);
    }
}
