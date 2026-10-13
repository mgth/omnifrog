//! A model summed up over zones.
//!
//! For each stem, the mean of what was measured over the zones in scope;
//! beside it, how often the ear made the output a reference or rejected it,
//! and why. A reject counts in the means like any other: the rating keeps an
//! output out of the reference, not out of the table.
//!
//! The same the other way round, [`by_zone`]: one model, and what it did on
//! each zone — where its mean comes from.

use crate::keys;
use crate::set::{Metric, Reason, Set, Side, Status, Stem, Zone};
use std::cmp::Ordering;
use std::collections::BTreeMap;

/// The means of one stem. A value nothing measured is none.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Mean {
    pub si_sdr: Option<f64>,
    pub residue: Option<f64>,
    pub loss: Option<f64>,
    pub artifacts: Option<f64>,
    pub gap: Option<f64>,
    pub excess: Option<f64>,
    pub lack: Option<f64>,
    /// Zones the means are over.
    pub n: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    pub key: String,
    pub stems: BTreeMap<Stem, Mean>,
    /// Zones of the scope the output has a measure on.
    pub zones: usize,
    /// Zones where it is a reference, in the order of [`Side::BOTH`];
    /// measured or not.
    pub refs: [usize; 2],
    /// Zones where it is rejected.
    pub rejects: [usize; 2],
    /// Of those, for each reason of [`Reason::ALL`].
    pub reasons: [[usize; 3]; 2],
    /// Mean time its outputs took on a zone, when known.
    pub seconds: Option<f64>,
}

impl Report {
    pub fn mean(&self, stem: Stem) -> Mean {
        self.stems.get(&stem).copied().unwrap_or_default()
    }

    /// Nothing to show on this side but ratings: the output is the only
    /// reference wherever it was measured, so it was never compared.
    pub fn only_reference(&self, side: Side) -> bool {
        self.mean(side.listened()).n == 0 && self.refs[side as usize] > 0
    }
}

fn mean_of(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let (sum, n) = values
        .flatten()
        .fold((0.0, 0usize), |(sum, n), value| (sum + value, n + 1));
    (n > 0).then(|| sum / n as f64)
}

/// The report of one key over the given zones.
pub fn of(set: &Set, zones: &[&Zone], key: &str) -> Report {
    let mut report = Report {
        key: key.to_string(),
        ..Report::default()
    };
    let mut by_stem: BTreeMap<Stem, Vec<&Metric>> = BTreeMap::new();
    let (mut seconds, mut timed) = (0.0, 0usize);
    for zone in zones {
        // The ratings count before the measures: a zone rated and not yet
        // measured counts too.
        for side in Side::BOTH {
            match zone.rating(side).status(key) {
                Status::Reference => report.refs[side as usize] += 1,
                Status::Rejected(reason) => {
                    report.rejects[side as usize] += 1;
                    if let Some(place) =
                        reason.and_then(|r| Reason::ALL.iter().position(|x| *x == r))
                    {
                        report.reasons[side as usize][place] += 1;
                    }
                }
                Status::Neutral => {}
            }
        }
        let Some(measure) = set.measure(&zone.id, key) else {
            continue;
        };
        // The time stands whether or not the measure does.
        if measure.seconds > 0.0 {
            seconds += measure.seconds;
            timed += 1;
        }
        if !measure.measured() {
            continue;
        }
        report.zones += 1;
        for (stem, metric) in &measure.stems {
            by_stem.entry(*stem).or_default().push(metric);
        }
    }
    for stem in Stem::ALL {
        let metrics = by_stem.remove(&stem).unwrap_or_default();
        let n = metrics.iter().filter(|m| m.si_sdr.is_some()).count();
        let mean = if n == 0 {
            Mean::default()
        } else {
            Mean {
                si_sdr: mean_of(metrics.iter().map(|m| m.si_sdr)),
                residue: mean_of(metrics.iter().map(|m| m.residue)),
                loss: mean_of(metrics.iter().map(|m| m.loss)),
                artifacts: mean_of(metrics.iter().map(|m| m.artifacts)),
                gap: mean_of(metrics.iter().map(|m| m.gap)),
                excess: mean_of(metrics.iter().map(|m| m.excess)),
                lack: mean_of(metrics.iter().map(|m| m.lack)),
                n,
            }
        };
        report.stems.insert(stem, mean);
    }
    report.seconds = (timed > 0).then(|| seconds / timed as f64);
    report
}

impl AsRef<Report> for Report {
    fn as_ref(&self) -> &Report {
        self
    }
}

/// A model on one zone: the report of that zone alone, so that its means
/// are what was measured there and its counts are one or none.
#[derive(Debug, Clone, PartialEq)]
pub struct ZoneReport {
    /// The zone's id.
    pub zone: String,
    pub report: Report,
}

impl AsRef<Report> for ZoneReport {
    fn as_ref(&self) -> &Report {
        &self.report
    }
}

/// One key zone by zone: the roles of [`table`] the other way round. Every
/// zone given has its row, in the order given, whether or not the key was
/// measured or rated there: what is missing shows as much as what is bad.
pub fn by_zone(set: &Set, zones: &[&Zone], key: &str) -> Vec<ZoneReport> {
    zones
        .iter()
        .map(|zone| ZoneReport {
            zone: zone.id.clone(),
            report: of(set, &[zone], key),
        })
        .collect()
}

/// Every key rated or measured over the zones, in the catalogue's order.
pub fn keys(set: &Set, zones: &[&Zone], order: &[String]) -> Vec<String> {
    let mut all: Vec<String> = Vec::new();
    for zone in zones {
        for side in Side::BOTH {
            let rating = zone.rating(side);
            all.extend(rating.refs.iter().chain(&rating.rejects).cloned());
        }
        all.extend(
            set.measures
                .get(&zone.id)
                .into_iter()
                .flatten()
                .map(|(k, _)| k.clone()),
        );
    }
    all.sort_by_cached_key(|key| {
        let (model, variant) = keys::split(key);
        // A cascade follows the model it starts with.
        let first = keys::stages(key).map_or(model, |(first, _, _)| first);
        (
            order.iter().position(|k| k == first).unwrap_or(usize::MAX),
            model.to_string(),
            variant.map_or(0, str::len),
        )
    });
    all.dedup();
    all
}

/// The reports of one side: every key rated or measured over the zones, in
/// the catalogue's order. A variant has nothing of its own on the
/// original's side, and is listed on the dub's once it has something there.
pub fn table(set: &Set, zones: &[&Zone], order: &[String], side: Side) -> Vec<Report> {
    keys(set, zones, order)
        .iter()
        .map(|key| of(set, zones, key))
        .filter(|report| {
            if !keys::is_variant(&report.key) {
                return true;
            }
            side == Side::Vf
                && (report.mean(Stem::VoiceVf).n > 0
                    || report.refs[Side::Vf as usize] + report.rejects[Side::Vf as usize] > 0)
        })
        .collect()
}

/// What a table of one side is sorted by. The metrics are those of the stem
/// that side is judged on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Column {
    /// The order the rows were made in: the catalogue's for [`table`], the
    /// zones' own for [`by_zone`].
    Model,
    SiSdr,
    Residue,
    Loss,
    Artifacts,
    Gap,
    Excess,
    Lack,
    /// The SI-SDR of the side's other stem.
    OtherSiSdr,
    Seconds,
    Refs,
    Rejects,
}

impl Column {
    pub fn label(self) -> &'static str {
        match self {
            Column::Model => "Model",
            Column::SiSdr => "SI-SDR",
            Column::Residue => "Residue",
            Column::Loss => "Loss",
            Column::Artifacts => "Artifacts",
            Column::Gap => "Gap",
            Column::Excess => "Excess",
            Column::Lack => "Lack",
            Column::OtherSiSdr => "Other",
            Column::Seconds => "Time",
            Column::Refs => "★",
            Column::Rejects => "✗",
        }
    }

    /// Whether the best is the highest: what a first click sorts first.
    pub fn higher_is_better(self) -> bool {
        matches!(self, Column::SiSdr | Column::OtherSiSdr | Column::Refs)
    }

    pub fn value(self, report: &Report, side: Side) -> Option<f64> {
        let mean = report.mean(side.listened());
        match self {
            Column::Model => None,
            Column::SiSdr => mean.si_sdr,
            Column::Residue => mean.residue,
            Column::Loss => mean.loss,
            Column::Artifacts => mean.artifacts,
            Column::Gap => mean.gap,
            Column::Excess => mean.excess,
            Column::Lack => mean.lack,
            Column::OtherSiSdr => report.mean(side.listened().other()).si_sdr,
            Column::Seconds => report.seconds,
            Column::Refs => Some(report.refs[side as usize] as f64),
            Column::Rejects => Some(report.rejects[side as usize] as f64),
        }
    }
}

/// Sort a table by a column. Rows with no value there go last either way,
/// and rows that tie keep the order they were in.
pub fn sort<R: AsRef<Report>>(reports: &mut [R], side: Side, column: Column, descending: bool) {
    if column == Column::Model {
        if descending {
            reports.reverse();
        }
        return;
    }
    reports.sort_by(|a, b| {
        match (
            column.value(a.as_ref(), side),
            column.value(b.as_ref(), side),
        ) {
            (Some(a), Some(b)) => {
                let order = a.partial_cmp(&b).unwrap_or(Ordering::Equal);
                if descending { order.reverse() } else { order }
            }
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        }
    });
}

/// The zones where a side has a single reference, with those sides. The
/// reference is then one model's output, and the measure rewards that
/// model's family rather than what the ear prefers; the cure is more
/// references, from other families.
pub fn single_reference<'a>(zones: &[&'a Zone]) -> Vec<(&'a Zone, Vec<Side>)> {
    zones
        .iter()
        .filter_map(|zone| {
            let sides: Vec<Side> = Side::BOTH
                .into_iter()
                .filter(|&side| zone.rating(side).refs.len() == 1)
                .collect();
            (!sides.is_empty()).then_some((*zone, sides))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::set::Measure;

    fn metric(si_sdr: Option<f64>, gap: Option<f64>) -> Metric {
        Metric {
            si_sdr,
            gap,
            ..Metric::default()
        }
    }

    /// Two zones. `a` is the only reference of the first on the original's
    /// side, `b` is rejected there for a leak; both are measured on the dub.
    fn set() -> Set {
        let mut set = Set::default();
        for id in ["one", "two"] {
            set.zones.push(Zone {
                id: id.into(),
                ..Zone::default()
            });
        }
        set.zones[0].vo.refs = vec!["a".into()];
        set.zones[0].vo.rejects = vec!["b".into()];
        set.zones[0].vo.reasons.insert("b".into(), Reason::Leak);
        set.zones[1].vo.refs = vec!["a".into(), "b".into()];
        set.zones[1].vf.rejects = vec!["b+cancel".into()];

        let mut put = |zone: &str, key: &str, seconds: f64, stems: &[(Stem, Metric)]| {
            let measure = Measure {
                seconds,
                date: String::new(),
                stems: stems.iter().cloned().collect(),
            };
            set.measures
                .entry(zone.into())
                .or_default()
                .insert(key.into(), measure);
        };
        let alone = Metric {
            reference: true,
            ..Metric::default()
        };
        put(
            "one",
            "a",
            0.0,
            &[
                (Stem::RestVo, alone),
                (Stem::VoiceVf, metric(Some(30.0), Some(1.0))),
            ],
        );
        put(
            "one",
            "b",
            12.0,
            &[
                (Stem::RestVo, metric(Some(10.0), Some(4.0))),
                (Stem::VoiceVf, metric(Some(20.0), None)),
            ],
        );
        put(
            "two",
            "a",
            0.0,
            &[(Stem::RestVo, metric(Some(26.0), Some(0.5)))],
        );
        put(
            "two",
            "b",
            8.0,
            &[(Stem::RestVo, metric(Some(20.0), Some(2.0)))],
        );
        set
    }

    #[test]
    fn a_report_is_the_mean_of_what_was_measured() {
        let set = set();
        let zones: Vec<&Zone> = set.zones.iter().collect();
        let b = of(&set, &zones, "b");
        assert_eq!(b.zones, 2);
        assert_eq!(b.seconds, Some(10.0));
        let rest = b.mean(Stem::RestVo);
        assert_eq!((rest.si_sdr, rest.gap, rest.n), (Some(15.0), Some(3.0), 2));
        // A value measured on one zone only is the mean of that one.
        let voice = b.mean(Stem::VoiceVf);
        assert_eq!((voice.si_sdr, voice.gap, voice.n), (Some(20.0), None, 1));
        assert_eq!((b.refs, b.rejects), ([1, 0], [1, 0]));
        assert_eq!(b.reasons[0], [1, 0, 0]);

        // The only reference of a zone has no number there.
        let a = of(&set, &zones, "a");
        assert_eq!(a.mean(Stem::RestVo).n, 1);
        assert_eq!(a.mean(Stem::RestVo).si_sdr, Some(26.0));
        assert_eq!(a.seconds, None);
        assert!(!a.only_reference(Side::Vo));
        let first = of(&set, &zones[..1], "a");
        assert!(first.only_reference(Side::Vo));
        assert_eq!(first.mean(Stem::RestVo), Mean::default());
    }

    #[test]
    fn a_variant_is_listed_on_the_dub_s_side_once_it_has_something_there() {
        let set = set();
        let zones: Vec<&Zone> = set.zones.iter().collect();
        let order = ["b".to_string(), "a".to_string()];
        let keys = |side| -> Vec<String> {
            table(&set, &zones, &order, side)
                .into_iter()
                .map(|r| r.key)
                .collect()
        };
        assert_eq!(keys(Side::Vo), ["b", "a"]);
        assert_eq!(keys(Side::Vf), ["b", "b+cancel", "a"]);
        // Out of the scope where it is rated, it is not listed.
        let first = &zones[..1];
        let listed: Vec<String> = table(&set, first, &order, Side::Vf)
            .into_iter()
            .map(|r| r.key)
            .collect();
        assert_eq!(listed, ["b", "a"]);
    }

    #[test]
    fn a_model_zone_by_zone_is_what_its_mean_is_made_of() {
        let mut set = set();
        // A third zone the model was never run on.
        set.zones.push(Zone {
            id: "three".into(),
            ..Zone::default()
        });
        let zones: Vec<&Zone> = set.zones.iter().collect();
        let rows = by_zone(&set, &zones, "b");
        let ids: Vec<&str> = rows.iter().map(|r| r.zone.as_str()).collect();
        assert_eq!(ids, ["one", "two", "three"]);
        let si_sdr = |row: &ZoneReport| row.report.mean(Stem::RestVo).si_sdr;
        assert_eq!(si_sdr(&rows[0]), Some(10.0));
        assert_eq!(si_sdr(&rows[1]), Some(20.0));
        // The mean of the rows is the model's line in the table.
        assert_eq!(of(&set, &zones, "b").mean(Stem::RestVo).si_sdr, Some(15.0));
        // The ratings and the time are the zone's own.
        assert_eq!(
            (rows[0].report.refs, rows[0].report.rejects),
            ([0, 0], [1, 0])
        );
        assert_eq!(rows[0].report.reasons[0], [1, 0, 0]);
        assert_eq!(
            (rows[1].report.refs, rows[1].report.rejects),
            ([1, 0], [0, 0])
        );
        assert_eq!(rows[0].report.seconds, Some(12.0));
        // The zone it was not run on has its row, with nothing in it.
        assert_eq!(rows[2].report.zones, 0);
        assert_eq!(rows[2].report.mean(Stem::RestVo), Mean::default());
        assert_eq!(rows[2].report.seconds, None);

        // Where it is the only reference there is no number, and it shows.
        let a = by_zone(&set, &zones, "a");
        assert!(a[0].report.only_reference(Side::Vo));
        assert!(!a[1].report.only_reference(Side::Vo));

        // Sorted, the zone with nothing goes last either way.
        let mut rows = by_zone(&set, &zones, "b");
        sort(&mut rows, Side::Vo, Column::SiSdr, true);
        let ids: Vec<&str> = rows.iter().map(|r| r.zone.as_str()).collect();
        assert_eq!(ids, ["two", "one", "three"]);
        sort(&mut rows, Side::Vo, Column::SiSdr, false);
        let ids: Vec<&str> = rows.iter().map(|r| r.zone.as_str()).collect();
        assert_eq!(ids, ["one", "two", "three"]);

        let order = ["b".to_string(), "a".to_string()];
        assert_eq!(keys(&set, &zones, &order), ["b", "b+cancel", "a"]);
    }

    #[test]
    fn a_sort_puts_what_has_no_value_last_either_way() {
        let set = set();
        let zones: Vec<&Zone> = set.zones.iter().collect();
        let order = ["b".to_string(), "a".to_string()];
        let mut reports = table(&set, &zones, &order, Side::Vf);
        let keys =
            |reports: &[Report]| -> Vec<String> { reports.iter().map(|r| r.key.clone()).collect() };
        sort(&mut reports, Side::Vf, Column::SiSdr, true);
        assert_eq!(keys(&reports), ["a", "b", "b+cancel"]);
        sort(&mut reports, Side::Vf, Column::SiSdr, false);
        assert_eq!(keys(&reports), ["b", "a", "b+cancel"]);
        sort(&mut reports, Side::Vf, Column::Rejects, true);
        assert_eq!(keys(&reports)[0], "b+cancel");
        assert!(Column::SiSdr.higher_is_better());
        assert!(!Column::Gap.higher_is_better());
    }

    #[test]
    fn a_side_with_a_single_reference_is_pointed_out() {
        let set = set();
        let zones: Vec<&Zone> = set.zones.iter().collect();
        let single = single_reference(&zones);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].0.id, "one");
        assert_eq!(single[0].1, [Side::Vo]);
    }
}
