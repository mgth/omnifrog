//! ROCm's idea of the AMD cards, against what the cards are.
//!
//! `HSA_OVERRIDE_GFX_VERSION` makes ROCm treat every card as the
//! architecture it names. Within a generation it is how a card ROCm ships
//! no kernels for gets to run: a gfx1031 run as a gfx1030 (`10.3.0`), a
//! gfx1102 as a gfx1100 (`11.0.0`). Across generations it hands the card
//! code for another instruction set: the first copy to the card faults, its
//! firmware stops answering and the driver resets the GPU, closing the
//! desktop's applications with it. A variable left over from an earlier
//! card does exactly that, so the worker is not started under one, and
//! `doctor` says why.
//!
//! What the cards are comes from the kernel, not from ROCm: the KFD
//! topology in sysfs (Linux only), where each GPU node has a
//! `gfx_target_version` — `120001` for a gfx1201, `0` for the processor.
//! Nothing here starts a program or opens a device.

use std::fmt;
use std::fs;
use std::path::Path;

/// The variable that makes ROCm run every card as another architecture.
pub const OVERRIDE: &str = "HSA_OVERRIDE_GFX_VERSION";
const ARCH: &str = "PYTORCH_ROCM_ARCH";

/// An AMD GPU architecture: gfx1201 is 12.0.1, gfx90a is 9.0.10.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gfx {
    pub major: u32,
    pub minor: u32,
    pub stepping: u32,
}

impl Gfx {
    /// From the topology's `gfx_target_version`, `major·10000 + minor·100
    /// + stepping`; `0`, the processor's node, is none.
    fn from_target_version(value: u64) -> Option<Self> {
        if value == 0 {
            return None;
        }
        Some(Self {
            major: u32::try_from(value / 10_000).ok()?,
            minor: (value / 100 % 100) as u32,
            stepping: (value % 100) as u32,
        })
    }

    /// From `HSA_OVERRIDE_GFX_VERSION`: `major.minor.stepping`, in decimal.
    fn from_override(value: &str) -> Option<Self> {
        let mut parts = value.trim().split('.').map(|p| p.parse::<u32>().ok());
        let gfx = Self {
            major: parts.next()??,
            minor: parts.next()??,
            stepping: parts.next()??,
        };
        parts.next().is_none().then_some(gfx)
    }

    /// From a name as `PYTORCH_ROCM_ARCH` lists it: `gfx1100`, `gfx90a`,
    /// `gfx90a:xnack-`. The last two digits are the minor and the stepping,
    /// in hexadecimal; the rest is the major.
    fn from_name(name: &str) -> Option<Self> {
        let digits = name.trim().split(':').next()?.strip_prefix("gfx")?;
        if digits.len() < 3 || !digits.is_ascii() {
            return None;
        }
        let (major, rest) = digits.split_at(digits.len() - 2);
        let (minor, stepping) = rest.split_at(1);
        Some(Self {
            major: major.parse().ok()?,
            minor: u32::from_str_radix(minor, 16).ok()?,
            stepping: u32::from_str_radix(stepping, 16).ok()?,
        })
    }

    /// Whether code built for `self` runs on a card that is an `other`: the
    /// same generation, which is what an override is for. gfx1201 code on
    /// a gfx1100 (or the reverse) is another instruction set.
    fn runs_on(self, other: Gfx) -> bool {
        self.major == other.major
    }
}

impl fmt::Display for Gfx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gfx{}{:x}{:x}", self.major, self.minor, self.stepping)
    }
}

/// A GPU node of the topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Card {
    /// The node's number in the topology.
    pub node: u32,
    pub gfx: Gfx,
    /// What ROCm makes the card's `GPU-…` id of; 0 when the card has none.
    pub unique_id: u64,
    /// The board's name, when the driver publishes one.
    pub name: Option<String>,
}

impl Card {
    fn describe(&self) -> String {
        match &self.name {
            Some(name) => format!("{} ({name})", self.gfx),
            None => self.gfx.to_string(),
        }
    }

    /// Whether `token` of a `*_VISIBLE_DEVICES` list is this card's
    /// `GPU-<hex>` id.
    fn has_uuid(&self, token: &str) -> bool {
        self.unique_id != 0
            && token
                .strip_prefix("GPU-")
                .or_else(|| token.strip_prefix("gpu-"))
                .and_then(|hex| u64::from_str_radix(hex, 16).ok())
                == Some(self.unique_id)
    }
}

/// The GPU nodes of the KFD topology under `sys` (`/sys` on a machine), in
/// the order ROCm numbers them; empty without the driver.
pub fn cards(sys: &Path) -> Vec<Card> {
    let nodes = sys.join("class/kfd/kfd/topology/nodes");
    let Ok(entries) = fs::read_dir(&nodes) else {
        return Vec::new();
    };
    let mut found: Vec<Card> = entries
        .flatten()
        .filter_map(|entry| {
            let node = entry.file_name().to_str()?.parse::<u32>().ok()?;
            let properties = fs::read_to_string(entry.path().join("properties")).ok()?;
            let gfx = Gfx::from_target_version(property(&properties, "gfx_target_version")?)?;
            let name = property(&properties, "drm_render_minor").and_then(|minor| {
                let path = sys.join(format!("class/drm/renderD{minor}/device/product_name"));
                fs::read_to_string(path)
                    .ok()
                    .map(|n| n.trim().to_string())
                    .filter(|n| !n.is_empty())
            });
            Some(Card {
                node,
                gfx,
                unique_id: property(&properties, "unique_id").unwrap_or(0),
                name,
            })
        })
        .collect();
    // The directory lists in no particular order, and 10 sorts before 2.
    found.sort_by_key(|card| card.node);
    found
}

/// One `key value` line of a node's `properties`.
fn property(properties: &str, key: &str) -> Option<u64> {
    properties.lines().find_map(|line| {
        let (k, v) = line.split_once(' ')?;
        (k == key).then(|| v.trim().parse().ok())?
    })
}

/// The cards ROCm shows torch: `ROCR_VISIBLE_DEVICES` picks among all of
/// them, then `HIP_VISIBLE_DEVICES` (or `CUDA_VISIBLE_DEVICES`, which HIP
/// reads in its absence) among those. The first is the one the separation
/// runs on.
fn visible<'a>(cards: &'a [Card], env: &dyn Fn(&str) -> Option<String>) -> Vec<&'a Card> {
    let set = |key: &str| env(key).filter(|v| !v.trim().is_empty());
    let mut shown: Vec<&Card> = cards.iter().collect();
    if let Some(list) = set("ROCR_VISIBLE_DEVICES") {
        shown = pick(&shown, &list);
    }
    if let Some(list) = set("HIP_VISIBLE_DEVICES").or_else(|| set("CUDA_VISIBLE_DEVICES")) {
        shown = pick(&shown, &list);
    }
    shown
}

/// The cards a `*_VISIBLE_DEVICES` list names, by index or by `GPU-…` id,
/// in its order. Like CUDA's, the list ends at the first entry that names
/// no card: `-1` hides them all.
fn pick<'a>(cards: &[&'a Card], list: &str) -> Vec<&'a Card> {
    let mut picked: Vec<&Card> = Vec::new();
    for token in list.split(',').map(str::trim) {
        let card = match token.parse::<usize>() {
            Ok(index) => cards.get(index).copied(),
            Err(_) => cards.iter().copied().find(|card| card.has_uuid(token)),
        };
        match card {
            Some(card) if !picked.contains(&card) => picked.push(card),
            Some(_) => {}
            None => break,
        }
    }
    picked
}

/// What the variables make of the cards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    /// The cards ROCm shows torch and what it runs them as:
    /// « gfx1201; hidden: gfx1036 ».
    pub summary: String,
    /// Why the worker must not start: the override hands the card the
    /// separation runs on code it cannot run.
    pub broken: Option<String>,
    /// What is off without endangering that card.
    pub warnings: Vec<String>,
}

impl Report {
    /// One line for `doctor`: why it is broken, else the summary, then the
    /// warnings.
    pub fn detail(&self) -> String {
        let mut parts = vec![self.broken.clone().unwrap_or_else(|| self.summary.clone())];
        parts.extend(self.warnings.iter().cloned());
        parts.join(" — ")
    }
}

/// The variables of this process against this machine's cards; `None`
/// where there is no AMD GPU to compare with, or off Linux.
pub fn check() -> Option<Report> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    check_in(Path::new("/sys"), &|key| std::env::var(key).ok())
}

/// [`check`] against the topology under `sys`, with `env` for the
/// variables.
pub fn check_in(sys: &Path, env: &dyn Fn(&str) -> Option<String>) -> Option<Report> {
    let cards = cards(sys);
    if cards.is_empty() {
        return None;
    }
    let shown = visible(&cards, env);
    let hidden: Vec<String> = cards
        .iter()
        .filter(|card| !shown.contains(card))
        .map(Card::describe)
        .collect();
    let mut summary = if shown.is_empty() {
        "no AMD card visible to ROCm".to_string()
    } else {
        shown
            .iter()
            .map(|card| card.describe())
            .collect::<Vec<_>>()
            .join(", ")
    };
    if !hidden.is_empty() {
        summary.push_str(&format!("; hidden: {}", hidden.join(", ")));
    }
    let mut report = Report {
        summary,
        broken: None,
        warnings: Vec::new(),
    };
    let Some(&first) = shown.first() else {
        return Some(report);
    };

    // What ROCm runs the first card as, when the override fits it.
    let mut runs_as: Option<Gfx> = None;
    if let Some(raw) = env(OVERRIDE).map(|v| v.trim().to_string())
        && !raw.is_empty()
    {
        match Gfx::from_override(&raw) {
            None => report.warnings.push(format!(
                "{OVERRIDE}={raw} is not a version ROCm reads (major.minor.stepping, e.g. {}.{}.{})",
                first.gfx.major, first.gfx.minor, first.gfx.stepping
            )),
            Some(over) if !over.runs_on(first.gfx) => {
                report.broken = Some(format!(
                    "{OVERRIDE}={raw} tells ROCm the card the separation runs on is a {over}, \
                     but it is a {}, of another generation: code for a {over} faults on it and \
                     the driver answers with a reset of the GPU that closes the desktop's \
                     applications. Unset {OVERRIDE} (likely left over from an earlier card, in \
                     ~/.profile, ~/.bashrc or ~/.zshrc) or set it to a {}.x.x version",
                    first.describe(),
                    first.gfx.major
                ));
            }
            Some(over) => {
                runs_as = Some(over);
                report.summary.push_str(&if over == first.gfx {
                    format!("; {OVERRIDE}={raw}, the card's own")
                } else {
                    format!("; run as {over} ({OVERRIDE}={raw})")
                });
                // The override is every card's: one of another generation
                // gets code it cannot run, although nothing is sent to it.
                for other in shown.iter().skip(1).filter(|c| !over.runs_on(c.gfx)) {
                    report.warnings.push(format!(
                        "{OVERRIDE}={raw} also makes the {} a {over}, which it cannot run: \
                         hide it with HIP_VISIBLE_DEVICES",
                        other.describe()
                    ));
                }
            }
        }
    }

    // Which architectures ROCm code built here is for: one the cards do
    // not have is a leftover too, harmless to the separation.
    if let Some(raw) = env(ARCH).map(|v| v.trim().to_string())
        && !raw.is_empty()
    {
        let fits = |gfx: Gfx| shown.iter().any(|card| card.gfx == gfx) || runs_as == Some(gfx);
        let named = raw
            .split([';', ',', ' '])
            .filter(|name| !name.is_empty())
            .any(|name| Gfx::from_name(name).is_some_and(fits));
        if !named {
            let have: Vec<String> = shown
                .iter()
                .map(|card| card.gfx.to_string())
                .chain(runs_as.map(|gfx| gfx.to_string()))
                .collect();
            report.warnings.push(format!(
                "{ARCH}={raw} names no architecture of the cards here ({}): what is compiled \
                 for ROCm under it would not run on them; likely left over from an earlier card",
                have.join(", ")
            ));
        }
    }
    Some(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// A topology under a temporary `sys`: one processor node, then one
    /// GPU node per `(gfx_target_version, unique_id, product name)`.
    struct Sys(PathBuf);

    impl Sys {
        fn new(name: &str, gpus: &[(u64, u64, Option<&str>)]) -> Self {
            let root =
                std::env::temp_dir().join(format!("omnifrog-kfd-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&root);
            let nodes = root.join("class/kfd/kfd/topology/nodes");
            let node = |n: usize, properties: String| {
                let dir = nodes.join(n.to_string());
                fs::create_dir_all(&dir).unwrap();
                fs::write(dir.join("properties"), properties).unwrap();
            };
            node(
                0,
                "cpu_cores_count 32\nsimd_count 0\ngfx_target_version 0\ndrm_render_minor 0\n"
                    .into(),
            );
            for (i, (version, unique_id, name)) in gpus.iter().enumerate() {
                let minor = 128 + i;
                node(
                    i + 1,
                    format!(
                        "cpu_cores_count 0\nsimd_count 128\ngfx_target_version {version}\n\
                         vendor_id 4098\ndrm_render_minor {minor}\nunique_id {unique_id}\n"
                    ),
                );
                if let Some(name) = name {
                    let device = root.join(format!("class/drm/renderD{minor}/device"));
                    fs::create_dir_all(&device).unwrap();
                    fs::write(device.join("product_name"), format!("{name}\n")).unwrap();
                }
            }
            Self(root)
        }

        fn check(&self, vars: &[(&str, &str)]) -> Option<Report> {
            let vars: HashMap<String, String> = vars
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            check_in(&self.0, &|key| vars.get(key).cloned())
        }
    }

    impl Drop for Sys {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    const RDNA4: u64 = 120001;
    const RDNA3: u64 = 110000;
    const IGPU: u64 = 100306;

    fn gfx(major: u32, minor: u32, stepping: u32) -> Gfx {
        Gfx {
            major,
            minor,
            stepping,
        }
    }

    #[test]
    fn architectures_read_from_the_topology_the_override_and_names() {
        assert_eq!(Gfx::from_target_version(120001), Some(gfx(12, 0, 1)));
        assert_eq!(Gfx::from_target_version(90010), Some(gfx(9, 0, 10)));
        assert_eq!(Gfx::from_target_version(0), None);
        assert_eq!(Gfx::from_override("11.0.0"), Some(gfx(11, 0, 0)));
        assert_eq!(Gfx::from_override(" 9.0.10 "), Some(gfx(9, 0, 10)));
        assert_eq!(Gfx::from_override("11.0"), None);
        assert_eq!(Gfx::from_override("11.0.0.1"), None);
        assert_eq!(Gfx::from_override("gfx1100"), None);
        assert_eq!(Gfx::from_name("gfx1100"), Some(gfx(11, 0, 0)));
        assert_eq!(Gfx::from_name("gfx90a:xnack-"), Some(gfx(9, 0, 10)));
        assert_eq!(Gfx::from_name("gfx942"), Some(gfx(9, 4, 2)));
        assert_eq!(Gfx::from_name("sm_86"), None);
        assert_eq!(gfx(12, 0, 1).to_string(), "gfx1201");
        assert_eq!(gfx(9, 0, 10).to_string(), "gfx90a");
        assert_eq!(gfx(10, 3, 6).to_string(), "gfx1036");
    }

    #[test]
    fn the_topology_lists_the_gpu_nodes_in_order_without_the_processor() {
        let sys = Sys::new(
            "nodes",
            &[(RDNA4, 77, Some("AMD Radeon RX 9070 XT")), (IGPU, 0, None)],
        );
        let found = cards(&sys.0);
        assert_eq!(found.len(), 2);
        assert_eq!((found[0].node, found[0].gfx), (1, gfx(12, 0, 1)));
        assert_eq!(found[0].unique_id, 77);
        assert_eq!(found[0].name.as_deref(), Some("AMD Radeon RX 9070 XT"));
        assert_eq!(
            (found[1].node, found[1].gfx, &found[1].name),
            (2, gfx(10, 3, 6), &None)
        );
        assert!(cards(&sys.0.join("absent")).is_empty());
    }

    #[test]
    fn an_override_from_another_generation_is_broken() {
        // The case this guard is for: a gfx1100's override on a gfx1201.
        let sys = Sys::new("leftover", &[(RDNA4, 0, Some("AMD Radeon RX 9070 XT"))]);
        let report = sys.check(&[(OVERRIDE, "11.0.0")]).unwrap();
        let broken = report.broken.as_deref().expect("broken");
        assert!(
            broken.contains("HSA_OVERRIDE_GFX_VERSION=11.0.0"),
            "{broken}"
        );
        assert!(broken.contains("a gfx1100"), "{broken}");
        assert!(
            broken.contains("gfx1201 (AMD Radeon RX 9070 XT)"),
            "{broken}"
        );
        assert!(broken.contains("12.x.x"), "{broken}");
        assert!(report.detail().starts_with(broken));
    }

    #[test]
    fn an_override_within_the_generation_is_what_it_is_for() {
        let sys = Sys::new("spoof", &[(110002, 0, None)]);
        let report = sys.check(&[(OVERRIDE, "11.0.0")]).unwrap();
        assert_eq!(report.broken, None);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert_eq!(
            report.summary,
            "gfx1102; run as gfx1100 (HSA_OVERRIDE_GFX_VERSION=11.0.0)"
        );
        let own = sys.check(&[(OVERRIDE, "11.0.2")]).unwrap();
        assert_eq!(own.broken, None);
        assert!(own.summary.ends_with("the card's own"), "{}", own.summary);
    }

    #[test]
    fn without_variables_the_cards_are_listed() {
        let sys = Sys::new("plain", &[(RDNA4, 0, None), (IGPU, 0, None)]);
        let report = sys.check(&[]).unwrap();
        assert_eq!(report.summary, "gfx1201, gfx1036");
        assert_eq!(report.broken, None);
        assert!(report.warnings.is_empty());
        assert_eq!(report.detail(), "gfx1201, gfx1036");
    }

    #[test]
    fn no_topology_is_nothing_to_compare() {
        let sys = Sys::new("empty", &[]);
        assert_eq!(sys.check(&[(OVERRIDE, "11.0.0")]), None);
    }

    #[test]
    fn the_card_that_matters_is_the_first_one_visible() {
        let sys = Sys::new("visible", &[(RDNA4, 0, None), (RDNA3, 0, None)]);
        // The gfx1100's override, the gfx1201 hidden: fine.
        let report = sys
            .check(&[(OVERRIDE, "11.0.0"), ("HIP_VISIBLE_DEVICES", "1")])
            .unwrap();
        assert_eq!(report.broken, None, "{}", report.detail());
        assert!(
            report.summary.starts_with("gfx1100; hidden: gfx1201"),
            "{}",
            report.summary
        );
        // The same through ROCr's own list, and through CUDA's name.
        for key in ["ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"] {
            let report = sys.check(&[(OVERRIDE, "11.0.0"), (key, "1")]).unwrap();
            assert_eq!(report.broken, None, "{key}");
        }
        // Both shown, the gfx1201 first: it is the one the separation runs on.
        let report = sys.check(&[(OVERRIDE, "11.0.0")]).unwrap();
        assert!(report.broken.is_some());
        // The gfx1100 first: it runs, and the gfx1201 is told about.
        let report = sys
            .check(&[(OVERRIDE, "11.0.0"), ("HIP_VISIBLE_DEVICES", "1,0")])
            .unwrap();
        assert_eq!(report.broken, None);
        assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
        assert!(
            report.warnings[0].contains("gfx1201"),
            "{:?}",
            report.warnings
        );
    }

    #[test]
    fn visible_devices_lists_end_at_what_names_no_card() {
        let sys = Sys::new("lists", &[(RDNA4, 0xabc, None), (RDNA3, 0, None)]);
        let shown = |vars: &[(&str, &str)]| sys.check(vars).unwrap().summary;
        // ROCr numbers among all, HIP among what ROCr left.
        assert_eq!(
            shown(&[
                ("ROCR_VISIBLE_DEVICES", "1,0"),
                ("HIP_VISIBLE_DEVICES", "1")
            ]),
            "gfx1201; hidden: gfx1100"
        );
        assert_eq!(
            shown(&[("HIP_VISIBLE_DEVICES", "-1")]),
            "no AMD card visible to ROCm; hidden: gfx1201, gfx1100"
        );
        assert_eq!(
            shown(&[("HIP_VISIBLE_DEVICES", "1,7,0")]),
            "gfx1100; hidden: gfx1201"
        );
        assert_eq!(
            shown(&[("ROCR_VISIBLE_DEVICES", "GPU-0000000000000abc")]),
            "gfx1201; hidden: gfx1100"
        );
        // HIP's own list wins over CUDA's; an empty one is no list.
        assert_eq!(
            shown(&[("HIP_VISIBLE_DEVICES", "0"), ("CUDA_VISIBLE_DEVICES", "1")]),
            "gfx1201; hidden: gfx1100"
        );
        assert_eq!(shown(&[("HIP_VISIBLE_DEVICES", " ")]), "gfx1201, gfx1100");
        // No card shown, nothing to break.
        let none = sys
            .check(&[(OVERRIDE, "11.0.0"), ("HIP_VISIBLE_DEVICES", "-1")])
            .unwrap();
        assert_eq!(none.broken, None);
    }

    #[test]
    fn an_override_rocm_cannot_read_is_a_warning() {
        let sys = Sys::new("garbled", &[(RDNA4, 0, None)]);
        let report = sys.check(&[(OVERRIDE, "gfx1100")]).unwrap();
        assert_eq!(report.broken, None);
        assert_eq!(report.warnings.len(), 1);
        assert!(
            report.warnings[0].contains("e.g. 12.0.1"),
            "{:?}",
            report.warnings
        );
    }

    #[test]
    fn a_torch_arch_the_cards_lack_is_a_warning() {
        let sys = Sys::new("arch", &[(RDNA4, 0, None)]);
        let report = sys.check(&[(ARCH, "gfx1100")]).unwrap();
        assert_eq!(report.broken, None);
        assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
        assert!(report.warnings[0].contains("PYTORCH_ROCM_ARCH=gfx1100"));
        assert!(
            report.warnings[0].contains("(gfx1201)"),
            "{:?}",
            report.warnings
        );
        // One of a list is enough.
        let report = sys.check(&[(ARCH, "gfx1100;gfx1201")]).unwrap();
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        // Both leftovers at once: the override breaks, the arch warns.
        let report = sys
            .check(&[(OVERRIDE, "11.0.0"), (ARCH, "gfx1100")])
            .unwrap();
        assert!(report.broken.is_some());
        assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
    }

    #[test]
    fn a_torch_arch_may_name_what_the_override_runs_the_card_as() {
        let sys = Sys::new("arch-spoof", &[(100301, 0, None)]);
        let report = sys
            .check(&[(OVERRIDE, "10.3.0"), (ARCH, "gfx1030")])
            .unwrap();
        assert_eq!(report.broken, None);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    }
}
