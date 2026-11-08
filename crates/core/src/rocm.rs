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
//!
//! The variables are read as ROCm 7.0.2 reads them, or the check would not
//! be about the card ROCm uses: one node's `HSA_OVERRIDE_GFX_VERSION_<n>`
//! before the one for every card (libhsakmt's `topology.c`),
//! `ROCR_VISIBLE_DEVICES` as ROCr's `RvdFilter` does, then
//! `HIP_VISIBLE_DEVICES` as HIP's `rocdevice.cpp` does — the two differ.

use std::fmt;
use std::fs;
use std::path::Path;

/// The variable that makes ROCm run every card as another architecture.
pub const OVERRIDE: &str = "HSA_OVERRIDE_GFX_VERSION";
const ARCH: &str = "PYTORCH_ROCM_ARCH";

/// Whether `key` is an override ROCm reads: [`OVERRIDE`], or one node's
/// `HSA_OVERRIDE_GFX_VERSION_<n>`.
pub fn is_override(key: &str) -> bool {
    key == OVERRIDE
        || key
            .strip_prefix(OVERRIDE)
            .and_then(|rest| rest.strip_prefix('_'))
            .is_some_and(|node| !node.is_empty() && node.bytes().all(|b| b.is_ascii_digit()))
}

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

    /// From an override, as libhsakmt reads it (`"%u.%u.%u%c"` must find
    /// three numbers and nothing after): `major.minor.stepping` in decimal,
    /// the major at most 63, the others at most 255.
    fn from_override(value: &str) -> Option<Self> {
        let number = |part: &str| {
            let digits = part.trim_start();
            let digits = digits.strip_prefix('+').unwrap_or(digits);
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            digits.parse::<u32>().ok()
        };
        let mut parts = value.split('.');
        let gfx = Self {
            major: number(parts.next()?)?,
            minor: number(parts.next()?)?,
            stepping: number(parts.next()?)?,
        };
        (parts.next().is_none() && gfx.major <= 63 && gfx.minor <= 255 && gfx.stepping <= 255)
            .then_some(gfx)
    }

    /// As an override writes it: gfx1201 is `12.0.1`.
    fn version(self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.stepping)
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
    /// The node's number in the topology, which its own override is named
    /// after (`HSA_OVERRIDE_GFX_VERSION_1`). libhsakmt numbers only the
    /// nodes it can open, so the two differ only past a card whose render
    /// node this user cannot open — a card ROCm cannot use either.
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

    /// The card's id as ROCr matches `ROCR_VISIBLE_DEVICES` with it, in
    /// upper case; none for a card without one.
    fn rocr_uuid(&self) -> Option<String> {
        (self.unique_id != 0).then(|| format!("GPU-{:016X}", self.unique_id))
    }

    /// The card's id as HIP matches `HIP_VISIBLE_DEVICES` with it, in lower
    /// case; `GPU-XX` for a card without one.
    fn hip_uuid(&self) -> String {
        match self.unique_id {
            0 => "GPU-XX".to_string(),
            id => format!("GPU-{id:016x}"),
        }
    }
}

/// The override ROCm applies to `card`, with its name, as libhsakmt picks
/// it: the card's own node's, else the one for every card. Set but empty
/// still counts: ROCm reads it, and refuses it.
fn override_for(card: &Card, env: &dyn Fn(&str) -> Option<String>) -> Option<(String, String)> {
    let own = format!("{OVERRIDE}_{}", card.node);
    match env(&own) {
        Some(value) => Some((own, value)),
        None => env(OVERRIDE).map(|value| (OVERRIDE.to_string(), value)),
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
/// them, then `HIP_VISIBLE_DEVICES` — or `CUDA_VISIBLE_DEVICES` when it is
/// empty — among those. The first is the one the separation runs on.
fn visible<'a>(cards: &'a [Card], env: &dyn Fn(&str) -> Option<String>) -> Vec<&'a Card> {
    let mut shown: Vec<&Card> = cards.iter().collect();
    // ROCr applies its list as soon as it is set: empty, it shows no card.
    if let Some(list) = env("ROCR_VISIBLE_DEVICES") {
        shown = rocr_pick(&shown, &list);
    }
    let hip = env("HIP_VISIBLE_DEVICES")
        .filter(|list| !list.is_empty())
        .or_else(|| env("CUDA_VISIBLE_DEVICES"));
    if let Some(list) = hip.filter(|list| !list.is_empty()) {
        shown = hip_pick(&shown, &list);
    }
    shown
}

/// `ROCR_VISIBLE_DEVICES` as ROCr reads it: each entry trimmed and upper
/// case, an index in C's notation (`1`, `0x1`) or the start of one card's
/// id (`GPU-1234`; the start of two cards' ids names none). The list ends
/// at the first entry that names no card, or one already named.
fn rocr_pick<'a>(cards: &[&'a Card], list: &str) -> Vec<&'a Card> {
    let mut picked: Vec<&Card> = Vec::new();
    for token in list.split(',') {
        let token = token.trim().to_ascii_uppercase();
        let index = if token.starts_with('G') {
            rocr_uuid_index(cards, &token)
        } else {
            c_index(&token)
        };
        match index.and_then(|index| cards.get(index).copied()) {
            Some(card) if !picked.contains(&card) => picked.push(card),
            _ => break,
        }
    }
    picked
}

/// Which card's id `token` starts, of 5 to 20 characters; none when it
/// starts two.
fn rocr_uuid_index(cards: &[&Card], token: &str) -> Option<usize> {
    if !(5..=20).contains(&token.len()) {
        return None;
    }
    let mut found = cards
        .iter()
        .enumerate()
        .filter(|(_, card)| card.rocr_uuid().is_some_and(|id| id.starts_with(token)));
    let (index, _) = found.next()?;
    found.next().is_none().then_some(index)
}

/// An index as `strtol` reads a whole entry in base 0 — `12`, `0X0C`,
/// `014`, with a sign — none when it is negative or anything is left over.
fn c_index(token: &str) -> Option<usize> {
    let (negative, body) = match token.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, token.strip_prefix('+').unwrap_or(token)),
    };
    let (digits, radix) = if let Some(hex) = body.strip_prefix("0X") {
        (hex, 16)
    } else if body.len() > 1 && body.starts_with('0') {
        (&body[1..], 8)
    } else {
        (body, 10)
    };
    if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
        return None;
    }
    let index = usize::from_str_radix(digits, radix).ok()?;
    (!negative || index == 0).then_some(index)
}

/// `HIP_VISIBLE_DEVICES` as HIP reads it: each entry as written, not
/// trimmed. One with `GPU-` in it stands for the first card whose id holds
/// it (`GPU-12345678` for `GPU-123456789abcdef0`, in lower case); any other
/// must be an index written as one (`1`, not `01` nor ` 1`). The list ends
/// at the first entry that names no card; one naming a card already named
/// is skipped.
fn hip_pick<'a>(cards: &[&'a Card], list: &str) -> Vec<&'a Card> {
    let mut picked: Vec<&Card> = Vec::new();
    for token in list.split(',') {
        let index = if token.contains("GPU-") {
            cards
                .iter()
                .position(|card| card.hip_uuid().contains(token))
        } else {
            token
                .parse::<usize>()
                .ok()
                .filter(|index| index.to_string() == token)
        };
        let Some(card) = index.and_then(|index| cards.get(index).copied()) else {
            break;
        };
        if !picked.contains(&card) {
            picked.push(card);
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
    // libhsakmt reads every node's override, hidden or not, and one it
    // cannot read stops ROCm from starting on any card.
    let mut unread: Vec<String> = Vec::new();
    for card in &cards {
        if let Some((name, value)) = override_for(card, env)
            && Gfx::from_override(&value).is_none()
            && !unread.contains(&name)
        {
            report.warnings.push(format!(
                "{name}={value:?} is not a version ROCm reads (major.minor.stepping, e.g. {}): \
                 ROCm then starts on no card, and the separation runs on the processor, for hours",
                card.gfx.version()
            ));
            unread.push(name);
        }
    }

    // What ROCm runs the visible cards as, where their override fits them.
    let mut runs_as: Vec<Gfx> = Vec::new();
    for (index, card) in shown.iter().enumerate() {
        let Some((name, value)) = override_for(card, env) else {
            continue;
        };
        let Some(over) = Gfx::from_override(&value) else {
            continue;
        };
        if over.runs_on(card.gfx) {
            runs_as.push(over);
            if index == 0 {
                report.summary.push_str(&if over == card.gfx {
                    format!("; {name}={value}, the card's own")
                } else {
                    format!("; run as {over} ({name}={value})")
                });
            }
        } else if index == 0 {
            report.broken = Some(format!(
                "{name}={value} tells ROCm the card the separation runs on is a {over}, but it \
                 is a {}, of another generation: code for a {over} faults on it and the driver \
                 answers with a reset of the GPU that closes the desktop's applications. Unset \
                 {name} (likely left over from an earlier card, in ~/.profile, ~/.bashrc or \
                 ~/.zshrc) or set it to a {}.x.x version",
                card.describe(),
                card.gfx.major
            ));
        } else {
            // Nothing is sent to it, but it is handed code it cannot run.
            report.warnings.push(format!(
                "{name}={value} makes the {} a {over}, of another generation: nothing is sent \
                 to it, but hide it with HIP_VISIBLE_DEVICES",
                card.describe()
            ));
        }
    }

    // Which architectures ROCm code built here is for: one the cards do
    // not have is a leftover too, harmless to the separation.
    if let Some(raw) = env(ARCH).map(|v| v.trim().to_string())
        && !raw.is_empty()
        && !shown.is_empty()
    {
        let fits = |gfx: Gfx| shown.iter().any(|card| card.gfx == gfx) || runs_as.contains(&gfx);
        let named = raw
            .split([';', ',', ' '])
            .filter(|name| !name.is_empty())
            .any(|name| Gfx::from_name(name).is_some_and(fits));
        if !named {
            let mut have: Vec<String> = Vec::new();
            for gfx in shown
                .iter()
                .map(|card| card.gfx)
                .chain(runs_as.iter().copied())
            {
                if !have.contains(&gfx.to_string()) {
                    have.push(gfx.to_string());
                }
            }
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
        assert_eq!(Gfx::from_override(" 9.0.10"), Some(gfx(9, 0, 10)));
        // libhsakmt's `%u.%u.%u%c`: anything after the third number, a
        // space included, makes four matches and a refusal.
        assert_eq!(Gfx::from_override("9.0.10 "), None);
        assert_eq!(Gfx::from_override("11.0"), None);
        assert_eq!(Gfx::from_override("11.0.0.1"), None);
        assert_eq!(Gfx::from_override(""), None);
        assert_eq!(Gfx::from_override("gfx1100"), None);
        assert_eq!(Gfx::from_override("64.0.0"), None);
        assert_eq!(Gfx::from_override("11.0.256"), None);
        assert_eq!(gfx(12, 0, 1).version(), "12.0.1");
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
        // HIP's own list wins over CUDA's, unless it is empty; a blank one
        // is a list, of an entry that names no card.
        assert_eq!(
            shown(&[("HIP_VISIBLE_DEVICES", "0"), ("CUDA_VISIBLE_DEVICES", "1")]),
            "gfx1201; hidden: gfx1100"
        );
        assert_eq!(
            shown(&[("HIP_VISIBLE_DEVICES", ""), ("CUDA_VISIBLE_DEVICES", "1")]),
            "gfx1100; hidden: gfx1201"
        );
        assert_eq!(
            shown(&[("HIP_VISIBLE_DEVICES", " ")]),
            "no AMD card visible to ROCm; hidden: gfx1201, gfx1100"
        );
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

    #[test]
    fn a_nodes_own_override_comes_before_every_cards() {
        // The gfx1201 is node 1, the processor node 0.
        let sys = Sys::new("per-node", &[(RDNA4, 0, None)]);
        let own = "HSA_OVERRIDE_GFX_VERSION_1";
        // Its own fits it: what every card's says does not matter to it.
        let report = sys.check(&[(OVERRIDE, "11.0.0"), (own, "12.0.1")]).unwrap();
        assert_eq!(report.broken, None, "{}", report.detail());
        assert!(
            report
                .summary
                .ends_with("; HSA_OVERRIDE_GFX_VERSION_1=12.0.1, the card's own"),
            "{}",
            report.summary
        );
        // Its own is another generation's: broken, and said under its name.
        let report = sys.check(&[(OVERRIDE, "12.0.1"), (own, "11.0.0")]).unwrap();
        let broken = report.broken.as_deref().expect("broken");
        assert!(
            broken.starts_with("HSA_OVERRIDE_GFX_VERSION_1=11.0.0 tells ROCm"),
            "{broken}"
        );
        assert!(
            broken.contains("Unset HSA_OVERRIDE_GFX_VERSION_1"),
            "{broken}"
        );
        // Another node's is not this card's.
        let report = sys
            .check(&[("HSA_OVERRIDE_GFX_VERSION_2", "11.0.0")])
            .unwrap();
        assert_eq!(report.broken, None);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    }

    #[test]
    fn hip_takes_a_card_by_part_of_its_id() {
        let sys = Sys::new(
            "hip-uuid",
            &[(RDNA3, 0x1111, None), (RDNA4, 0x1234_5678_9abc_def0, None)],
        );
        // HIP shows the gfx1201 by the start of its id: the override is
        // broken on it, not hidden.
        let report = sys
            .check(&[
                (OVERRIDE, "11.0.0"),
                ("HIP_VISIBLE_DEVICES", "GPU-12345678"),
            ])
            .unwrap();
        assert_eq!(report.summary, "gfx1201; hidden: gfx1100");
        assert!(report.broken.is_some(), "{}", report.detail());
        let shown = |vars: &[(&str, &str)]| sys.check(vars).unwrap().summary;
        // Any part HIP finds in the id; the id in lower case, as HIP has it.
        assert_eq!(
            shown(&[("HIP_VISIBLE_DEVICES", "GPU-123456789abcdef0")]),
            "gfx1201; hidden: gfx1100"
        );
        assert_eq!(
            shown(&[("HIP_VISIBLE_DEVICES", "GPU-12345678ABC")]),
            "no AMD card visible to ROCm; hidden: gfx1100, gfx1201"
        );
        // The first card whose id holds it.
        assert_eq!(
            shown(&[("HIP_VISIBLE_DEVICES", "GPU-0")]),
            "gfx1100; hidden: gfx1201"
        );
        // A card without an id is `GPU-XX` to HIP.
        let bare = Sys::new("hip-xx", &[(RDNA3, 0x1111, None), (RDNA4, 0, None)]);
        assert_eq!(
            bare.check(&[("HIP_VISIBLE_DEVICES", "GPU-XX")])
                .unwrap()
                .summary,
            "gfx1201; hidden: gfx1100"
        );
    }

    #[test]
    fn hip_reads_entries_as_written_and_skips_a_card_named_twice() {
        let sys = Sys::new("hip-list", &[(RDNA4, 0, None), (RDNA3, 0, None)]);
        let shown = |list: &str| sys.check(&[("HIP_VISIBLE_DEVICES", list)]).unwrap().summary;
        assert_eq!(shown("0,0,1"), "gfx1201, gfx1100");
        assert_eq!(shown("0, 1"), "gfx1201; hidden: gfx1100");
        assert_eq!(
            shown("01"),
            "no AMD card visible to ROCm; hidden: gfx1201, gfx1100"
        );
        assert_eq!(
            shown("+1"),
            "no AMD card visible to ROCm; hidden: gfx1201, gfx1100"
        );
        assert_eq!(shown("1,"), "gfx1100; hidden: gfx1201");
    }

    #[test]
    fn rocr_reads_its_list_its_own_way() {
        let sys = Sys::new(
            "rocr-list",
            &[
                (RDNA4, 0x1234_5678_9abc_def0, None),
                (RDNA3, 0x1234_0000_0000_0001, None),
            ],
        );
        let shown = |list: &str| {
            sys.check(&[("ROCR_VISIBLE_DEVICES", list)])
                .unwrap()
                .summary
        };
        let none = "no AMD card visible to ROCm; hidden: gfx1201, gfx1100";
        // Set, even empty, it applies: no card.
        assert_eq!(shown(""), none);
        // Entries trimmed and in upper case; indices in C's notation.
        assert_eq!(shown(" 1 "), "gfx1100; hidden: gfx1201");
        assert_eq!(shown("0x1"), "gfx1100; hidden: gfx1201");
        assert_eq!(shown("gpu-12345678"), "gfx1201; hidden: gfx1100");
        // The start of two ids names neither; too short names none.
        assert_eq!(shown("GPU-1234"), none);
        assert_eq!(shown("GPU-"), none);
        assert_eq!(shown("GPU-12340"), "gfx1100; hidden: gfx1201");
        // A card named twice ends the list, where HIP would skip it.
        assert_eq!(shown("0,0,1"), "gfx1201; hidden: gfx1100");
        assert_eq!(shown("-1"), none);
    }

    #[test]
    fn an_override_rocm_refuses_on_any_card_is_said_once() {
        let sys = Sys::new("refused", &[(RDNA4, 0, None), (IGPU, 0, None)]);
        // Set but empty, as `export HSA_OVERRIDE_GFX_VERSION=` leaves it.
        let report = sys.check(&[(OVERRIDE, "")]).unwrap();
        assert_eq!(report.broken, None);
        assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
        assert!(
            report.warnings[0].starts_with("HSA_OVERRIDE_GFX_VERSION=\"\" is not"),
            "{:?}",
            report.warnings
        );
        assert!(
            report.warnings[0].contains("processor"),
            "{:?}",
            report.warnings
        );
        // A hidden card's own, which libhsakmt reads all the same.
        let report = sys
            .check(&[
                ("HSA_OVERRIDE_GFX_VERSION_2", "10.3.0 "),
                ("HIP_VISIBLE_DEVICES", "0"),
            ])
            .unwrap();
        assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
        assert!(
            report.warnings[0].starts_with("HSA_OVERRIDE_GFX_VERSION_2=\"10.3.0 \""),
            "{:?}",
            report.warnings
        );
    }

    #[test]
    fn the_override_variables_are_every_cards_and_each_nodes() {
        assert!(is_override("HSA_OVERRIDE_GFX_VERSION"));
        assert!(is_override("HSA_OVERRIDE_GFX_VERSION_1"));
        assert!(is_override("HSA_OVERRIDE_GFX_VERSION_12"));
        assert!(!is_override("HSA_OVERRIDE_GFX_VERSION_"));
        assert!(!is_override("HSA_OVERRIDE_GFX_VERSION_X"));
        assert!(!is_override("HSA_OVERRIDE_GFX_VERSIONS"));
        assert!(!is_override("PYTORCH_ROCM_ARCH"));
    }
}
