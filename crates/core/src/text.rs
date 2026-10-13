//! Two ways of flattening a string, both needed by more than one step.
//!
//! [`ascii`] because Dee cannot open a path that holds an accent — the job
//! directory is named after the film, so the film's accents go first — and
//! [`fold`] because a track title is written by whoever muxed the file:
//! « VFQ », « Québécois », « Audiodescription » and « AUDIO DESC. » all name
//! the same two things, and a match on them must not depend on the case or
//! on which of é and e the muxer typed.

/// Lower-cased and stripped of the accents a Latin-1 title carries, so a
/// substring search matches however the title was typed.
pub fn fold(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match unaccent(c) {
            Some(plain) => out.push_str(plain),
            None => out.extend(c.to_lowercase()),
        }
    }
    out.make_ascii_lowercase();
    out
}

/// What [`fold`] gives with every run of non-alphanumerics turned into one
/// space, and a space at each end: a haystack [`has_word`] can search for a
/// whole word. « AD » in `Ad Libitum` is a word, the `ad` of `ad hoc` is one
/// too, and neither is the `ad` inside `Roadside`.
pub fn words(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push(' ');
    for c in fold(text).chars() {
        if c.is_alphanumeric() {
            out.push(c);
        } else if !out.ends_with(' ') {
            out.push(' ');
        }
    }
    if !out.ends_with(' ') {
        out.push(' ');
    }
    out
}

/// Whether `text` holds `word` as a whole word, exactly as typed. For the
/// initialisms that only mean something in capitals: `AD` is an audio
/// description, `Ad Libitum` is a title and `ad hoc` is neither.
pub fn has_exact_word(text: &str, word: &str) -> bool {
    text.split(|c: char| !c.is_alphanumeric())
        .any(|w| w == word)
}

/// Whether a haystack from [`words`] holds `word` whole. `word` must
/// already be folded and lower-case.
pub fn has_word(haystack: &str, word: &str) -> bool {
    let mut needle = String::with_capacity(word.len() + 2);
    needle.push(' ');
    needle.push_str(word);
    needle.push(' ');
    haystack.contains(&needle)
}

/// A file name Dee can open: accents folded to their letter, and everything
/// else outside printable ASCII — or forbidden in a file name — replaced by
/// `_`. Case is kept: this names the job's directory after the film.
pub fn ascii(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut push = |c: char| {
        out.push(match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_ascii_graphic() || c == ' ' => c,
            _ => '_',
        })
    };
    for c in text.chars() {
        match unaccent(c) {
            Some(plain) => plain.chars().for_each(&mut push),
            None => push(c),
        }
    }
    out
}

/// The unaccented form of one letter, case kept; `None` when the character
/// carries no accent this table knows. Latin-1 and the two French ligatures
/// cover what a track title or a film name holds in practice.
fn unaccent(c: char) -> Option<&'static str> {
    Some(match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => "a",
        'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => "A",
        'ç' => "c",
        'Ç' => "C",
        'è' | 'é' | 'ê' | 'ë' => "e",
        'È' | 'É' | 'Ê' | 'Ë' => "E",
        'ì' | 'í' | 'î' | 'ï' => "i",
        'Ì' | 'Í' | 'Î' | 'Ï' => "I",
        'ñ' => "n",
        'Ñ' => "N",
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' => "o",
        'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' => "O",
        'ù' | 'ú' | 'û' | 'ü' => "u",
        'Ù' | 'Ú' | 'Û' | 'Ü' => "U",
        'ý' | 'ÿ' => "y",
        'Ý' => "Y",
        'æ' => "ae",
        'Æ' => "AE",
        'œ' => "oe",
        'Œ' => "OE",
        'ß' => "ss",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accents_fold_to_their_letter() {
        assert_eq!(fold("Québécois VFF"), "quebecois vff");
        assert_eq!(fold("CŒUR"), "coeur");
    }

    #[test]
    fn a_word_search_does_not_match_inside_a_word() {
        let title = words("Ad Libitum (VFF)");
        assert!(has_word(&title, "ad"));
        assert!(has_word(&title, "vff"));
        assert!(!has_word(&words("Roadside"), "ad"));
        assert!(!has_word(&title, "vfq"));
    }

    #[test]
    fn an_initialism_only_counts_in_the_case_it_was_typed() {
        assert!(has_exact_word("VFF AAC 2.0 (AD)", "AD"));
        assert!(!has_exact_word("Ad Libitum", "AD"));
        assert!(!has_exact_word("ad hoc", "AD"));
    }

    #[test]
    fn a_job_directory_keeps_the_case_and_loses_the_accents() {
        assert_eq!(ascii("Générique: Le/Début"), "Generique_ Le_Debut");
        assert!(ascii("日本語 Some Film").is_ascii());
    }
}
